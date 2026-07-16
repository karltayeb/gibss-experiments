#!/usr/bin/env python
"""L=1 threshold sweep comparing logistic / cox / cox_reversed / ORA.

For a single factor, sweep the selection fraction f (top-f most significant genes):
  - logistic(f)  binary hit list = top f
  - ora(f)       same hit list, Fisher exact
  - cox(f)       right-censored: top f are events (ranked), rest censored
And, fraction-independent (full ranked list, no censoring):
  - cox_full     genes ranked most -> least significant
  - cox_reversed genes ranked least -> most significant

At L=1 the SER exposes a per-set logBF (feature_log_evidence, up to an additive
constant), which is the right cross-method quantity to compare because alpha/PIP
saturates. Output: one long parquet with method, fraction, set_id, logbf, pip.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from gseasusie.gene_data import GeneList, GeneStandardizedEffects
from gseasusie.genesets import load_gene_sets
from gseasusie.fit import fit_gsea_susie_logistic, fit_gsea_susie_cox, fit_ora

from fit_gsea import _load_factor, _signed_loading_by_set
from scoring import select_hits

L = 1


def _logbf_rows(result, method: str, fraction, signed_map) -> list[dict]:
    fe = np.asarray(result.model.single_effects[0].feature_log_evidence, dtype=float)
    pip = np.asarray(result.model.pip, dtype=float)
    sizes = dict(zip(result.results["set_id"].to_list(), result.results["set_size"].to_list()))
    return [
        {
            "method": method,
            "fraction": fraction,
            "set_id": sid,
            "set_size": int(sizes[sid]),
            "signed_loading_mean": signed_map.get(sid),
            "logbf": float(fe[j]),
            "pip": float(pip[j]),
        }
        for j, sid in enumerate(result.set_ids)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True)
    ap.add_argument("--factor-tsv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--min-set-size", type=int, required=True)
    ap.add_argument("--max-set-size", type=int, required=True)
    ap.add_argument("--lfsr-floor", type=float, required=True)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--fractions", default="0.01,0.02,0.05,0.1,0.2,0.35,0.5")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fractions = [float(x) for x in args.fractions.split(",")]

    entrez, score, posterior_mean = _load_factor(
        args.factor_tsv, score_kind="neglog_lfsr", lfsr_floor=args.lfsr_floor, sd_floor=1e-4
    )
    collection = load_gene_sets(source="msigdb", collection=args.collection)
    signed_map = _signed_loading_by_set(
        GeneStandardizedEffects(gene_ids=entrez, z_scores=score),
        dict(zip(entrez, posterior_mean)),
        collection,
    )
    fit_kwargs = dict(
        min_set_size=args.min_set_size,
        max_set_size=args.max_set_size,
        L=L,
        max_iter=args.max_iter,
    )
    gse = GeneStandardizedEffects(gene_ids=entrez, z_scores=score)

    rows: list[dict] = []
    ora_rows: list[dict] = []
    for f in fractions:
        mask = select_hits(score, quantile=1.0 - f)
        gene_list = GeneList(gene_ids=entrez, included=mask)
        log_fit = fit_gsea_susie_logistic(collection, gene_list, **fit_kwargs)
        rows += _logbf_rows(log_fit, "logistic", f, signed_map)

        ranks_cens = gse.to_gene_ranks(top_fraction=f, direction="positive")
        cox_fit = fit_gsea_susie_cox(collection, ranks_cens, **fit_kwargs)
        rows += _logbf_rows(cox_fit, "cox", f, signed_map)

        ora = fit_ora(
            collection, gene_list,
            min_set_size=args.min_set_size, max_set_size=args.max_set_size,
        ).with_columns(pl.lit(f).alias("fraction"))
        ora_rows += ora.to_dicts()
        print(f"[{args.factor}] f={f} n_hits={int(mask.sum())}")

    # Fraction-independent references.
    cox_full = fit_gsea_susie_cox(
        collection, gse.to_gene_ranks(direction="positive"), **fit_kwargs
    )
    rows += _logbf_rows(cox_full, "cox_full", None, signed_map)
    cox_rev = fit_gsea_susie_cox(
        collection,
        GeneStandardizedEffects(gene_ids=entrez, z_scores=-score).to_gene_ranks(
            direction="positive"
        ),
        **fit_kwargs,
    )
    rows += _logbf_rows(cox_rev, "cox_reversed", None, signed_map)

    pl.from_dicts(rows).write_parquet(outdir / "sweep_logbf.parquet")
    pl.from_dicts(ora_rows).write_parquet(outdir / "sweep_ora.parquet")
    print(f"[{args.factor}] wrote sweep ({len(rows)} logbf rows)")


if __name__ == "__main__":
    main()
