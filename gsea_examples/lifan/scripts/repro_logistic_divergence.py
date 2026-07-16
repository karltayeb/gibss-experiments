#!/usr/bin/env python
"""E2E reproducer: default logistic SuSiE diverges where localjj is stable.

Mirrors the real fit_gsea.py path (same factor TSV -> Entrez, same C2 alignment,
same top-fraction hit list) but fits ONE binary hit list at L=1 with both
`method="logistic"` (gibss front-door default) and `method="localjj"`, then
reports the per-set logBF / pip for the sets where the default blows up.

The default preset uses an unconstrained variational q with a Gauss-Hermite /
Newton inner solve on the plain Bernoulli likelihood. For a strongly enriched
large gene set at a strict hit list the single-effect MLE quasi-separates, the
inner Newton step runs away, and the returned per-set logBF collapses to a huge
negative constant (~ -1e5) with pip -> 0 -- a catastrophic FALSE NEGATIVE on
exactly the sets that are most enriched. localjj's Jaakkola-Jordan lower bound
is globally concave, so its fixed point is finite and the logBF stays sane.

Run from gsea_examples/lifan:  uv run python scripts/repro_logistic_divergence.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gseasusie.genesets import load_gene_sets
from gseasusie.gene_data import GeneStandardizedEffects, intersect_gene_ids

import gibss.methods as gmethods

from scoring import ranking_score, select_hits


def load_factor(factor_tsv, *, score_kind="abs_loading", lfsr_floor=1e-20, sd_floor=1e-4):
    df = pl.read_csv(factor_tsv, separator="\t")
    df = df.filter(pl.col("match_type") != "unmapped").with_columns(pl.col("entrez").cast(pl.Utf8))
    pm = df["posterior_mean"].to_numpy()
    sd = df["posterior_sd"].to_numpy()
    lfsr = df["lfsr"].to_numpy()
    score = ranking_score(pm, sd, lfsr, kind=score_kind, lfsr_floor=lfsr_floor, sd_floor=sd_floor)
    df = df.with_columns(pl.Series("score", score))
    df = df.sort("score", descending=True).unique(subset=["entrez"], keep="first", maintain_order=True)
    return df["entrez"].to_list(), df["score"].to_numpy()


def fit(X, y, method, *, L=1, epv=True, max_iter=100):
    res = gmethods.fit_glm_susie(
        X, jnp.asarray(y, dtype=jnp.float32), method=method, L=L,
        estimate_prior_variance=epv, max_iter=max_iter,
    )
    logbf = np.asarray(res.single_effects[0].feature_log_bf)
    pip = np.asarray(res.pip)
    return logbf, pip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-tsv", default="results/factors/factor_001.tsv")
    ap.add_argument("--collection", default="c2.all")
    ap.add_argument("--min-set-size", type=int, default=15)
    ap.add_argument("--max-set-size", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.05, help="top fraction = hits")
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args()

    entrez, score = load_factor(args.factor_tsv)
    collection = load_gene_sets(source="msigdb", collection=args.collection)

    gse = GeneStandardizedEffects(gene_ids=entrez, z_scores=score)
    shared = intersect_gene_ids(gse, collection)
    coll = collection.reindex_to_gene_ids(shared).filter_sets(
        min_size=args.min_set_size, max_size=args.max_set_size
    )
    set_ids = list(coll.set_ids)
    set_sizes = np.asarray(coll.set_sizes, dtype=int)
    X = sparse.BCOO.fromdense(jnp.asarray(np.asarray(coll.membership, dtype=np.float32)))

    pos = {g: i for i, g in enumerate(entrez)}
    keep = [pos[g] for g in shared]
    score_al = score[keep]
    y = select_hits(score_al, quantile=1.0 - args.threshold)

    n_universe = len(shared)
    n_hits = int(y.sum())
    # overlap of each set with the hit list -- the "enrichment" driver
    overlap = np.asarray(X.T @ jnp.asarray(y.astype(np.float32)))

    print(f"factor_tsv     : {args.factor_tsv}")
    print(f"universe genes : {n_universe}")
    print(f"gene sets      : {len(set_ids)}")
    print(f"threshold      : top {args.threshold:.0%}  ->  {n_hits} hits "
          f"({n_hits / n_universe:.1%} of universe)")
    print()

    lb_log, pip_log = fit(X, y, "logistic", max_iter=args.max_iter)
    lb_jj, pip_jj = fit(X, y, "localjj", max_iter=args.max_iter)

    df = pl.DataFrame({
        "set_id": set_ids,
        "size": set_sizes,
        "overlap": overlap.astype(int),
        "frac_hit": (overlap / np.maximum(set_sizes, 1)).round(3),
        "logbf_logistic": lb_log.round(2),
        "logbf_localjj": lb_jj.round(2),
        "pip_logistic": pip_log,
        "pip_localjj": pip_jj,
    })

    n_blown = int((lb_log < -1e3).sum())
    print(f"sets with logistic logBF < -1000 (catastrophic) : {n_blown} / {len(set_ids)}")
    print(f"min logistic logBF : {lb_log.min():,.1f}    min localjj logBF : {lb_jj.min():,.1f}")
    print()

    pl.Config.set_tbl_rows(25)
    pl.Config.set_tbl_width_chars(200)

    print("=== worst logistic blow-ups (most enriched sets the default MISSES) ===")
    worst = df.sort("logbf_logistic").head(12)
    print(worst)
    print()

    print("=== top sets by localjj logBF (the true winners) and what logistic said ===")
    top = df.sort("logbf_localjj", descending=True).head(12)
    print(top)


if __name__ == "__main__":
    main()
