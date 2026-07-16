"""Snakemake `script:` step: align a factor to the collection once and cache the
sparse design, so the per-method fit jobs don't each re-align + re-materialize it.

Outputs:
  {factor}_design.npz   COO of the aligned membership + the aligned per-gene score
  {factor}_sets.parquet set_id, set_size, signed_loading_mean (per retained set)
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

from gseasusie.genesets import load_gene_sets
from gseasusie.gene_data import GeneStandardizedEffects, intersect_gene_ids

from scoring import ranking_score


def prep_factor(factor_tsv, collection, *, min_set_size, max_set_size, score_kind, lfsr_floor, sd_floor):
    df = (
        pl.read_csv(factor_tsv, separator="\t")
        .filter(pl.col("match_type") != "unmapped")
        .with_columns(pl.col("entrez").cast(pl.Utf8))
    )
    score = ranking_score(
        df["posterior_mean"].to_numpy(), df["posterior_sd"].to_numpy(), df["lfsr"].to_numpy(),
        kind=score_kind, lfsr_floor=lfsr_floor, sd_floor=sd_floor,
    )
    df = df.with_columns(pl.Series("score", score), pl.Series("pm", df["posterior_mean"].to_numpy()))
    # collapse duplicate Entrez ids, keeping the most significant
    df = df.sort("score", descending=True).unique(subset=["entrez"], keep="first", maintain_order=True)
    entrez = df["entrez"].to_list()
    score = df["score"].to_numpy()
    pm = df["pm"].to_numpy()

    coll = load_gene_sets(source="msigdb", collection=collection)
    shared = intersect_gene_ids(GeneStandardizedEffects(gene_ids=entrez, z_scores=score), coll)
    collf = coll.reindex_to_gene_ids(shared).filter_sets(min_size=min_set_size, max_size=max_set_size)
    membership = np.asarray(collf.membership, dtype=np.float32)  # genes x sets

    pos = {g: i for i, g in enumerate(entrez)}
    keep = [pos[g] for g in shared]
    score_al = score[keep]
    pm_al = pm[keep]
    set_sizes = membership.sum(axis=0).astype(int)
    signed = (membership.T @ pm_al) / np.maximum(set_sizes, 1)
    rows, cols = np.nonzero(membership)
    return {
        "rows": rows.astype(np.int32), "cols": cols.astype(np.int32),
        "n_genes": len(shared), "n_sets": len(collf.set_ids), "score": score_al,
        "set_ids": list(collf.set_ids), "set_sizes": set_sizes, "signed": signed,
    }


def save_prep(prep, design_path, sets_path):
    np.savez_compressed(
        design_path, rows=prep["rows"], cols=prep["cols"],
        n_genes=prep["n_genes"], n_sets=prep["n_sets"], score=prep["score"],
    )
    pl.DataFrame({
        "set_id": prep["set_ids"], "set_size": prep["set_sizes"],
        "signed_loading_mean": prep["signed"],
    }).write_parquet(sets_path)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821  (injected by Snakemake)
    p = smk.params
    save_prep(
        prep_factor(
            smk.input.factor_tsv, p.collection,
            min_set_size=p.min_set_size, max_set_size=p.max_set_size,
            score_kind=p.score, lfsr_floor=float(p.lfsr_floor), sd_floor=float(p.sd_floor),
        ),
        smk.output.design, smk.output.sets,
    )
