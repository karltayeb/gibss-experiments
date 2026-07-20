"""Shared `prep` step: align one analysis unit's per-gene scores to the MSigDB
collection once and cache the sparse design, so the per-method `fit` jobs don't
each re-align + re-materialize it.

This step is example-agnostic. Everything upstream (reading the DE / topic-model
source, mapping ids to Entrez, and defining the per-gene ranking `score`) lives in
each example's bespoke `convert` step, which emits a tidy per-unit TSV with columns:

  entrez         gene id (Entrez; the id space MSigDB collections use)
  score          per-gene ranking statistic, higher = more interesting
  signed_effect  signed per-gene effect (log2FC / topic loading), for set-level
                 direction annotation only - it does not affect the hit list

Outputs (the design-cache contract consumed by fit_susie/fit_ora/fit_gsea, identical to lifan's):
  {unit}_design.npz    COO of the aligned membership + the aligned per-gene score
  {unit}_sets.parquet  set_id, set_size, signed_loading_mean (mean signed_effect per set)
"""
import numpy as np
import polars as pl

from gseasusie.genesets import load_gene_sets
from gseasusie.gene_data import GeneStandardizedEffects, intersect_gene_ids


def prep_unit(unit_tsv, collection, *, prefix=None, min_set_size, max_set_size):
    df = (
        pl.read_csv(unit_tsv, separator="\t")
        .with_columns(pl.col("entrez").cast(pl.Utf8))
        .drop_nulls("entrez")
    )
    # collapse duplicate Entrez ids (many-symbols-to-one-gene), keeping the most
    # significant, so each gene enters the design exactly once.
    df = df.sort("score", descending=True).unique(subset=["entrez"], keep="first", maintain_order=True)
    entrez = df["entrez"].to_list()
    score = df["score"].to_numpy()
    signed_effect = df["signed_effect"].to_numpy()
    se = df["se"].to_numpy() if "se" in df.columns else None  # optional obs-variance weight

    coll = load_gene_sets(source="msigdb", collection=collection)
    if prefix:  # e.g. GOBP_ to take GO:BP out of the combined C5 GMT
        coll = coll.subset_sets(set_ids=[s for s in coll.set_ids if s.startswith(prefix)])
    shared = intersect_gene_ids(GeneStandardizedEffects(gene_ids=entrez, z_scores=score), coll)
    collf = coll.reindex_to_gene_ids(shared).filter_sets(min_size=min_set_size, max_size=max_set_size)
    membership = np.asarray(collf.membership, dtype=np.float32)  # genes x sets

    pos = {g: i for i, g in enumerate(entrez)}
    keep = [pos[g] for g in shared]
    score_al = score[keep]
    eff_al = signed_effect[keep]
    set_sizes = membership.sum(axis=0).astype(int)
    # signed_loading_mean: mean signed per-gene effect over each set's members
    # (annotation for reading a credible set's direction; not used in fitting).
    signed = (membership.T @ eff_al) / np.maximum(set_sizes, 1)
    rows, cols = np.nonzero(membership)
    return {
        "rows": rows.astype(np.int32), "cols": cols.astype(np.int32),
        "n_genes": len(shared), "n_sets": len(collf.set_ids), "score": score_al,
        "gene_signed_effect": eff_al,  # per-gene signed effect (log2FC), SLPR response
        "se": (se[keep] if se is not None else None),  # per-gene SE (log2fc_weighted)
        "set_ids": list(collf.set_ids), "set_sizes": set_sizes, "signed": signed,
    }


def save_prep(prep, design_path, sets_path):
    arrs = dict(
        rows=prep["rows"], cols=prep["cols"],
        n_genes=prep["n_genes"], n_sets=prep["n_sets"], score=prep["score"],
        gene_signed_effect=prep["gene_signed_effect"],
    )
    if prep.get("se") is not None:
        arrs["se"] = prep["se"]
    np.savez_compressed(design_path, **arrs)
    pl.DataFrame({
        "set_id": prep["set_ids"], "set_size": prep["set_sizes"],
        "signed_loading_mean": prep["signed"],
    }).write_parquet(sets_path)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821  (injected by Snakemake)
    p = smk.params
    save_prep(
        prep_unit(
            smk.input.unit_tsv, p.collection, prefix=p.prefix,
            min_set_size=p.min_set_size, max_set_size=p.max_set_size,
        ),
        smk.output.design, smk.output.sets,
    )
