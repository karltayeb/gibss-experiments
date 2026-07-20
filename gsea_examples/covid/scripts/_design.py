"""Shared design loader for the covid fit_* rules (numpy/polars only; no jax, so
fit_ora / fit_gsea don't pay the jax import). fit_susie builds its BCOO from the
returned dense matrix."""
import numpy as np
import polars as pl


def load_dense(design_path, sets_path):
    d = np.load(design_path)
    n_genes, n_sets = int(d["n_genes"]), int(d["n_sets"])
    dense = np.zeros((n_genes, n_sets), dtype=np.float32)
    dense[d["rows"], d["cols"]] = 1.0
    se = d["se"] if "se" in d.files else None  # optional obs-variance weight (log2fc_weighted)
    sets = pl.read_parquet(sets_path)
    return (dense, d["score"], se, sets["set_id"].to_list(),
            sets["set_size"].to_numpy(), sets["signed_loading_mean"].to_numpy())


def load_coo(design_path, sets_path):
    """COO membership + per-gene score/effect + set metadata, without densifying.
    Used by the R-backed fits (fit_slpr / fit_mgsa) which stream the sparse
    membership to R rather than build a dense genes x sets matrix."""
    d = np.load(design_path)
    sets = pl.read_parquet(sets_path)
    return {
        "rows": np.asarray(d["rows"]), "cols": np.asarray(d["cols"]),
        "n_genes": int(d["n_genes"]), "n_sets": int(d["n_sets"]),
        "score": np.asarray(d["score"]),
        # gene_signed_effect is the SLPR response (log2FC); added by prep.py, so a
        # design cached before that change is missing it and must be regenerated.
        "gene_signed_effect": np.asarray(d["gene_signed_effect"]),
        "set_ids": sets["set_id"].to_list(),
        "set_sizes": sets["set_size"].to_numpy(),
        "signed_loading_mean": sets["signed_loading_mean"].to_numpy(),
    }
