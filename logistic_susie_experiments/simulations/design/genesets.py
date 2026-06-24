from __future__ import annotations

from functools import lru_cache

import numpy as np

from gseasusie.genesets import load_gene_sets


@lru_cache(maxsize=None)
def _load_sparse_collection(collection: str):
    """Load and sparsify a gene-set matrix once per process.

    The matrix is deterministic (no rng dependence), so caching avoids the
    multi-second reload that otherwise repeats for every replicate in a batch.
    The returned matrix is treated as read-only by simulate.
    """
    return load_gene_sets(source="msigdb", collection=collection).to_sparse()


def msigdb_gene_sets_X(
    rng: np.random.Generator, *, collection: str
) -> np.ndarray:
    del rng
    return _load_sparse_collection(collection)


def hallmark_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    return msigdb_gene_sets_X(rng, collection="h.all")


def c4_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    return msigdb_gene_sets_X(rng, collection="c4.all")
