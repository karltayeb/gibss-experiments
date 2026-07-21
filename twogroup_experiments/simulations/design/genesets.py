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


@lru_cache(maxsize=None)
def _load_sparse_subset(collection: str, prefix: str, min_size: int, max_size: int):
    """Sparse membership for a name-prefixed, size-filtered slice of a collection.

    Used for GO:BP (``GOBP_``-prefixed sets of ``c5.all``) restricted to a
    biologically usable size window. Cached like ``_load_sparse_collection`` so
    the reload+subset cost is paid once per (collection, prefix, size) key.
    """
    coll = load_gene_sets(source="msigdb", collection=collection)
    if prefix:
        coll = coll.subset_sets(set_id_prefixes=[prefix])
    coll = coll.filter_sets(min_size=int(min_size), max_size=int(max_size))
    return coll.to_sparse()


def msigdb_gene_sets_X(
    rng: np.random.Generator, *, collection: str
) -> np.ndarray:
    del rng
    return _load_sparse_collection(collection)


def gobp_gene_sets_X(
    rng: np.random.Generator, *, min_size: int = 10, max_size: int = 500
) -> np.ndarray:
    """GO:BP membership matrix (GOBP_ sets of c5.all) restricted to [min_size, max_size]."""
    del rng
    return _load_sparse_subset("c5.all", "GOBP_", min_size, max_size)


def hallmark_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    return msigdb_gene_sets_X(rng, collection="h.all")


def c4_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    return msigdb_gene_sets_X(rng, collection="c4.all")
