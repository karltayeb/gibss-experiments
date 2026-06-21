from __future__ import annotations

import numpy as np

from gseasusie.genesets import load_gene_sets


def msigdb_gene_sets_X(
    rng: np.random.Generator, *, collection: str
) -> np.ndarray:
    del rng
    gene_sets = load_gene_sets(source="msigdb", collection=collection)
    return gene_sets.to_sparse()


def hallmark_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    return msigdb_gene_sets_X(rng, collection="h.all")


def c4_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    return msigdb_gene_sets_X(rng, collection="c4.all")
