from __future__ import annotations

import numpy as np

from gseasusie.genesets import load_gene_sets


def hallmark_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    del rng
    gene_sets = load_gene_sets(source="msigdb", collection="h.all")
    return gene_sets.to_sparse()


def c4_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    del rng
    gene_sets = load_gene_sets(source="msigdb", collection="c4.all")
    return gene_sets.to_sparse()
