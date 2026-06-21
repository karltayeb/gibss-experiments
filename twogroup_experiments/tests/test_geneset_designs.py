from __future__ import annotations

import numpy as np

import core
from experiments import loader
from simulations.design.genesets import c4_gene_sets_X, hallmark_gene_sets_X


MSIGDB_COLLECTIONS = (
    "c1.all",
    "c2.all",
    "c3.all",
    "c4.all",
    "c5.all",
    "c6.all",
    "c7.all",
    "c8.all",
    "c9.all",
    "h.all",
    "msigdb",
)


def test_core_reexports_generic_msigdb_gene_set_loader():
    assert core.msigdb_gene_sets_X.__name__ == "msigdb_gene_sets_X"


def test_msigdb_gene_sets_loader_accepts_collection_kwarg():
    rng = np.random.default_rng(1)

    X = core.msigdb_gene_sets_X(rng, collection="c9.all")

    assert X.shape == (3396, 62)
    assert X.nse > 0


def test_existing_gene_set_wrappers_match_generic_loader():
    rng = np.random.default_rng(1)

    hallmark = hallmark_gene_sets_X(rng)
    c4 = c4_gene_sets_X(rng)

    assert hallmark.shape == core.msigdb_gene_sets_X(rng, collection="h.all").shape
    assert c4.shape == core.msigdb_gene_sets_X(rng, collection="c4.all").shape


def test_library_exposes_all_msigdb_collections_as_designs():
    library = loader.load_library()

    for collection in MSIGDB_COLLECTIONS:
        design_name = f"msigdb_{collection.replace('.', '_')}"
        assert library["designs"][design_name] == {
            "function": "msigdb_gene_sets_X",
            "arguments": {"collection": collection},
        }
