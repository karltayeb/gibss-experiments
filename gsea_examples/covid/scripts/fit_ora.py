"""fit_ora: Fisher-exact over-representation of each gene set in the top-fraction
hit list, at each threshold. Writes the fit_ora Fisher columns (odds_ratio,
overlap_size, pvalue, log_or_haldane, padj, rank, ...) plus method + threshold.
No jax: this is a hypergeometric test, fast and light.
"""
import os
import sys

if "snakemake" in globals():
    _NT = str(max(1, int(snakemake.threads)))  # noqa: F821
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, _NT)

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

from gseasusie.gene_data import GeneList, GeneSetCollection
from gseasusie.fit import fit_ora
from scoring import select_hits
from _design import load_dense


def run(design_path, sets_path, method, thresholds, out_path):
    dense, score, _se, set_ids, set_sizes, signed = load_dense(design_path, sets_path)
    gene_ids = [str(i) for i in range(len(score))]
    coll = GeneSetCollection(gene_ids=gene_ids, set_ids=set_ids, membership=dense)
    out = []
    for t in thresholds:
        gl = GeneList(gene_ids=gene_ids, included=select_hits(score, quantile=1.0 - t))
        ora = fit_ora(coll, gl, min_set_size=1)
        out += ora.with_columns(pl.lit(method).alias("method"), pl.lit(t).alias("threshold")).to_dicts()
    pl.from_dicts(out, infer_schema_length=None).write_parquet(out_path)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821
    run(smk.input.design, smk.input.sets, smk.wildcards.method,
        list(smk.params.thresholds), smk.output[0])
