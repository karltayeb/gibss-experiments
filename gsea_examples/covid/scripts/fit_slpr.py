"""fit_slpr: SLPR (Frost & Amos 2017) - two-stage LASSO gene set selection. The
per-gene EFFECT SIZE (log2FC, cached by prep as gene_signed_effect) is regressed on
the genes x sets membership under a LASSO penalty (glmnet), then refit unpenalized
(OLS) on the selected sets. Effect sizes - not z/t-stats - are the recommended
response (paper SI 1.2.1). The glmnet/OLS core runs in slpr.R; this shim loads the
cached design, hands it to R via a temp dir, and writes the per-set parquet.

No jax. One row per gene set:
  method, set_id, set_size, signed_loading_mean, coef_lasso (shrunken beta),
  coef_ols (unshrunken; 0 if not selected), ols_pvalue (non-inferential; null if
  not selected), selected, direction (sign of coef_lasso), rank (by |coef_lasso|
  among selected; null otherwise), lambda, n_selected.
"""
import os
import sys
import tempfile

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

from _design import load_coo
from _rbridge import write_mm, run_r

_R = os.path.join(os.path.dirname(__file__), "slpr.R")


def run(design_path, sets_path, method, spec, out_path, threads=1):
    d = load_coo(design_path, sets_path)
    n_genes, n_sets = d["n_genes"], d["n_sets"]

    work = tempfile.mkdtemp(prefix="slpr_")
    try:
        write_mm(os.path.join(work, "membership.mtx"), d["rows"], d["cols"], n_genes, n_sets)
        pl.DataFrame({"y": np.asarray(d["gene_signed_effect"], dtype=float)}).write_csv(
            os.path.join(work, "response.tsv"), separator="\t")
        params = {
            "abs_response": 1 if spec.get("abs_response", False) else 0,
            "alpha": spec.get("alpha", 1.0),          # 1 = pure LASSO
            "nfolds": spec.get("nfolds", 10),
            "num_cv_iter": spec.get("num_cv_iter", 1),
            "cv_criteria": spec.get("cv_criteria", "lambda.min"),
            "seed": spec.get("seed", 0),
        }
        pl.DataFrame({"key": list(params), "value": [str(v) for v in params.values()]}).write_csv(
            os.path.join(work, "params.tsv"), separator="\t")

        run_r(_R, work, threads=threads)
        res = pl.read_csv(os.path.join(work, "slpr_out.tsv"), separator="\t",
                          null_values=["NA"], schema_overrides={"ols_pvalue": pl.Float64})
    finally:
        __import__("shutil").rmtree(work, ignore_errors=True)

    res = res.sort("set_idx")
    out = pl.DataFrame({
        "method": method,
        "set_id": d["set_ids"],
        "set_size": d["set_sizes"].astype(int),
        "signed_loading_mean": d["signed_loading_mean"].astype(float),
        "coef_lasso": res["coef_lasso"],
        "coef_ols": res["coef_ols"],
        "ols_pvalue": res["ols_pvalue"],
        "selected": res["selected"].cast(pl.Boolean),
        "lambda": res["lambda"],
        "n_selected": res["n_selected"],
    }).with_columns(
        direction=pl.col("coef_lasso").sign().cast(pl.Int8),
        # rank selected sets by |shrunken coef| (paper's default ranking); null otherwise
        rank=pl.when(pl.col("selected"))
              .then(pl.col("coef_lasso").abs().rank(method="ordinal", descending=True))
              .otherwise(None),
    )
    out.write_parquet(out_path)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821
    run(smk.input.design, smk.input.sets, smk.wildcards.method,
        dict(smk.params.spec), smk.output[0], threads=int(smk.threads))
