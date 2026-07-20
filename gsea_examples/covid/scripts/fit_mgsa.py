"""fit_mgsa: model-based gene set analysis (Bauer et al. 2010; Bioconductor `mgsa`).
A Bayesian multiset model over the binary top-fraction hit list, run once per
threshold (the SAME hit lists ORA/logistic use - built here in Python for
cross-method consistency). The MCMC core runs in mgsa.R; this shim loads the cached
design, hands R the membership + per-threshold hit lists, and writes the per-set
parquet. No jax.

One row per (threshold, gene set):
  method, threshold, set_id, set_size, signed_loading_mean,
  estimate (posterior P(set active)), std_error (null unless restarts>1),
  in_study_set, in_population, alpha_post/beta_post/p_post (posterior-mean global
  params), rank (by estimate within threshold).
"""
import os
import sys
import shutil
import tempfile

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

from _design import load_coo
from _rbridge import run_r
from scoring import select_hits

_R = os.path.join(os.path.dirname(__file__), "mgsa.R")


def run(design_path, sets_path, method, spec, thresholds, out_path, threads=1):
    d = load_coo(design_path, sets_path)
    score = d["score"]

    work = tempfile.mkdtemp(prefix="mgsa_")
    try:
        pl.DataFrame({
            "gene_idx": d["rows"].astype(np.int64),
            "set_idx": d["cols"].astype(np.int64),
        }).write_csv(os.path.join(work, "membership_long.tsv"), separator="\t")

        # top-fraction hit list per threshold (identical selection to fit_ora)
        obs = []
        for t in thresholds:
            hits = np.nonzero(select_hits(score, quantile=1.0 - t))[0]
            obs.append(pl.DataFrame({"threshold": np.full(len(hits), t), "gene_idx": hits.astype(np.int64)}))
        pl.concat(obs).write_csv(os.path.join(work, "obs.tsv"), separator="\t")

        params = {
            "n_genes": d["n_genes"],
            "steps": int(spec.get("steps", 1_000_000)),      # mgsa default
            "restarts": int(spec.get("restarts", 1)),        # >1 needed for std errors
            "thin": int(spec.get("thin", 100)),              # mgsa default
            "threads": int(threads),
            "seed": int(spec.get("seed", 0)),
        }
        pl.DataFrame({"key": list(params), "value": [str(v) for v in params.values()]}).write_csv(
            os.path.join(work, "params.tsv"), separator="\t")

        run_r(_R, work, threads=threads)
        res = pl.read_csv(os.path.join(work, "mgsa_out.tsv"), separator="\t",
                          null_values=["NA"], schema_overrides={"std_error": pl.Float64})
    finally:
        shutil.rmtree(work, ignore_errors=True)

    meta = pl.DataFrame({
        "set_idx": np.arange(d["n_sets"], dtype=np.int64),
        "set_id": d["set_ids"],
        "set_size": d["set_sizes"].astype(int),
        "signed_loading_mean": d["signed_loading_mean"].astype(float),
    })
    out = (
        res.join(meta, on="set_idx", how="left")
        .with_columns(method=pl.lit(method))
        .with_columns(rank=pl.col("estimate").rank(method="ordinal", descending=True).over("threshold"))
        .drop("set_idx")
    )
    out.write_parquet(out_path)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821
    run(smk.input.design, smk.input.sets, smk.wildcards.method,
        dict(smk.params.spec), list(smk.params.thresholds), smk.output[0], threads=int(smk.threads))
