"""fit_gsea: gseapy prerank on the per-gene ranking score, gene sets taken straight
from the design membership (same GMT as the SuSiE design). On a signed score the NES
sign is the direction (up vs down); on an abs score it is one-sided magnitude. No
thresholds. Writes: method, set_id, es, nes, nom_pval, fdr, fwer. No jax.
"""
import os
import sys

if "snakemake" in globals():
    _NT = str(max(1, int(snakemake.threads)))  # noqa: F821
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, _NT)

import numpy as np
import pandas as pd
import polars as pl
import gseapy as gp

sys.path.insert(0, os.path.dirname(__file__))

from _design import load_dense


def run(design_path, sets_path, method, spec, out_path):
    dense, score, _se, set_ids, set_sizes, signed = load_dense(design_path, sets_path)
    gene_ids = [str(i) for i in range(len(score))]
    rnk = pd.DataFrame({"gene": gene_ids, "score": np.asarray(score, dtype=float)})
    gene_sets = {sid: [gene_ids[g] for g in np.nonzero(dense[:, j])[0]] for j, sid in enumerate(set_ids)}
    res = gp.prerank(
        rnk=rnk, gene_sets=gene_sets, min_size=1, max_size=10 ** 9,
        permutation_num=int(spec.get("permutations", 1000)),
        threads=int(os.environ.get("OMP_NUM_THREADS", "4")),
        seed=int(spec.get("seed", 0)), no_plot=True, outdir=None, verbose=False,
    ).res2d
    out = [{
        "method": method, "set_id": r["Term"],
        "es": float(r["ES"]), "nes": float(r["NES"]),
        "nom_pval": float(r["NOM p-val"]), "fdr": float(r["FDR q-val"]),
        "fwer": float(r["FWER p-val"]),
    } for _, r in res.iterrows()]
    pl.from_dicts(out, infer_schema_length=None).write_parquet(out_path)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821
    run(smk.input.design, smk.input.sets, smk.wildcards.method,
        dict(smk.params.spec), smk.output[0])
