"""Snakemake `script:` step: fit ONE method (cell) for one factor, from the
cached design. `spec` is the resolved config (model_defaults + the method's fields).

  family: logistic -> GeneList hit list at each threshold, fit_glm_susie (quad kernel)
          cox      -> GeneRanks, fit_cox_susie (method="poisson")
                        select: top_fraction -> censored at each threshold; else all events
                        order:  ascending    -> reversed ranking (rank the negated score)
          ora      -> Fisher exact on each threshold's hit list

SuSiE output is one row per (component, gene set): method, threshold, L, component,
set_id, set_size, signed_loading_mean, alpha, feature_log_bf, feature_log_marginal,
effect (mu; log-HR/log-odds), effect_var, prior_variance, ser_log_bf. ORA writes its
own Fisher columns instead.
"""
import os
import sys

# Cap threadpools to the cores SLURM actually allocated (snakemake.threads) BEFORE
# importing jax/numpy. By default the BLAS backends and XLA's Eigen pool size
# themselves to the whole node's core count, which oversubscribes a multi-job
# SLURM allocation and thrashes. OMP_NUM_THREADS also bounds XLA's CPU Eigen pool.
# (XLA_FLAGS is not used to cap threads here: XLA aborts on any unknown --xla_*
# flag and offers no stable intra-op-thread knob, so OMP is the portable cap.)
if "snakemake" in globals():
    _NT = str(max(1, int(snakemake.threads)))  # noqa: F821  (injected by Snakemake)
    os.environ.setdefault("OMP_NUM_THREADS", _NT)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
    os.environ.setdefault("MKL_NUM_THREADS", _NT)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", _NT)

import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

from gseasusie.gene_data import GeneList, GeneStandardizedEffects, GeneSetCollection
from gseasusie.fit import fit_ora
import gibss

from scoring import select_hits

# fit_glm_susie / fit_cox_susie kwargs we forward from the resolved spec.
_MODEL_KEYS = ("center", "offset_integration", "offset_quadrature_points", "intercept", "baseline", "time_bins")
_LOGISTIC_DROP = ("baseline", "time_bins")   # cox-only kwargs the glm door rejects
_COX_DROP = ("intercept",)                    # glm-only kwargs the cox door rejects


def _load_design(design_path, sets_path):
    d = np.load(design_path)
    n_genes, n_sets = int(d["n_genes"]), int(d["n_sets"])
    dense = np.zeros((n_genes, n_sets), dtype=np.float32)
    dense[d["rows"], d["cols"]] = 1.0
    X = sparse.BCOO.fromdense(jnp.asarray(dense))
    sets = pl.read_parquet(sets_path)
    return X, dense, d["score"], sets["set_id"].to_list(), sets["set_size"].to_numpy(), sets["signed_loading_mean"].to_numpy()


def _susie_rows(state, method, threshold, set_ids, set_sizes, signed):
    # One row per (SuSiE component, gene set). The per-component alpha and
    # feature_log_bf don't saturate the way the combined PIP does (which is
    # recoverable as 1 - prod_c(1 - alpha_c)); prior_variance / ser_log_bf are
    # per-component scalars repeated across the component's sets. L is the number
    # of components actually fit (= the discovered L for greedy L="auto").
    Lval = len(state.single_effects)
    prior_variance = np.asarray(state.prior_variance)     # (L,)
    ser_log_bf = np.asarray(state.ser_log_bf)             # (L,)
    rows = []
    for comp, effect in enumerate(state.single_effects, start=1):
        alpha = np.asarray(effect.alpha)
        flbf = np.asarray(effect.feature_log_bf)
        flm = np.asarray(effect.feature_log_marginal)
        mu = np.asarray(effect.mu)
        var = np.asarray(effect.var)
        for j, sid in enumerate(set_ids):
            rows.append({
                "method": method, "threshold": threshold, "L": Lval, "component": comp,
                "set_id": sid, "set_size": int(set_sizes[j]),
                "signed_loading_mean": float(signed[j]),
                "alpha": float(alpha[j]),
                "feature_log_bf": float(flbf[j]),
                "feature_log_marginal": float(flm[j]),
                "effect": float(mu[j]),
                "effect_var": float(var[j]),
                "prior_variance": float(prior_variance[comp - 1]),
                "ser_log_bf": float(ser_log_bf[comp - 1]),
            })
    return rows


def run_cell(design_path, sets_path, method, spec, thresholds, out_path):
    X, dense, score, set_ids, set_sizes, signed = _load_design(design_path, sets_path)
    gene_ids = [str(i) for i in range(len(score))]  # positional labels; fit is order-aligned to X
    family = spec["family"]

    model_kw = {k: spec[k] for k in _MODEL_KEYS if k in spec}
    for k in (_LOGISTIC_DROP if family == "logistic" else _COX_DROP):
        model_kw.pop(k, None)
    common = dict(
        L=spec.get("L"), estimate_prior_variance=spec.get("estimate_prior_variance", True),
        max_iter=spec.get("max_iter", 100),
        max_L=spec.get("max_L"), stride=spec.get("stride", 1),  # greedy (L="auto"); inert for fixed L
    )
    out = []

    if family == "logistic":
        for t in thresholds:
            y = jnp.asarray(select_hits(score, quantile=1.0 - t).astype(np.float32))
            state = gibss.fit_glm_susie(X, y, **common, **model_kw)
            out += _susie_rows(state, method, t, set_ids, set_sizes, signed)

    elif family == "cox":
        z = -score if spec.get("order") == "ascending" else score
        gse = GeneStandardizedEffects(gene_ids=gene_ids, z_scores=z)
        selects = thresholds if spec.get("select") == "top_fraction" else [None]
        for t in selects:
            gr = gse.to_gene_ranks(direction="positive") if t is None else gse.to_gene_ranks(top_fraction=t, direction="positive")
            state = gibss.fit_cox_susie(
                X, event_time=np.asarray(gr.rank, np.float32), event_type=np.asarray(gr.event_observed, np.float32),
                method="poisson", **common, **model_kw,
            )
            out += _susie_rows(state, method, t, set_ids, set_sizes, signed)

    elif family == "ora":
        coll = GeneSetCollection(gene_ids=gene_ids, set_ids=set_ids, membership=dense)
        for t in thresholds:
            gl = GeneList(gene_ids=gene_ids, included=select_hits(score, quantile=1.0 - t))
            ora = fit_ora(coll, gl, min_set_size=1)
            out += ora.with_columns(pl.lit(method).alias("method"), pl.lit(t).alias("threshold")).to_dicts()

    else:
        raise ValueError(f"unknown family: {family!r}")

    pl.from_dicts(out, infer_schema_length=None).write_parquet(out_path)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821  (injected by Snakemake)
    run_cell(
        smk.input.design, smk.input.sets, smk.wildcards.method,
        dict(smk.params.spec), list(smk.params.thresholds), smk.output[0],
    )
