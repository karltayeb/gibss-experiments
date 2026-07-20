"""fit_susie: SuSiE fit (logistic hit-list quad kernel, or Cox poisson) for one
method, from the cached design. One row per (component, gene set):
method, threshold, L, component, set_id, set_size, signed_loading_mean, alpha,
feature_log_bf, feature_log_marginal, effect (mu; log-HR/log-odds), effect_var,
prior_variance, ser_log_bf.
"""
import os
import sys

# Cap threadpools to the SLURM-allocated cores BEFORE importing jax/numpy (the BLAS
# backends and XLA's Eigen pool otherwise size to the whole node and thrash a
# multi-job allocation). OMP_NUM_THREADS also bounds XLA's CPU Eigen pool.
if "snakemake" in globals():
    _NT = str(max(1, int(snakemake.threads)))  # noqa: F821  (injected by Snakemake)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, _NT)

import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

import gibss
from gseasusie.gene_data import GeneStandardizedEffects
from scoring import select_hits
from _design import load_dense

# fit_glm_susie / fit_cox_susie / fit_linear_susie kwargs forwarded from the spec.
# Each front door only accepts a subset, so drop the ones it would reject.
_MODEL_KEYS = ("center", "offset_integration", "offset_quadrature_points", "intercept", "baseline", "time_bins")
_FAMILY_DROP = {
    "logistic": ("baseline", "time_bins"),                                        # cox-only
    "cox": ("intercept",),                                                        # glm-only
    "linear": ("offset_integration", "offset_quadrature_points", "intercept", "baseline", "time_bins"),
}


def _susie_rows(state, method, threshold, set_ids, set_sizes, signed):
    # per-component alpha / feature_log_bf don't saturate the way the combined PIP
    # does; prior_variance / ser_log_bf are per-component scalars repeated over sets.
    Lval = len(state.single_effects)
    prior_variance = np.asarray(state.prior_variance)
    ser_log_bf = np.asarray(state.ser_log_bf)
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


def run(design_path, sets_path, method, spec, thresholds, out_path):
    dense, score, se, set_ids, set_sizes, signed = load_dense(design_path, sets_path)
    X = sparse.BCOO.fromdense(jnp.asarray(dense))
    gene_ids = [str(i) for i in range(len(score))]  # positional; fit is order-aligned to X
    family = spec["family"]

    model_kw = {k: spec[k] for k in _MODEL_KEYS if k in spec}
    for k in _FAMILY_DROP[family]:
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

    elif family == "linear":
        # regress the continuous per-gene score (log2fc) on the membership design (no
        # hit-list threshold). If the design carries per-gene SEs (log2fc_weighted),
        # pass them as observation variance for a measurement-error-weighted fit.
        y = jnp.asarray(np.asarray(score, dtype=np.float32))
        lin_kw = dict(common, **model_kw)
        if se is not None:
            lin_kw["obs_variance"] = jnp.asarray(np.asarray(se, dtype=np.float32) ** 2)
        state = gibss.fit_linear_susie(X, y, **lin_kw)
        out += _susie_rows(state, method, None, set_ids, set_sizes, signed)

    else:
        raise ValueError(f"fit_susie: non-SuSiE family {family!r}")

    pl.from_dicts(out, infer_schema_length=None).write_parquet(out_path)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821
    run(smk.input.design, smk.input.sets, smk.wildcards.method,
        dict(smk.params.spec), list(smk.params.thresholds), smk.output[0])
