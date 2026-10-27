"""Aggregations over the pipeline's long tables (pip, cs, fit), one row per (cell, arm, L, rep).

Headline metrics follow the 019/022 results notebooks: credible-set coverage / power /
size on declared sets, PIP calibration and area under the precision-recall curve, and
effect recovery at the causal sites for the arms that share the truth's scale.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["declared", "cs_summary", "pip_calibration", "pip_auprc", "dispersion_summary", "timing_summary"]

SCENARIO = ("n", "mu0", "phi", "beta_abs", "k_causal", "libsize_sd")


def declared(cs: pl.DataFrame, *, min_log_bf: float = 2.0) -> pl.DataFrame:
    """Components that count as discoveries: ser_log_bf >= min_log_bf."""
    return cs.filter(pl.col("ser_log_bf") >= min_log_bf)


def cs_summary(cs: pl.DataFrame, fit: pl.DataFrame, *, by=SCENARIO, min_log_bf: float = 2.0) -> pl.DataFrame:
    """Per (by.., arm, L): coverage = share of declared sets holding a causal; power = share
    of causal sites inside some declared set; median size and purity of declared sets;
    false sets per rep = declared sets with no causal; sets per rep."""
    keys = [*by, "arm", "L"]
    d = declared(cs, min_log_bf=min_log_bf)
    per_set = d.group_by(keys).agg(
        pl.len().alias("n_declared"),
        pl.col("covered").mean().alias("coverage"),
        pl.col("cs_size").median().alias("median_size"),
        pl.col("purity").median().alias("median_purity"),
        (~pl.col("covered")).sum().alias("n_false"),
    )
    reps = fit.group_by(keys).agg(pl.len().alias("n_reps"), pl.col("k_causal").sum().alias("n_causal"))
    # a causal is "found" if any declared set in its rep contains it; count distinct
    # (rep, causal) hits through n_causal_in_cs (sets are disjoint for the overwhelming majority)
    hits = d.group_by([*keys, "rep"]).agg(pl.col("n_causal_in_cs").sum().clip(upper_bound=pl.col("n_causal_in_cs").sum()).alias("hits"))
    hits = hits.group_by(keys).agg(pl.col("hits").sum())
    out = reps.join(per_set, on=keys, how="left").join(hits, on=keys, how="left")
    # counts default to 0 when nothing was declared; set-level summaries stay null
    out = out.with_columns(pl.col("n_declared", "n_false", "hits").fill_null(0))
    return out.with_columns(
        pl.when(pl.col("n_causal") > 0).then((pl.col("hits") / pl.col("n_causal")).clip(upper_bound=1.0)).otherwise(None).alias("power"),
        (pl.col("n_false") / pl.col("n_reps")).alias("false_sets_per_rep"),
        (pl.col("n_declared") / pl.col("n_reps")).alias("sets_per_rep"),
    ).sort(keys)


def pip_calibration(pip: pl.DataFrame, *, by=("arm", "L"), bins: int = 10) -> pl.DataFrame:
    """Binned PIP vs realized causal rate; `n` features per bin (size the dots by it)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    b = pip.with_columns(
        pl.col("pip").clip(0.0, 1.0 - 1e-12).map_batches(lambda s: pl.Series(np.digitize(s.to_numpy(), edges[1:-1]))).alias("bin")
    )
    return b.group_by([*by, "bin"]).agg(
        pl.col("pip").mean().alias("mean_pip"),
        pl.col("causal").mean().alias("causal_rate"),
        pl.len().alias("n"),
    ).sort([*by, "bin"])


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    lab = labels[order].astype(float)
    tp = np.cumsum(lab)
    precision = tp / np.arange(1, lab.shape[0] + 1)
    recall = tp / lab.sum()
    # step-wise area, as sklearn's average_precision_score
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def pip_auprc(pip: pl.DataFrame, *, by=(*SCENARIO, "arm", "L")) -> pl.DataFrame:
    rows = []
    for key, g in pip.group_by(list(by), maintain_order=True):
        rows.append({**dict(zip(by, key)), "auprc": _auprc(g["pip"].to_numpy(), g["causal"].to_numpy())})
    return pl.DataFrame(rows).sort(list(by))


def dispersion_summary(fit: pl.DataFrame, *, by=SCENARIO) -> pl.DataFrame:
    """nb_est only: the null-model size estimate against the truth, as log2(r_used / r_true).
    On Poisson truth (phi = 0, r_true infinite) the ratio is -inf, so the summary reports
    log2(r_used) itself (`median_log2_r_hat`; the cap is log2(1e4) = 13.3) and the ratio
    columns are null."""
    e = fit.filter(pl.col("arm") == "nb_est").with_columns(
        pl.when(pl.col("r_true").is_finite()).then((pl.col("r_used") / pl.col("r_true")).log(2)).alias("log2_ratio"),
        pl.col("r_used").log(2).alias("log2_r_hat"),
    )
    return e.group_by(list(by)).agg(
        pl.col("log2_ratio").median().alias("median_log2_ratio"),
        pl.col("log2_ratio").quantile(0.1).alias("q10"),
        pl.col("log2_ratio").quantile(0.9).alias("q90"),
        pl.col("log2_r_hat").median().alias("median_log2_r_hat"),
        pl.col("null_at_cap").mean().alias("at_cap_rate"),
        pl.len().alias("n_reps"),
    ).sort(list(by))


def timing_summary(fit: pl.DataFrame, *, by=("arm", "L", "n")) -> pl.DataFrame:
    return fit.group_by(list(by)).agg(
        pl.col("seconds").median().alias("median_seconds"),
        pl.col("seconds").max().alias("max_seconds"),
        pl.col("n_iter").median().alias("median_sweeps"),
        pl.col("converged").mean().alias("converged_rate"),
    ).sort(list(by))
