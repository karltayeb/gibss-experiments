"""Load + summarize experiment 022 (gIBSS-Laplace / gIBSS-Q2 / global-JJ / CAVI-Q2, logistic SuSiE).

Reads each fit's `fits.parquet` directly (the shared pip/cs reductions are binned and carry
no effect-size posterior, intercept, or runtime) and returns tidy frames:

  * ``fit_frame(sc)``    one row per (cell, method, fit_L, rep): cell coordinates (set size m,
                         T, L*, gap, null flag), q2_elbo, runtime, n_iter, intercept, the
                         leading effect's EB prior variance, per-feature PIPs, declared CSs.
  * ``causal_frame(ff)`` one row per (fit, causal feature): the conditional posterior
                         N(mu, var) of that feature's effect in the component that selects it
                         most (argmax_l alpha_lj), its log BF, and the truth.
  * ``paired(ff, col)``  a column differenced against CAVI-Q2 on the SAME simulated data.

Method keys: cavi = CAVI-Q2, gibss = gIBSS-Q2, laplace = gIBSS-Laplace, globaljj = global-JJ,
score = one Newton step at the null.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
import experiments.loader as loader  # noqa: E402

_BETAS_JSON = json.load(open(os.path.join(_HERE, "betas.json")))
_BETAS = _BETAS_JSON["betas"]
N_ROWS = int(_BETAS_JSON["_meta"]["n"])
B0_TRUE = float(_BETAS_JSON["_meta"]["b0"])
MIN_LOG_BF = 2.0          # declared-CS threshold (the experiment's default_args min_log_bf)

METHODS = ["cavi", "gibss", "laplace", "globaljj", "score"]   # most to least accurate
METHOD_LABEL = {"laplace": "gIBSS-Laplace", "gibss": "gIBSS-Q2", "globaljj": "global-JJ",
                "cavi": "CAVI-Q2", "score": "score"}
# The 019 logistic palette (resultslib.METHOD_COLOR): gIBSS blue, CAVI vermillion, global-JJ
# green, score reddish purple; gIBSS-Laplace sky blue (a gIBSS variant, as gibss_profiled in 019).
# Validated with the dataviz validate_palette.js, light mode, all pairs: score/global-JJ is
# deutan dE 7.6 (floor band), so the markers + legend + x-dodge carry identity too.
METHOD_COLOR = {"laplace": "#56B4E9", "gibss": "#0072B2", "globaljj": "#009E73",
                "cavi": "#D55E00", "score": "#CC79A7"}
METHOD_MARKER = {"laplace": "v", "gibss": "s", "globaljj": "D", "cavi": "o", "score": "P"}


def _method_key(mname: str) -> str:
    if mname.endswith("_globaljj"):
        return "globaljj"
    if mname.endswith("_score"):
        return "score"
    if mname.endswith("_gibss_laplace"):
        return "laplace"
    if mname.endswith("_gibss"):
        return "gibss"
    if mname.endswith("_cavi"):
        return "cavi"
    raise ValueError(mname)


def cell_meta(scoord: dict) -> dict:
    dargs = scoord["design"]["arguments"]
    enr = scoord["enrichment"]
    a = enr.get("arguments") or {}
    m = int(round(float(dargs["density"]) * int(dargs["n"])))
    if "causal_effects" in a:
        effs = [float(e) for e in a["causal_effects"]]
        beta, lstar, gap = effs[0], len(effs), int(a["gap"])
    else:
        beta, gap = float(a["causal_effect"]), None
        lstar = 0 if beta == 0.0 else 1
    T = None
    if beta != 0.0:
        row = _BETAS[str(m)]
        T = int(min(row, key=lambda t: abs(beta - row[t])))
    return {"m": m, "T": T, "beta": beta, "Lstar": lstar, "gap": gap, "null": lstar == 0}


def _pip(alphas: np.ndarray) -> np.ndarray:
    return 1.0 - np.prod(1.0 - alphas, axis=0)


def fit_frame(sc: str = "022-laplace-pilot",
              results_root: str = os.path.join(_TG_ROOT, "results")) -> pl.DataFrame:
    cfg = loader.load_config()
    rows, seen = [], set()
    for coll in loader.collection_method_pairs(cfg, sc).values():
        for bh, mh, mname, mcoord, scoord in coll["pairs"]:
            if (bh, mh) in seen:
                continue
            seen.add((bh, mh))
            f = f"{results_root}/by_batch/{bh}/fits/{mh}/fits.parquet"
            if not os.path.exists(f):
                continue
            meta = cell_meta(scoord)
            L = int((mcoord.get("kwargs") or {}).get("L", 1) or 1)
            method = _method_key(mname)
            for r in pl.read_parquet(f).iter_rows(named=True):
                se = r["single_effects"]
                alpha = np.array([e["alpha"] for e in se])
                lbf = np.array([e["ser_log_bf"] for e in se])
                causal = list(r["credible_sets"][0]["causal_indices"])
                declared = [(len(c["cs"]), bool(c["causal_in_cs"]))
                            for c, b in zip(r["credible_sets"], lbf) if b >= MIN_LOG_BF]
                lead = int(np.argmax(lbf))
                rows.append({
                    **meta, "method": method, "fit_L": L, "batch_hash": bh,
                    "rep": int(r["replicate"]), "q2_elbo": r["q2_elbo"],
                    "fit_seconds": r["fit_seconds"], "n_iter": r["fit_summary"]["n_iter"],
                    "converged": r["fit_summary"]["converged"],
                    "intercept": r["family_state"]["intercept"],
                    "lead_prior_variance": float(se[lead]["prior_variance"]),
                    "pip": _pip(alpha).tolist(), "causal": causal,
                    "cs_sizes": [s for s, _ in declared], "cs_covers": [c for _, c in declared],
                    "_se": [{k: e[k] for k in ("alpha", "mu", "var", "feature_log_bf")} for e in se],
                    # causal features captured by a declared CS (component log BF >= MIN_LOG_BF)
                    "detected": sorted({j for c, b in zip(r["credible_sets"], lbf)
                                        if b >= MIN_LOG_BF for j in c["cs"]} & set(causal)),
                })
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def causal_frame(ff: pl.DataFrame) -> pl.DataFrame:
    """Per (fit, causal feature): conditional posterior of b_j in its selecting component.
    `detected` = a declared CS contains j. Effect-size metrics (bias, beta interval coverage)
    are only meaningful for detected effects: an undetected causal is read from an ARD-shrunk
    near-null component (EB prior variance ~0.2), whose interval sits at 0 by construction."""
    out = []
    for r in ff.filter(~pl.col("null")).iter_rows(named=True):
        alpha = np.array([e["alpha"] for e in r["_se"]])
        for j in r["causal"]:
            l = int(np.argmax(alpha[:, j]))
            e = r["_se"][l]
            mu, var = float(e["mu"][j]), float(e["var"][j])
            sd = float(np.sqrt(max(var, 1e-300)))
            out.append({
                k: r[k] for k in ("m", "T", "beta", "Lstar", "gap", "method", "fit_L",
                                  "batch_hash", "rep")
            } | {
                "j": int(j), "detected": int(j) in set(r["detected"]),
                "alpha": float(alpha[l, j]), "mu": mu, "sd": sd,
                "z": (mu - r["beta"]) / sd, "covered": abs(mu - r["beta"]) <= 1.959964 * sd,
                "feature_log_bf": float(e["feature_log_bf"][j]),
            })
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


_KEY = ["m", "T", "beta", "Lstar", "gap", "fit_L", "batch_hash", "rep"]


def paired(df: pl.DataFrame, col: str, *, extra_key=(), ratio=False) -> pl.DataFrame:
    """`col` of each non-CAVI arm minus (or over) CAVI-Q2's on the same simulated data."""
    key = [k for k in _KEY if k in df.columns] + list(extra_key)
    ref = df.filter(pl.col("method") == "cavi").select(key + [pl.col(col).alias("_ref")])
    j = df.filter(pl.col("method") != "cavi").join(ref, on=key, how="inner", nulls_equal=True)
    op = (pl.col(col) / pl.col("_ref")) if ratio else (pl.col(col) - pl.col("_ref"))
    return j.with_columns(op.alias(f"{col}_vs_cavi"))


def pip_long(ff: pl.DataFrame) -> pl.DataFrame:
    """One row per (fit, feature): pip, is_causal. For AUPRC and paired PIP differences."""
    rows = []
    for r in ff.filter(~pl.col("null")).iter_rows(named=True):
        cz = set(r["causal"])
        for j, p in enumerate(r["pip"]):
            rows.append({k: r[k] for k in _KEY + ["method"]} | {"j": j, "pip": p, "is_causal": j in cz})
    return pl.DataFrame(rows, infer_schema_length=None)


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """AP = mean precision at each true positive, ranking by score (ties broken stably)."""
    order = np.argsort(-score, kind="stable")
    y = label[order].astype(float)
    if y.sum() == 0:
        return float("nan")
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def auprc(pl_long: pl.DataFrame, by=("m", "fit_L", "method")) -> pl.DataFrame:
    """AUPRC pooled over reps (and T / gap) within each `by` group."""
    out = []
    for keys, g in pl_long.group_by(list(by)):
        out.append(dict(zip(by, keys)) | {
            "auprc": average_precision(g["pip"].to_numpy(), g["is_causal"].to_numpy())})
    return pl.DataFrame(out).sort(list(by))


# ------------------------------------------------------------------------------ tables
def md_table(tab: pl.DataFrame, *, rows: list[str], value: str, fmt="{:.3f}",
             legend: str = "") -> str:
    """Markdown table: one row per combo of `rows`, one column per method (fixed order)."""
    header = rows + [METHOD_LABEL[m] for m in METHODS if m in tab["method"].unique().to_list()]
    present = [m for m in METHODS if m in tab["method"].unique().to_list()]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for combo in tab.select(rows).unique().sort(rows).iter_rows():
        g = tab
        for c, v in zip(rows, combo):
            g = g.filter(pl.col(c).is_null() if v is None else pl.col(c) == v)
        cells = []
        for m in present:
            x = g.filter(pl.col("method") == m)[value]
            cells.append("-" if x.len() == 0 or x[0] is None else fmt.format(x[0]))
        lines.append("| " + " | ".join(["-" if v is None else str(v) for v in combo] + cells) + " |")
    return "\n".join(lines) + (f"\n\n{legend}" if legend else "")


# ------------------------------------------------------------------------------ summaries
def mean_se(df: pl.DataFrame, value: str, by: list[str]) -> pl.DataFrame:
    return (df.group_by(by)
              .agg(pl.col(value).cast(pl.Float64).mean().alias("mean"),
                   (pl.col(value).cast(pl.Float64).std() / pl.len().sqrt()).alias("se"),
                   pl.len().alias("n"))
              .sort(by))


# ------------------------------------------------------------------------------ figures
# Multiplicative x-dodge on the log-m axis so coincident arms (gIBSS-Q2 ~ CAVI-Q2 at L=1) stay
# visible side by side instead of stacking.
_DODGE = {"cavi": 0.89, "gibss": 0.945, "laplace": 1.0, "globaljj": 1.055, "score": 1.11}


def _draw_vs_m(ax, tab: pl.DataFrame, *, methods, ref, m_ticks, symlog=None):
    for mth in methods:
        s = tab.filter(pl.col("method") == mth).sort("m")
        if s.height == 0:
            continue
        x = s["m"].to_numpy() * _DODGE[mth]
        ax.errorbar(x, s["mean"].to_numpy(), yerr=s["se"].fill_null(0).to_numpy(),
                    color=METHOD_COLOR[mth], marker=METHOD_MARKER[mth], ms=5, lw=1.6,
                    capsize=0, label=METHOD_LABEL[mth])
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=0.8, ls="--", zorder=0)
    if symlog is not None:   # gaps spanning 1e-3..10 nats: linear inside +-symlog, log outside
        ax.set_yscale("symlog", linthresh=symlog)
        shown = tab.filter(pl.col("method").is_in(list(methods)))
        hi = float((shown["mean"] + shown["se"].fill_null(0)).max())
        if hi <= symlog:   # all gaps on one side: drop the empty positive decades
            ax.set_ylim(top=symlog)
    ax.set_xscale("log")
    ax.set_xticks(m_ticks)
    ax.set_xticklabels([str(v) for v in m_ticks])
    ax.minorticks_off()
    ax.set_xlabel("expected set size m")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.6)


def _legend_on_top(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=8)


def line_vs_m(tab: pl.DataFrame, *, ylabel: str, facet: str | None = None, ref: float | None = None,
              methods=METHODS, title_fmt="{}={}", ax_w=2.6, ax_h=2.3, symlog=None):
    """mean +- 1 se vs set size m (log x), one line per method, optional facet columns.
    `tab` is a `mean_se` frame keyed by m (+ facet) + method."""
    import matplotlib.pyplot as plt
    fvals = [None] if facet is None else sorted(v for v in tab[facet].unique().to_list() if v is not None)
    m_ticks = sorted(tab["m"].unique().to_list())
    width = ax_w * len(fvals) + 0.4 if facet is not None else 3.9   # room for a 3-arm legend
    fig, axes = plt.subplots(1, len(fvals), figsize=(width, ax_h), sharey=True, squeeze=False)
    for ax, fv in zip(axes[0], fvals):
        g = tab if facet is None else tab.filter(pl.col(facet) == fv)
        _draw_vs_m(ax, g, methods=methods, ref=ref, m_ticks=m_ticks, symlog=symlog)
        if fv is not None:
            ax.set_title(title_fmt.format(facet, fv), fontsize=9)
    axes[0][0].set_ylabel(ylabel)
    _legend_on_top(fig, axes[0][0])
    fig.tight_layout()
    return fig


def panels_vs_m(panels: list[tuple], *, methods=METHODS, ax_w=3.1, ax_h=2.3):
    """Side-by-side panels with their own y-axis: each is (mean_se frame, ylabel, ref line) or
    (mean_se frame, ylabel, ref line, symlog linthresh)."""
    import matplotlib.pyplot as plt
    m_ticks = sorted({v for spec in panels for v in spec[0]["m"].unique().to_list()})
    fig, axes = plt.subplots(1, len(panels), figsize=(ax_w * len(panels) + 0.3, ax_h), squeeze=False)
    for ax, spec in zip(axes[0], panels):
        tab, ylabel, ref = spec[:3]
        _draw_vs_m(ax, tab, methods=methods, ref=ref, m_ticks=m_ticks,
                   symlog=spec[3] if len(spec) > 3 else None)
        ax.set_ylabel(ylabel)
    _legend_on_top(fig, axes[0][0])
    fig.tight_layout()
    return fig
