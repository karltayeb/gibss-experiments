"""Agreement of each arm's variable-selection posterior with CAVI-Q2's, on the same replicate.

Per single effect l, alpha_l is the posterior over which feature carries the effect. With a
uniform prior over features, alpha_l = softmax(feature_log_bf_l) (checked to 1e-15), so log alpha
comes from a log-softmax and KL needs no floor. Three distances, arm vs CAVI-Q2:

* TV  = 1/2 sum_j |alpha_j - alpha'_j| = max over feature SETS A of |alpha(A) - alpha'(A)|.
* KL  = KL(alpha_CAVI || alpha_arm) = sum_j alpha_j (log alpha_j - log alpha'_j). Large when the
        arm puts little mass where CAVI puts mass (an overconfident arm).
* max = max_j |alpha_j - alpha'_j|, the largest single-feature change (TV over singletons).

At L=1 alpha IS the PIP vector. At L=5 the components are unordered, so the arm's components are
matched to CAVI's by minimum total TV (Hungarian). At both L, only pairs where EITHER side's component is
declared (component log BF >= MIN_LOG_BF) are kept: a null component's diffuse alpha is noise.
Separately, per fit, max_j |PIP_arm - PIP_CAVI| over the combined PIPs (no matching needed).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.special import log_softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import experiments.loader as loader  # noqa: E402
import laplacelib as R  # noqa: E402

RESULTS = os.path.join(_TG_ROOT, "results")
# (experiment, L, supercollection). Accuracy, not timing, so the full 022 grids are fine here.
SOURCES = [
    ("022", 5, "022-laplace"), ("022", 1, "022-laplace-ser"),
    ("023", 5, "023-laplace-gaussian"), ("023", 1, "023-laplace-gaussian-ser"),
    ("024", 5, "024-laplace-nested"), ("024", 1, "024-laplace-nested-ser"),
]
ARMS = [m for m in R.METHODS if m != "cavi"]
METRICS = {"tv": "TV", "kl": "KL", "max_alpha": "max |Δα|"}


def _dist(la: np.ndarray, lb: np.ndarray) -> dict:
    """Distances from reference alpha (log la) to arm alpha (log lb), both log-softmaxed."""
    a, b = np.exp(la), np.exp(lb)
    d = np.abs(a - b)
    return {"tv": 0.5 * d.sum(), "kl": float(np.sum(a * (la - lb))), "max_alpha": float(d.max())}


def _components(se: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(log alpha [L, p], component log BF [L])."""
    la = np.stack([log_softmax(np.asarray(e["feature_log_bf"], dtype=float)) for e in se])
    return la, np.array([e["ser_log_bf"] for e in se], dtype=float)


def _pip(la: np.ndarray) -> np.ndarray:
    return 1.0 - np.prod(1.0 - np.exp(la), axis=0)


def compare(ref_se: list[dict], arm_se: list[dict]) -> tuple[list[dict], float]:
    """Component-pair distances (matched, declared-on-either-side) and the per-fit max |dPIP|."""
    la, lbf_a = _components(ref_se)
    lb, lbf_b = _components(arm_se)
    max_pip = float(np.abs(_pip(la) - _pip(lb)).max())
    tv = 0.5 * np.abs(np.exp(la)[:, None, :] - np.exp(lb)[None, :, :]).sum(-1)
    rows, cols = linear_sum_assignment(tv)   # identity at L=1
    out = []
    for i, k in zip(rows, cols):
        dr, da = bool(lbf_a[i] >= R.MIN_LOG_BF), bool(lbf_b[k] >= R.MIN_LOG_BF)
        if dr or da:
            out.append({**_dist(la[i], lb[k]), "declared_ref": dr, "declared_arm": da})
    return out, max_pip


def load(sources=SOURCES, results_root: str = RESULTS) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(component-pair frame, per-fit frame). One row per matched declared component pair /
    per (arm, replicate). Cells carry exp, L, m, T, depth, gap, null."""
    cfg = loader.load_config()
    comp_rows, fit_rows = [], []
    for exp, L, sc in sources:
        by_batch: dict[str, dict] = {}
        for coll in loader.collection_method_pairs(cfg, sc).values():
            for bh, mh, mname, _mc, scoord in coll["pairs"]:
                f = f"{results_root}/by_batch/{bh}/fits/{mh}/fits.parquet"
                if not os.path.exists(f):
                    continue
                cell = by_batch.setdefault(bh, {"meta": R.cell_meta(scoord), "fits": {}})
                cell["fits"][R._method_key(mname)] = f
        for bh, cell in by_batch.items():
            if "cavi" not in cell["fits"]:
                continue
            meta = {k: cell["meta"][k] for k in ("m", "T", "depth", "gap", "null")}
            ref = {r["replicate"]: r["single_effects"] for r in
                   pl.read_parquet(cell["fits"]["cavi"], columns=["replicate", "single_effects"])
                   .iter_rows(named=True)}
            for method, f in cell["fits"].items():
                if method == "cavi":
                    continue
                for r in pl.read_parquet(f, columns=["replicate", "single_effects"]).iter_rows(named=True):
                    rep = r["replicate"]
                    if rep not in ref:
                        continue
                    pairs, max_pip = compare(ref[rep], r["single_effects"])
                    key = {"exp": exp, "L": L, "method": method, "batch_hash": bh, "rep": rep, **meta}
                    fit_rows.append({**key, "max_pip": max_pip, "n_pairs": len(pairs)})
                    comp_rows.extend({**key, **p} for p in pairs)
    return (pl.DataFrame(comp_rows, infer_schema_length=None),
            pl.DataFrame(fit_rows, infer_schema_length=None))


def _boot_ci(x: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% interval for the mean of x, resampling whole replicates (groups) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=x)
    cnts = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(df: pl.DataFrame, value: str, by: list[str]) -> pl.DataFrame:
    """Mean of `value` per group with a replicate-bootstrap 95% CI, plus median and 90th pct."""
    out = []
    for key, g in df.group_by(by, maintain_order=True):
        x = g[value].to_numpy().astype(float)
        rep_id = (g["batch_hash"] + ":" + g["rep"].cast(pl.Utf8)).to_numpy()
        lo, hi = _boot_ci(x, rep_id)
        out.append({**dict(zip(by, key)), "mean": float(x.mean()), "lo": lo, "hi": hi,
                    "median": float(np.median(x)), "q90": float(np.quantile(x, 0.9)), "n": len(x)})
    return pl.DataFrame(out)


def fmt(tab: pl.DataFrame, digits: int = 3) -> pl.DataFrame:
    """'mean (lo, hi)' cell strings."""
    f = f"{{:.{digits}f}}"
    return tab.with_columns(pl.struct("mean", "lo", "hi").map_elements(
        lambda s: f"{f.format(s['mean'])} ({f.format(s['lo'])}, {f.format(s['hi'])})",
        return_dtype=pl.Utf8).alias("cell"))


def axis_col(exp: str) -> str:
    return {"022": "m", "023": "T", "024": "depth"}[exp]


_AXIS_LABEL = {"022": "set size $m$ (022)", "023": "signal $T$ (023)", "024": "depth $d$ (024)"}


def by_axis_figure(comp: pl.DataFrame, L: int):
    """Rows = metrics (TV, KL, max |dalpha|), columns = experiments; mean per cell vs the sweep
    axis with a replicate-bootstrap 95% interval, one line per arm; non-null cells, log y."""
    import matplotlib.pyplot as plt
    exps = ["022", "023", "024"]
    fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.2), sharex="col", sharey="row")
    sub_l = comp.filter((pl.col("L") == L) & ~pl.col("null"))
    for i, (metric, label) in enumerate(METRICS.items()):
        for j, exp in enumerate(exps):
            ax, col = axes[i, j], axis_col(exp)
            sub = sub_l.filter(pl.col("exp") == exp)
            if sub.height:
                tab = summarize(sub, metric, ["method", col]).sort(col)
                for m in ARMS:
                    t = tab.filter(pl.col("method") == m)
                    if t.height == 0:
                        continue
                    ax.errorbar(t[col].to_numpy(), t["mean"],
                                yerr=[t["mean"] - t["lo"], t["hi"] - t["mean"]],
                                color=R.METHOD_COLOR[m], marker=R.METHOD_MARKER[m], ms=3.5, lw=1,
                                capsize=0, label=R.METHOD_LABEL[m])
            ax.set_yscale("log")
            if exp == "022":
                ax.set_xscale("log")
                ax.set_xlim(3.5, 560)
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=7)
            if i == 2:
                ax.set_xlabel(_AXIS_LABEL[exp], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=max(len(handles), 1),
               fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
