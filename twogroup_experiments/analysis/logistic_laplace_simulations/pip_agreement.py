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


def match(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """Arm component index matched to each reference component (minimum total TV)."""
    tv = 0.5 * np.abs(np.exp(la)[:, None, :] - np.exp(lb)[None, :, :]).sum(-1)
    _, cols = linear_sum_assignment(tv)   # identity at L=1
    return cols


def compare(ref_se: list[dict], arm_se: list[dict]) -> tuple[list[dict], dict, np.ndarray, np.ndarray]:
    """Component-pair distances (matched, declared-on-either-side), per-fit summaries, and the
    two PIP vectors (reference, arm)."""
    la, lbf_a = _components(ref_se)
    lb, lbf_b = _components(arm_se)
    pa, pb = _pip(la), _pip(lb)
    dpip = np.abs(pa - pb)
    out = []
    for i, k in enumerate(match(la, lb)):
        dr, da = bool(lbf_a[i] >= R.MIN_LOG_BF), bool(lbf_b[k] >= R.MIN_LOG_BF)
        group = GROUPS[0] if dr and da else GROUPS[1] if dr else GROUPS[2] if da else GROUPS[3]
        out.append({**_dist(la[i], lb[k]), "declared_ref": dr, "declared_arm": da, "group": group})
    # Matched alphas compared feature by feature, grouped by which side declares the pair. A pair
    # that crosses the threshold (CAVI-only / arm-only) is its own group. NaN when a fit has no
    # pair in the group. |dPIP_j| <= sum_l |dalpha_lj|, so these bound the PIP gap.
    order = match(la, lb)
    ea, eb = np.exp(la), np.exp(lb)[order]
    da, db = lbf_a >= R.MIN_LOG_BF, lbf_b[order] >= R.MIN_LOG_BF
    dmax = np.abs(ea - eb).max(axis=1)
    fit = {"max_pip": float(dpip.max()), "argmax_pip": int(dpip.argmax()),
           "n_decl_ref": int(da.sum()), "n_decl_arm": int(db.sum())}
    for g, mask in zip(GROUPS, (da & db, da & ~db, ~da & db, ~da & ~db)):
        fit[f"max_alpha_{g}"] = float(dmax[mask].max()) if mask.any() else np.nan
        fit[f"n_{g}"] = int(mask.sum())
    return out, fit, pa, pb


PIP_FLOOR = 0.05   # feature frame keeps features with PIP >= this on either side


def load(sources=SOURCES, results_root: str = RESULTS
         ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(component-pair frame, per-fit frame, feature frame).

    * component pairs: one row per matched component pair (all L), with its declaration group.
    * fits: one row per (arm, replicate): max |dPIP| and the feature attaining it, declared
      component counts on each side, q2 ELBO difference (arm - CAVI-Q2), and both fits.parquet
      paths so a single replicate can be re-read (``replicate_pair``).
    * features: one row per (arm, replicate, feature) with PIP >= PIP_FLOOR on either side;
      every other feature has both PIPs below the floor.

    Cells carry exp, L, m, T, depth, gap, null."""
    cfg = loader.load_config()
    comp_rows, fit_rows, feat_parts = [], [], []
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
            cols = ["replicate", "single_effects", "q2_elbo"]
            ref = {r["replicate"]: r for r in
                   pl.read_parquet(cell["fits"]["cavi"], columns=cols).iter_rows(named=True)}
            for method, f in cell["fits"].items():
                if method == "cavi":
                    continue
                for r in pl.read_parquet(f, columns=cols).iter_rows(named=True):
                    rep = r["replicate"]
                    if rep not in ref:
                        continue
                    pairs, fit, pa, pb = compare(ref[rep]["single_effects"], r["single_effects"])
                    key = {"exp": exp, "L": L, "method": method, "batch_hash": bh, "rep": rep, **meta}
                    fit_rows.append({**key, **fit, "n_pairs": sum(p["group"] != GROUPS[3] for p in pairs),
                                     "d_elbo": r["q2_elbo"] - ref[rep]["q2_elbo"],
                                     "ref_file": cell["fits"]["cavi"], "arm_file": f})
                    comp_rows.extend({**key, **p} for p in pairs)
                    keep = np.flatnonzero(np.maximum(pa, pb) >= PIP_FLOOR)
                    feat_parts.append(pl.DataFrame({
                        "exp": exp, "L": L, "method": method, "null": meta["null"],
                        "j": keep.astype(np.int32), "pip_ref": pa[keep], "pip_arm": pb[keep]}))
    return (pl.DataFrame(comp_rows, infer_schema_length=None),
            pl.DataFrame(fit_rows, infer_schema_length=None),
            pl.concat(feat_parts))


def replicate_pair(row: dict) -> tuple[list[dict], list[dict], list[int]]:
    """(CAVI-Q2 single effects, arm single effects, causal feature indices) for one per-fit
    row of ``load``."""
    def read(f):
        return (pl.read_parquet(f, columns=["replicate", "single_effects", "credible_sets"])
                .filter(pl.col("replicate") == row["rep"]).row(0, named=True))
    ref, arm = read(row["ref_file"]), read(row["arm_file"])
    causal = sorted({j for cs in ref["credible_sets"] for j in cs["causal_indices"]})
    return ref["single_effects"], arm["single_effects"], causal


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


# ------------------------------------------------------------------------------ per replicate
BANDS = [(0.01, "≤ 0.01"), (0.1, "0.01-0.1"), (1.0, "> 0.1")]


def band(col: str = "max_pip") -> pl.Expr:
    return (pl.when(pl.col(col) <= 0.01).then(pl.lit(BANDS[0][1]))
            .when(pl.col(col) <= 0.1).then(pl.lit(BANDS[1][1]))
            .otherwise(pl.lit(BANDS[2][1])).alias("band"))


def quantile_cells(fits: pl.DataFrame, value: str = "max_pip") -> pl.DataFrame:
    """'median / 90th / 99th percentile' strings per (exp, L, method)."""
    return (fits.group_by("exp", "L", "method")
            .agg([pl.col(value).quantile(q, "linear").alias(f"q{int(q * 100)}")
                  for q in (0.5, 0.9, 0.99)])
            .with_columns(pl.format("{} / {} / {}", *[pl.col(c).round(3) for c in ("q50", "q90", "q99")])
                          .alias("cell")))


def share_cells(fits: pl.DataFrame, value: str = "max_pip") -> pl.DataFrame:
    """'% <= 0.01 / % > 0.1' strings per (exp, L, method)."""
    return (fits.group_by("exp", "L", "method")
            .agg((pl.col(value) <= 0.01).mean().alias("lo"), (pl.col(value) > 0.1).mean().alias("hi"))
            .with_columns(pl.format("{}% / {}%", (100 * pl.col("lo")).round(1), (100 * pl.col("hi")).round(1))
                          .alias("cell")))


def tail_table(fits: pl.DataFrame, methods=("gibss", "laplace"), nats: float = 0.1) -> pl.DataFrame:
    """Per arm and max |dPIP| band: fit count, median ELBO difference (arm - CAVI-Q2), the share
    of fits where CAVI-Q2's ELBO is higher by more than `nats` / the arm's is, and the share with
    a different number of declared components."""
    return (fits.filter(pl.col("method").is_in(list(methods))).with_columns(band())
            .group_by("method", "L", "band")
            .agg(pl.len().alias("fits"),
                 pl.col("d_elbo").median().alias("median ΔELBO"),
                 (pl.col("d_elbo") < -nats).mean().alias("CAVI-Q2 higher"),
                 (pl.col("d_elbo") > nats).mean().alias("arm higher"),
                 (pl.col("n_decl_ref") != pl.col("n_decl_arm")).mean().alias("declared count differs"))
            .with_columns(pl.col("band").replace_strict({b: i for i, (_, b) in enumerate(BANDS)},
                                                       return_dtype=pl.Int8).alias("_o"))
            .sort(pl.col("method").replace_strict({m: i for i, m in enumerate(R.METHODS)},
                                                  return_dtype=pl.Int8), "L", "_o")
            .drop("_o")
            .with_columns(pl.col("method").replace_strict(R.METHOD_LABEL).alias("arm"))
            .select("arm", "L", "band", "fits", "median ΔELBO", "CAVI-Q2 higher",
                    "arm higher", "declared count differs"))


def md_frame(df: pl.DataFrame, fmts: dict[str, str], legend: str = "") -> str:
    """Markdown table of a frame; `fmts` maps column -> format string (others via str)."""
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows(named=True):
        lines.append("| " + " | ".join(fmts[c].format(r[c]) if c in fmts else str(r[c])
                                       for c in cols) + " |")
    return "\n".join(lines) + (f"\n\n{legend}" if legend else "")


def survival_figure(fits: pl.DataFrame, rows=None, xlabel: str = "max |ΔPIP| in the fit",
                    unit: str = "fits", floor: float = 1e-4):
    """Rows = (L, column, label[, filter expr]) (default: max_pip at L = 1 and L = 5), columns =
    experiments: share of fits with the column's value > x, one line per arm, log-log. Values
    below `floor` are drawn at the floor."""
    import matplotlib.pyplot as plt
    exps = ["022", "023", "024"]
    rows = rows or [(1, "max_pip", "L = 1"), (5, "max_pip", "L = 5")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.5, 2.2 * len(rows)), sharex=True,
                             sharey=True, squeeze=False)
    for i, row in enumerate(rows):
        L, col, label = row[:3]
        keep = row[3] if len(row) > 3 else pl.lit(True)
        for j, exp in enumerate(exps):
            ax = axes[i, j]
            sub = fits.filter((pl.col("L") == L) & (pl.col("exp") == exp) & keep)
            for m in ARMS:
                x = sub.filter(pl.col("method") == m)[col].to_numpy()
                x = np.sort(np.maximum(x[~np.isnan(x)], floor))
                if x.size == 0:
                    continue
                surv = 1.0 - np.arange(1, x.size + 1) / x.size
                ax.step(x[:-1], surv[:-1], where="post", color=R.METHOD_COLOR[m], lw=1.1,
                        label=R.METHOD_LABEL[m])
            for v in (0.01, 0.1):
                ax.axvline(v, color="0.6", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(floor, 1)
            ax.set_ylim(1e-3, 1.05)
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"{exp} ({_DESIGN[exp]})", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{label}\nshare of {unit} above x", fontsize=8)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 / len(rows)))
    return fig


GROUPS = ["both", "cavi", "arm", "neither"]
GROUP_LABEL = {"both": "both declare", "cavi": "CAVI-Q2 only", "arm": "arm only",
               "neither": "neither declares"}


def group_cells(fits: pl.DataFrame, group: str) -> pl.DataFrame:
    """'% of fits > 0.1 (fits with such a pair)' per (exp, L, method) for one pair group."""
    col = f"max_alpha_{group}"
    return (fits.filter(pl.col(col).is_not_nan()).group_by("exp", "L", "method")
            .agg((pl.col(col) > 0.1).mean().alias("hi"), pl.len().alias("n"))
            .with_columns(pl.format("{}% ({})", (100 * pl.col("hi")).round(1), pl.col("n"))
                          .alias("cell")))

_DESIGN = {"022": "binary Markov", "023": "Gaussian AR(1)", "024": "nested"}


def scatter_figure(feat: pl.DataFrame, L: int = 5):
    """Rows = experiments, columns = arms: per-feature PIP, arm vs CAVI-Q2, pooled over
    replicates and non-null cells, as a hexbin with log counts. Only features with PIP >=
    PIP_FLOOR on either side (every other feature sits in the corner below it)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    exps = ["022", "023", "024"]
    fig, axes = plt.subplots(3, len(ARMS), figsize=(6.5, 4.3), sharex=True, sharey=True)
    sub_l = feat.filter((pl.col("L") == L) & ~pl.col("null"))
    for i, exp in enumerate(exps):
        for k, m in enumerate(ARMS):
            ax = axes[i, k]
            d = sub_l.filter((pl.col("exp") == exp) & (pl.col("method") == m))
            ax.hexbin(d["pip_ref"].to_numpy(), d["pip_arm"].to_numpy(), gridsize=30,
                      extent=(0, 1, 0, 1), norm=LogNorm(vmin=1, vmax=1e5), cmap="Blues",
                      mincnt=1, linewidths=0)
            ax.plot([0, 1], [0, 1], color="0.5", lw=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(R.METHOD_LABEL[m], fontsize=7.5, color=R.METHOD_COLOR[m])
            if i == 2:
                ax.set_xlabel("PIP, CAVI-Q2", fontsize=7)
            if k == 0:
                ax.set_ylabel(f"{exp}\nPIP, arm", fontsize=7)
    fig.tight_layout()
    return fig


def _cs(alpha: np.ndarray, coverage: float = 0.95) -> np.ndarray:
    o = np.argsort(-alpha)
    return o[: int(np.searchsorted(np.cumsum(alpha[o]), coverage) + 1)]


class _State:
    """The interface gibss.plotting.plot_pip reads, built from stored single effects."""

    def __init__(self, se: list[dict], order=None):
        se = [se[k] for k in order] if order is not None else se
        self.alpha = np.stack([np.asarray(e["alpha"], dtype=float) for e in se])
        self.ser_log_bf = np.array([e["ser_log_bf"] for e in se], dtype=float)
        self.pip = 1.0 - np.prod(1.0 - self.alpha, axis=0)

    def get_credible_sets(self, coverage: float = 0.95):
        return [_cs(a, coverage) for a in self.alpha]


def replicate_figure(rows: list[dict], window: int = 24):
    """One row per fit: CAVI-Q2 (left) and the arm (right) PIP plots via gibss.plotting.plot_pip,
    declared credible sets only, the arm's components reordered to their CAVI-Q2 match so a
    matched pair shares a colour. x is cropped to features with PIP >= 0.02 on either side,
    padded by `window`."""
    import matplotlib.pyplot as plt
    from gibss.plotting import plot_pip
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.5, 1.75 * len(rows)), sharey=True,
                             squeeze=False)
    for i, row in enumerate(rows):
        ref_se, arm_se, causal = replicate_pair(row)
        why = explain(ref_se, arm_se, causal)
        order = match(_components(ref_se)[0], _components(arm_se)[0])
        states = [_State(ref_se), _State(arm_se, order)]
        hot = np.flatnonzero(np.maximum(states[0].pip, states[1].pip) >= 0.02)
        lo = max(int(hot.min()) - window, 0) if hot.size else 0
        hi = int(hot.max()) + window if hot.size else len(states[0].pip)
        for k, (st, name) in enumerate(zip(states, ["cavi", row["method"]])):
            ax = axes[i, k]
            plot_pip(st, causal_idx=causal, min_log_bf=R.MIN_LOG_BF, ax=ax, show_legend=False)
            ax.set_xlim(lo - 0.5, min(hi, len(st.pip)) - 0.5)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("feature" if i == len(rows) - 1 else "", fontsize=7)
            ax.set_ylabel("PIP" if k == 0 else "", fontsize=7)
            cellname = ", ".join(f"{c} = {row[c]}" for c in dict.fromkeys((axis_col(row["exp"]), "T", "gap")))
            title = f"{R.METHOD_LABEL[name]}"
            if k == 0:
                title += f"   ({row['exp']}: {cellname}, rep {row['rep']})"
            else:
                title += (f"   max |ΔPIP| {row['max_pip']:.2f}, ΔELBO {row['d_elbo']:+.2f}, "
                          f"driver: {why['driver']}")
            ax.set_title(title, fontsize=7, loc="left")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------ why PIPs differ
# Matched components make log(1 - PIP_j) additive over pairs:
#   log(1 - PIP'_j) - log(1 - PIP_j) = sum_l [log(1 - alpha'_{s(l)j}) - log(1 - alpha_lj)],
# so the PIP gap at the worst feature j* splits exactly into one term per matched pair. The pair
# with the largest |term| drives the gap; it is classified by which side declares it.
DRIVERS = ["same CS", "CS moved", "CAVI-only", "arm-only", "undeclared"]
NEAR = 1.0   # a one-sided pair is "near threshold" when the other side's log BF >= MIN_LOG_BF - NEAR


def explain(ref_se: list[dict], arm_se: list[dict], causal: list[int]) -> dict:
    """Attribute the fit's largest PIP difference to a matched component pair."""
    la, lbf_a = _components(ref_se)
    lb, lbf_b = _components(arm_se)
    order = match(la, lb)
    a, b, lbf_b = np.exp(la), np.exp(lb)[order], lbf_b[order]
    pa, pb = _pip(la), 1.0 - np.prod(1.0 - b, axis=0)
    j = int(np.abs(pa - pb).argmax())
    eps = 1e-12
    terms = np.log1p(-np.minimum(b[:, j], 1 - eps)) - np.log1p(-np.minimum(a[:, j], 1 - eps))
    l = int(np.abs(terms).argmax())
    dr, da = lbf_a >= R.MIN_LOG_BF, lbf_b >= R.MIN_LOG_BF
    if dr[l] and da[l]:
        same = j in set(_cs(a[l])) and j in set(_cs(b[l]))
        driver = DRIVERS[0] if same else DRIVERS[1]
        other = np.nan
    elif dr[l]:
        driver, other = DRIVERS[2], float(lbf_b[l])
    elif da[l]:
        driver, other = DRIVERS[3], float(lbf_a[l])
    else:
        driver, other = DRIVERS[4], np.nan
    truth = 1.0 if j in set(causal) else 0.0
    return {"j": j, "j_causal": bool(truth), "pip_ref_j": float(pa[j]), "pip_arm_j": float(pb[j]),
            "ref_closer": bool(abs(pa[j] - truth) < abs(pb[j] - truth)),
            "driver": driver, "driver_share": float(abs(terms[l]) / np.abs(terms).sum()),
            "other_log_bf": other, "near": bool(other >= R.MIN_LOG_BF - NEAR),
            "driver_log_bf_ref": float(lbf_a[l]), "driver_log_bf_arm": float(lbf_b[l]),
            "driver_alpha_ref_j": float(a[l, j]), "driver_alpha_arm_j": float(b[l, j]),
            "n_shared": int((dr & da).sum()), "n_ref_only": int((dr & ~da).sum()),
            "n_arm_only": int((~dr & da).sum())}


def tail_frame(fits: pl.DataFrame, threshold: float = 0.1, methods=ARMS) -> pl.DataFrame:
    """`explain` for every fit with max |dPIP| > threshold (re-reads the two fits.parquet)."""
    tail = fits.filter((pl.col("max_pip") > threshold) & pl.col("method").is_in(list(methods)))
    out = []
    cols = ["replicate", "single_effects", "credible_sets"]
    for (rf, af), g in tail.group_by("ref_file", "arm_file", maintain_order=True):
        reps = set(g["rep"].to_list())
        ref = {r["replicate"]: r for r in pl.read_parquet(rf, columns=cols).iter_rows(named=True)
               if r["replicate"] in reps}
        arm = {r["replicate"]: r for r in pl.read_parquet(af, columns=cols).iter_rows(named=True)
               if r["replicate"] in reps}
        for row in g.iter_rows(named=True):
            r0, r1 = ref[row["rep"]], arm[row["rep"]]
            causal = sorted({j for cs in r0["credible_sets"] for j in cs["causal_indices"]})
            out.append({**{k: row[k] for k in ("exp", "L", "method", "batch_hash", "rep",
                                                "max_pip", "d_elbo")},
                        **explain(r0["single_effects"], r1["single_effects"], causal)})
    return pl.DataFrame(out, infer_schema_length=None)


def driver_table(tf: pl.DataFrame) -> pl.DataFrame:
    """Per arm and L: tail fits, the share driven by each pair type (one-sided split into near /
    far from the threshold), the median share of the gap the driving pair explains, and how often
    CAVI-Q2's PIP at j* is the closer one to the truth (1 if j* is causal, else 0)."""
    g = (tf.group_by("method", "L")
         .agg(pl.len().alias("fits"),
              *[(pl.col("driver") == d).mean().alias(d) for d in DRIVERS[:2]],
              *[((pl.col("driver") == d) & f).mean().alias(f"{d} {lab}")
                for d in DRIVERS[2:4] for f, lab in ((pl.col("near"), "near"), (~pl.col("near"), "far"))],
              (pl.col("driver") == DRIVERS[4]).mean().alias(DRIVERS[4]),
              pl.col("driver_share").median().alias("driver explains"),
              pl.col("ref_closer").mean().alias("CAVI-Q2 closer")))
    return (g.sort(pl.col("method").replace_strict({m: i for i, m in enumerate(R.METHODS)},
                                                   return_dtype=pl.Int8), "L")
            .with_columns(pl.col("method").replace_strict(R.METHOD_LABEL).alias("arm"))
            .drop("method").select("arm", pl.exclude("arm")))


def pattern_table(tf: pl.DataFrame, method: str = "gibss", L: int = 5, top: int = 8) -> pl.DataFrame:
    """Most common (shared, CAVI-only, arm-only) declared-pair counts among tail fits."""
    t = tf.filter((pl.col("method") == method) & (pl.col("L") == L))
    return (t.group_by("n_shared", "n_ref_only", "n_arm_only")
            .agg(pl.len().alias("fits"), pl.col("d_elbo").median().alias("median ΔELBO"),
                 pl.col("ref_closer").mean().alias("CAVI-Q2 closer to truth"))
            .with_columns((pl.col("fits") / t.height).alias("share"))
            .sort("fits", descending=True).head(top)
            .rename({"n_shared": "shared", "n_ref_only": "CAVI-only", "n_arm_only": "arm-only"}))


def undeclared_table(tf: pl.DataFrame, methods=("gibss", "laplace")) -> pl.DataFrame:
    """Tail fits driven by a pair neither side declares: the pair's log BFs, its largest alpha at
    j*, whether j* is causal, and which side has the higher PIP there."""
    u = tf.filter((pl.col("driver") == DRIVERS[4]) & pl.col("method").is_in(list(methods)))
    return (u.group_by("method", "L")
            .agg(pl.len().alias("fits"),
                 pl.col("driver_log_bf_ref").median().alias("log BF, CAVI-Q2"),
                 pl.col("driver_log_bf_arm").median().alias("log BF, arm"),
                 pl.col("driver_alpha_ref_j").median().alias("α at j*, CAVI-Q2"),
                 pl.col("driver_alpha_arm_j").median().alias("α at j*, arm"),
                 pl.col("j_causal").mean().alias("j* causal"),
                 (pl.col("pip_ref_j") > pl.col("pip_arm_j")).mean().alias("CAVI-Q2 PIP higher"))
            .sort(pl.col("method").replace_strict({m: i for i, m in enumerate(R.METHODS)},
                                                  return_dtype=pl.Int8), "L")
            .with_columns(pl.col("method").replace_strict(R.METHOD_LABEL).alias("arm"))
            .drop("method").select("arm", pl.exclude("arm")))
