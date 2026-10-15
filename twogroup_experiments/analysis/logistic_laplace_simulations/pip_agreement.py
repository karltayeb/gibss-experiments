"""Agreement of each arm's variable-selection posterior with CAVI-Q2's, on the same replicate.

Per single effect l, alpha_l is the posterior over which feature carries the effect. With a
uniform prior over features, alpha_l = softmax(feature_log_bf_l) (checked to 1e-15), so log alpha
comes from a log-softmax and KL needs no floor. Three distances, arm vs CAVI-Q2:

* TV  = 1/2 sum_j |alpha_j - alpha'_j| = max over feature SETS A of |alpha(A) - alpha'(A)|.
* KL  = KL(alpha_CAVI || alpha_arm) = sum_j alpha_j (log alpha_j - log alpha'_j). Large when the
        arm puts little mass where CAVI puts mass (an overconfident arm).
* max = max_j |alpha_j - alpha'_j|, the largest single-feature change (TV over singletons).

Each single effect is q(gamma) q(b | gamma), with q(b | gamma = j) = N(mu_j, var_j) (the stored
mu/var are these conditional moments). Conditionals, arm vs CAVI-Q2, averaged over CAVI-Q2's
alpha:

* dmu     = sum_j alpha_j |mu'_j - mu_j| / sd_j, the mean shift in CAVI-Q2 posterior SDs.
* lvar    = sum_j alpha_j log(var'_j / var_j), signed: > 0 when the arm is wider.
* kl_cond = sum_j alpha_j KL(N(mu_j, var_j) || N(mu'_j, var'_j)).
* kl_joint = kl + kl_cond = KL(q_CAVI || q_arm) over (gamma, b), by the chain rule.

At L=1 alpha IS the PIP vector. At L=5 the components are unordered, so the arm's components are
matched to CAVI's by minimum total TV (Hungarian). At both L, only pairs where EITHER side's component is
declared (component log BF >= MIN_LOG_BF) are kept: a null component's diffuse alpha is noise.
Separately, per fit, max_j |PIP_arm - PIP_CAVI| over the combined PIPs (no matching needed).

Separation: a feature separates y when every row with x_j != 0, or every row with x_j == 0, has
the same y. Its likelihood has no maximum, so its conditional mean and variance are set by the
prior (the EB prior variance runs to its cap). A pair is flagged `sep` when either side's top
feature separates y; the conditional comparisons leave these out. Found by re-running the
pipeline's simulation for each replicate (``posterior_figure.simulate_dense``).
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
from experiments import loader

import laplacelib as R
import posterior_figure as PF

RESULTS = os.path.join(_TG_ROOT, "results")
# (experiment, L, supercollection). Accuracy, not timing, so the full 022 grids are fine here.
SOURCES = [
    ("022", 5, "022-laplace"), ("022", 1, "022-laplace-ser"),
    ("023", 5, "023-laplace-gaussian"), ("023", 1, "023-laplace-gaussian-ser"),
    ("024", 5, "024-laplace-nested"), ("024", 1, "024-laplace-nested-ser"),
]
ARMS = [m for m in R.METHODS if m != "cavi"]
METRICS = {"tv": "TV", "kl": "KL", "max_alpha": "max |Δα|"}
# The per-cell figure: one distance for alpha, one for the conditional effect sizes.
BY_AXIS_METRICS = {"tv": "TV, alpha", "dmu": "|Δμ| / σ, conditional mean"}


def _dist(la: np.ndarray, lb: np.ndarray) -> dict:
    """Distances from reference alpha (log la) to arm alpha (log lb), both log-softmaxed."""
    a, b = np.exp(la), np.exp(lb)
    d = np.abs(a - b)
    return {"tv": 0.5 * d.sum(), "kl": float(np.sum(a * (la - lb))), "max_alpha": float(d.max())}


def _cond(la: np.ndarray, mu: np.ndarray, var: np.ndarray, mu_b: np.ndarray,
          var_b: np.ndarray) -> dict:
    """Conditional-posterior distances, reference N(mu, var) to arm N(mu_b, var_b) per feature,
    averaged over the reference alpha (exp la)."""
    a = np.exp(la)
    kl_j = 0.5 * (var / var_b + (mu - mu_b) ** 2 / var_b - 1.0 - np.log(var / var_b))
    return {"dmu": float(np.sum(a * np.abs(mu_b - mu) / np.sqrt(var))),
            "lvar": float(np.sum(a * np.log(var_b / var))),
            "kl_cond": float(np.sum(a * kl_j))}


def _components(se: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(log alpha [L, p], component log BF [L])."""
    la = np.stack([log_softmax(np.asarray(e["feature_log_bf"], dtype=float)) for e in se])
    return la, np.array([e["ser_log_bf"] for e in se], dtype=float)


def _moments(se: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(conditional mean [L, p], conditional variance [L, p])."""
    return (np.stack([np.asarray(e["mu"], dtype=float) for e in se]),
            np.stack([np.asarray(e["var"], dtype=float) for e in se]))


def _pip(la: np.ndarray) -> np.ndarray:
    return 1.0 - np.prod(1.0 - np.exp(la), axis=0)


def match(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """Arm component index matched to each reference component (minimum total TV)."""
    tv = 0.5 * np.abs(np.exp(la)[:, None, :] - np.exp(lb)[None, :, :]).sum(-1)
    _, cols = linear_sum_assignment(tv)   # identity at L=1
    return cols


PIP_FLOOR = 0.05   # feature frames keep features with PIP (alpha) >= this on either side


def separated(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per feature: True when y is constant over the rows with x_j != 0, or over the (non-empty)
    rows with x_j == 0. Never true for a dense continuous column unless y is constant."""
    on = X != 0
    n_on, y_on = on.sum(0), y @ on
    n_off, y_off = len(y) - n_on, y.sum() - y_on
    return (y_on == 0) | (y_on == n_on) | ((n_off > 0) & ((y_off == 0) | (y_off == n_off)))


def compare(ref_se: list[dict], arm_se: list[dict], sep: np.ndarray | None = None
            ) -> tuple[list[dict], dict, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Component-pair distances (matched, declared-on-either-side), per-fit summaries, the
    two PIP vectors (reference, arm), and the matched alphas (reference, arm) of the pairs
    either side declares, at the features where either alpha is >= PIP_FLOOR. `sep` marks the
    features that separate y (``separated``); None for null cells."""
    la, lbf_a = _components(ref_se)
    lb, lbf_b = _components(arm_se)
    (mu_a, var_a), (mu_b, var_b) = _moments(ref_se), _moments(arm_se)
    pa, pb = _pip(la), _pip(lb)
    dpip = np.abs(pa - pb)
    out, a_ref, a_arm = [], [], []
    for i, k in enumerate(match(la, lb)):
        dr, da = bool(lbf_a[i] >= R.MIN_LOG_BF), bool(lbf_b[k] >= R.MIN_LOG_BF)
        if dr or da:
            ai, ak = np.exp(la[i]), np.exp(lb[k])
            keep = np.maximum(ai, ak) >= PIP_FLOOR
            a_ref.append(ai[keep])
            a_arm.append(ak[keep])
        d = {**_dist(la[i], lb[k]), **_cond(la[i], mu_a[i], var_a[i], mu_b[k], var_b[k])}
        j = int(la[i].argmax())   # CAVI-Q2's top feature for this component
        jb = int(lb[k].argmax())
        out.append({**d, "kl_joint": d["kl"] + d["kl_cond"], "abs_lvar": abs(d["lvar"]),
                    "sep": bool(sep is not None and (sep[j] or sep[jb])),
                    "mu_ref": float(mu_a[i, j]), "mu_arm": float(mu_b[k, j]),
                    "var_ref": float(var_a[i, j]), "var_arm": float(var_b[k, j]),
                    "declared_ref": dr, "declared_arm": da,
                    "lbf_ref": float(lbf_a[i]), "lbf_arm": float(lbf_b[k]),
                    "pv_ref": float(ref_se[i]["prior_variance"]),
                    "pv_arm": float(arm_se[k]["prior_variance"]),
                    "min_log_bf": float(min(lbf_a[i], lbf_b[k]))})
    fit = {"max_pip": float(dpip.max()), "argmax_pip": int(dpip.argmax()),
           "n_decl_ref": int((lbf_a >= R.MIN_LOG_BF).sum()),
           "n_decl_arm": int((lbf_b >= R.MIN_LOG_BF).sum())}
    alphas = (np.concatenate(a_ref), np.concatenate(a_arm)) if a_ref else (np.empty(0), np.empty(0))
    return out, fit, pa, pb, alphas




def load(sources=SOURCES, results_root: str = RESULTS
         ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(component-pair frame, per-fit frame, feature frame, matched-alpha frame).

    * component pairs: one row per matched component pair (all L), with each side's
      declaration and the smaller of the two component log BFs.
    * fits: one row per (arm, replicate): max |dPIP| and the feature attaining it, declared
      component counts on each side, q2 ELBO difference (arm - CAVI-Q2), and both fits.parquet
      paths so a single replicate can be re-read (``replicate_pair``).
    * features: one row per (arm, replicate, feature) with PIP >= PIP_FLOOR on either side;
      every other feature has both PIPs below the floor.
    * matched alphas: one row per (arm, replicate, matched pair either side declares, feature)
      with alpha >= PIP_FLOOR on either side, columns alpha_ref / alpha_arm.

    Cells carry exp, L, m, T, depth, gap, null."""
    cfg = loader.load_config()
    comp_rows, fit_rows, feat_parts, alpha_parts = [], [], [], []
    for exp, L, sc in sources:
        by_batch: dict[str, dict] = {}
        for coll in loader.collection_method_pairs(cfg, sc).values():
            for bh, mh, mname, _mc, scoord in coll["pairs"]:
                f = f"{results_root}/by_batch/{bh}/fits/{mh}/fits.parquet"
                if not os.path.exists(f):
                    continue
                cell = by_batch.setdefault(bh, {"meta": R.cell_meta(scoord), "scoord": scoord,
                                                "fits": {}})
                cell["fits"][R._method_key(mname)] = f
        for bh, cell in by_batch.items():
            if "cavi" not in cell["fits"]:
                continue
            meta = {k: cell["meta"][k] for k in ("m", "T", "depth", "gap", "null")}
            cols = ["replicate", "single_effects", "q2_elbo"]
            ref = {r["replicate"]: r for r in
                   pl.read_parquet(cell["fits"]["cavi"], columns=cols).iter_rows(named=True)}
            sep = {}
            if not meta["null"]:
                spec = loader.resolve_simulation_from_coord(cell["scoord"])
                for rep in ref:
                    sim = PF.simulate_dense(spec, rep)
                    sep[rep] = separated(sim.X, np.asarray(sim.z, dtype=float))
            for method, f in cell["fits"].items():
                if method == "cavi":
                    continue
                for r in pl.read_parquet(f, columns=cols).iter_rows(named=True):
                    rep = r["replicate"]
                    if rep not in ref:
                        continue
                    pairs, fit, pa, pb, (ar, aa) = compare(ref[rep]["single_effects"], r["single_effects"],
                                                           sep.get(rep))
                    key = {"exp": exp, "L": L, "method": method, "batch_hash": bh, "rep": rep, **meta}
                    fit_rows.append({**key, **fit, "n_pairs": sum(p["declared_ref"] or p["declared_arm"] for p in pairs),
                                     "d_elbo": r["q2_elbo"] - ref[rep]["q2_elbo"],
                                     "ref_file": cell["fits"]["cavi"], "arm_file": f})
                    comp_rows.extend({**key, **p} for p in pairs)
                    keep = np.flatnonzero(np.maximum(pa, pb) >= PIP_FLOOR)
                    feat_parts.append(pl.DataFrame({
                        "exp": exp, "L": L, "method": method, "null": meta["null"],
                        "j": keep.astype(np.int32), "pip_ref": pa[keep], "pip_arm": pb[keep]}))
                    alpha_parts.append(pl.DataFrame({
                        "exp": exp, "L": L, "method": method, "null": meta["null"],
                        "alpha_ref": ar, "alpha_arm": aa}))
    return (pl.DataFrame(comp_rows, infer_schema_length=None),
            pl.DataFrame(fit_rows, infer_schema_length=None),
            pl.concat(feat_parts), pl.concat(alpha_parts))


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


# Display names for the three designs, and the order they appear in (binary designs together).
DESIGN = {"022": "Binary", "024": "Binary-Block", "023": "Gaussian"}
EXPS = list(DESIGN)
_AXIS_LABEL = {"022": "set size $m$", "023": "signal $T$", "024": "depth $d$"}


def with_design(df: pl.DataFrame) -> pl.DataFrame:
    """Adds a `design` display column (Binary / Binary-Block / Gaussian) from `exp`."""
    return df.with_columns(pl.col("exp").replace_strict(DESIGN).alias("design"))


def figure_legend(fig, axes, rect_bottom: float = 0.0, height: float = 1.0):
    """One legend above the panels with every arm drawn in ANY panel, in R.METHODS order (an
    arm missing from the first panel, e.g. an L=5-only arm in an L=1 row, still gets an
    entry); then tight_layout below it. `height` is the figure height in inches."""
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [R.METHOD_LABEL[m] for m in ARMS if R.METHOD_LABEL[m] in handles]
    ncol = min(len(order), 5) or 1
    nrow = -(-len(order) // ncol)
    fig.legend([handles[k] for k in order], order, loc="upper center", ncol=ncol, fontsize=7,
               frameon=False)
    fig.tight_layout(rect=(0, rect_bottom, 1, 1 - (0.08 + 0.17 * nrow) / height))


def by_axis_figure(comp: pl.DataFrame, L: int, metrics: dict | None = None):
    """Rows = metrics (default BY_AXIS_METRICS: TV for alpha, |dmu|/sd for the conditional
    means), columns = experiments; mean per cell vs the sweep axis with a replicate-bootstrap
    95% interval, one line per arm; non-null cells, log y."""
    import matplotlib.pyplot as plt
    exps = EXPS
    metrics = metrics or BY_AXIS_METRICS
    height = 0.6 + 2.0 * len(metrics)
    fig, axes = plt.subplots(len(metrics), 3, figsize=(6.5, height), sharex="col", sharey="row",
                             squeeze=False)
    sub_l = comp.filter((pl.col("L") == L) & ~pl.col("null"))
    for i, (metric, label) in enumerate(metrics.items()):
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
            if i == 0:
                ax.set_title(DESIGN[exp], fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel(_AXIS_LABEL[exp], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
    figure_legend(fig, axes, height=height)
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
                    unit: str = "fits", floor: float = 1e-4, xmax: float = 1.0):
    """Rows = (L, column, label[, filter expr]) (default: max_pip at L = 1 and L = 5), columns =
    experiments: share of fits with the column's value > x, one line per arm, log-log. Values
    below `floor` are drawn at the floor."""
    import matplotlib.pyplot as plt
    exps = EXPS
    rows = rows or [(1, "max_pip", "L = 1"), (5, "max_pip", "L = 5")]
    height = 0.5 + 2.2 * len(rows)
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.5, height), sharex=True,
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
            ax.set_xlim(floor, xmax)
            ax.set_ylim(1e-3, 1.05)
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(DESIGN[exp], fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{label}\nshare of {unit} above x", fontsize=8)
    figure_legend(fig, axes, height=height)
    return fig


BARS = [0.0, 1.0, 2.0]


def bar_rows(L: int, bars=BARS, col: str = "tv") -> list:
    """survival_figure rows: all matched pairs, then pairs whose smaller component log BF
    exceeds each bar (both sides must clear it)."""
    return [(L, col, "all pairs")] + [(L, col, f"min log BF > {t:g}", pl.col("min_log_bf") > t)
                                      for t in bars]


def scatter_figure(feat: pl.DataFrame, L: int = 5, quantity: str = "pip"):
    """Rows = experiments, columns = arms: per-feature PIP (`quantity="pip"`, the feature
    frame) or matched alpha (`"alpha"`, the matched-alpha frame), arm vs CAVI-Q2, pooled over
    replicates and non-null cells, as a hexbin with log counts. Only features with a value >=
    PIP_FLOOR on either side (every other feature sits in the corner below it)."""
    name = {"pip": "PIP", "alpha": "α"}[quantity]
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    exps = EXPS
    fig, axes = plt.subplots(3, len(ARMS), figsize=(6.5, 3.1), sharex=True, sharey=True)
    sub_l = feat.filter((pl.col("L") == L) & ~pl.col("null"))
    for i, exp in enumerate(exps):
        for k, m in enumerate(ARMS):
            ax = axes[i, k]
            d = sub_l.filter((pl.col("exp") == exp) & (pl.col("method") == m))
            ax.hexbin(d[f"{quantity}_ref"].to_numpy(), d[f"{quantity}_arm"].to_numpy(), gridsize=30,
                      extent=(0, 1, 0, 1), norm=LogNorm(vmin=1, vmax=1e5), cmap="Blues",
                      mincnt=1, linewidths=0)
            ax.plot([0, 1], [0, 1], color="0.5", lw=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.tick_params(labelsize=5, length=2, pad=1)
            if i == 0:
                ax.set_title(R.METHOD_LABEL[m], fontsize=6.5, color=R.METHOD_COLOR[m])
            if i == 2:
                ax.set_xlabel(f"{name}, CAVI-Q2", fontsize=6)
            if k == 0:
                ax.set_ylabel(f"{DESIGN[exp]}\n{name}, arm", fontsize=6.5)
    fig.tight_layout(w_pad=0.2, h_pad=0.3)
    return fig


def moment_scatter_figure(comp: pl.DataFrame, L: int = 5, moment: str = "mu"):
    """Rows = experiments, columns = arms: the conditional posterior mean (`moment="mu"`) or
    variance (`"var"`) of b at CAVI-Q2's top feature of each matched pair, arm against CAVI-Q2, as
    a hexbin with log counts. Pairs that BOTH sides declare, non-null cells, of the frame passed in
    (the caller drops separated pairs). Mean axes are linear and shared within a row (the designs
    use different effect scales); variance axes are log10."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    exps = EXPS
    sub_l = comp.filter((pl.col("L") == L) & ~pl.col("null") & pl.col("declared_ref")
                        & pl.col("declared_arm"))
    fig, axes = plt.subplots(3, len(ARMS), figsize=(6.5, 3.0), squeeze=False)
    ref, arm = f"{moment}_ref", f"{moment}_arm"
    for i, exp in enumerate(exps):
        row = sub_l.filter(pl.col("exp") == exp)
        both = np.concatenate([row[ref].to_numpy(), row[arm].to_numpy()])
        if moment == "var":
            lo, hi = np.log10(np.quantile(both, [0.0005, 0.9995]))
            lo, hi = lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)
        else:
            lo, hi = np.quantile(both, [0.001, 0.999])
            pad = 0.05 * (hi - lo)
            lo, hi = lo - pad, hi + pad
        for k, m in enumerate(ARMS):
            ax = axes[i, k]
            d = row.filter(pl.col("method") == m)
            x, y = d[ref].to_numpy(), d[arm].to_numpy()
            if moment == "var":
                x, y = np.log10(x), np.log10(y)
            ax.hexbin(x, y, gridsize=25, extent=(lo, hi, lo, hi), norm=LogNorm(vmin=1, vmax=1e3),
                      cmap="Blues", mincnt=1, linewidths=0)
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=5, length=2, pad=1)
            if k:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(R.METHOD_LABEL[m], fontsize=6.5, color=R.METHOD_COLOR[m])
            if k == 0:
                ax.set_ylabel(f"{DESIGN[exp]}\narm", fontsize=6.5)
    what = "posterior mean" if moment == "mu" else "log10 posterior variance"
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.2, h_pad=0.3)
    fig.text(0.5, 0.01, f"conditional {what}, CAVI-Q2", ha="center", fontsize=7)
    return fig


def _slog(x):
    """sign(x) log10(1 + |x|): linear near 0, log in both tails (log BFs span -5 to 1e3)."""
    return np.sign(x) * np.log10(1.0 + np.abs(x))


def _pair_scatter(comp: pl.DataFrame, L: int, col: str, transform, lim: tuple[float, float],
                  ticks: list[float], label: str, guides: list[float]):
    """Rows = experiments, columns = arms: per matched pair, `{col}_arm` against `{col}_ref`
    on the `transform`ed scale, hexbin with log counts; grey y = x, dashed lines at `guides`
    (raw units). Every matched pair, declared or not, non-null cells."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    sub_l = comp.filter((pl.col("L") == L) & ~pl.col("null"))
    lo, hi = lim
    fig, axes = plt.subplots(3, len(ARMS), figsize=(6.5, 3.1), sharex=True, sharey=True)
    for i, exp in enumerate(EXPS):
        for k, m in enumerate(ARMS):
            ax = axes[i, k]
            d = sub_l.filter((pl.col("exp") == exp) & (pl.col("method") == m))
            ax.hexbin(transform(d[f"{col}_ref"].to_numpy()), transform(d[f"{col}_arm"].to_numpy()),
                      gridsize=30, extent=(lo, hi, lo, hi), norm=LogNorm(vmin=1, vmax=1e4),
                      cmap="Blues", mincnt=1, linewidths=0)
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.5)
            for g in guides:
                ax.axvline(transform(g), color="0.6", lw=0.5, ls="--")
                ax.axhline(transform(g), color="0.6", lw=0.5, ls="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xticks([transform(t) for t in ticks], [f"{t:g}" for t in ticks])
            ax.set_yticks([transform(t) for t in ticks], [f"{t:g}" for t in ticks])
            ax.tick_params(labelsize=4.5, length=2, pad=1)
            if i == 0:
                ax.set_title(R.METHOD_LABEL[m], fontsize=6.5, color=R.METHOD_COLOR[m])
            if k == 0:
                ax.set_ylabel(DESIGN[exp], fontsize=6.5)
    fig.tight_layout(rect=(0.025, 0.035, 1, 1), w_pad=0.2, h_pad=0.3)
    fig.text(0.5, 0.008, f"{label}, CAVI-Q2", ha="center", fontsize=7)
    fig.text(0.005, 0.5, f"{label}, arm", va="center", rotation="vertical", fontsize=7)
    return fig


def lbf_scatter_figure(comp: pl.DataFrame, L: int = 5):
    """Component log BF of every matched pair, arm against CAVI-Q2, on a sign-log scale
    (sign(x) log10(1 + |x|)); dashed lines at the declaration bar R.MIN_LOG_BF."""
    sub_l = comp.filter((pl.col("L") == L) & ~pl.col("null"))
    top = np.quantile(np.concatenate([sub_l["lbf_ref"].to_numpy(), sub_l["lbf_arm"].to_numpy()]), 0.999)
    hi = max(_slog(top), _slog(100.0))
    ticks = [t for t in (-10, 0, 2, 10, 100, 1000) if _slog(t) <= hi]
    return _pair_scatter(comp, L, "lbf", _slog, (_slog(-10.0), hi), ticks, "log BF", [R.MIN_LOG_BF])


PV_FLOOR, PV_CAP = 0.01, 100.0   # freeze_prior_variance and max_prior_variance of every L=5 arm


def pv_scatter_figure(comp: pl.DataFrame, L: int = 5):
    """Estimated prior variance of every matched pair's component, arm against CAVI-Q2, log10
    scale; dashed lines at the freeze floor (0.01) and the cap (100)."""
    lo, hi = np.log10(PV_FLOOR) - 0.3, np.log10(PV_CAP) + 0.3
    return _pair_scatter(comp, L, "pv", np.log10, (lo, hi), [0.01, 1, 100], "prior variance",
                         [PV_FLOOR, PV_CAP])


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
                title += f"   ({DESIGN[row['exp']]}: {cellname}, rep {row['rep']})"
            else:
                title += f"   max |ΔPIP| {row['max_pip']:.2f}, ΔELBO {row['d_elbo']:+.2f}"
            ax.set_title(title, fontsize=7, loc="left")
    fig.tight_layout()
    return fig

