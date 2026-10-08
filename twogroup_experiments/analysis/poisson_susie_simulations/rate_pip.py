"""025: agreement of each arm's variable-selection posterior with CAVI-Q2's, on the same replicate.

The Poisson analog of logistic_laplace_simulations/pip_agreement.py. Per single effect l, alpha_l
is the posterior over which feature carries the effect. Three distances, arm vs CAVI-Q2:

* TV  = 1/2 sum_j |alpha_j - alpha'_j| = max over feature SETS A of |alpha(A) - alpha'(A)|.
* KL  = KL(alpha_CAVI || alpha_arm). Large when the arm puts little mass where CAVI puts mass.
* max = max_j |alpha_j - alpha'_j|, the largest single-feature change.

Poisson fits store alpha but not feature_log_bf, so log alpha is log(alpha) floored at 1e-300
(alpha underflows to 0 only for features with no mass on either side, which add nothing to TV
and nothing to KL unless CAVI puts mass there). At L=1 alpha IS the PIP vector. At L=5
components are unordered, so the arm's are matched to CAVI's by minimum total TV (Hungarian).
Every matched pair is kept with the smaller of its two component log BFs, so figures can bar on
it (both sides must clear the bar). Per fit, max_j |PIP_arm - PIP_CAVI| over the combined PIPs
needs no matching.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import rate_cs as RC  # noqa: E402

MIN_LOG_BF = RC.MIN_LOG_BF
ARMS = ["gibss", "laplace", "score"]
LABEL = {"gibss": "gIBSS-Q2", "laplace": "gIBSS-Laplace", "score": "score"}
COLOR = {"gibss": "#0072B2", "laplace": "#009E73", "score": "#CC79A7"}
METRICS = {"tv": "TV", "kl": "KL", "max_alpha": "max |Δα|"}
DESIGNS = ["gaussian", "binary", "block"]
DESIGN_SHORT = {"gaussian": "Gaussian AR1", "binary": "binary Markov", "block": "nested block"}


def _components(se: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(alpha [L, p], component log BF [L])."""
    a = np.stack([np.asarray(e["alpha"], dtype=float) for e in se])
    return a, np.array([e["ser_log_bf"] for e in se], dtype=float)


def _dist(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    la, lb = np.log(np.maximum(a, 1e-300)), np.log(np.maximum(b, 1e-300))
    return {"tv": 0.5 * float(d.sum()), "kl": float(np.sum(a * (la - lb))), "max_alpha": float(d.max())}


def compare(ref_se: list[dict], arm_se: list[dict]) -> tuple[list[dict], dict]:
    a, lbf_a = _components(ref_se)
    b, lbf_b = _components(arm_se)
    pa, pb = 1 - np.prod(1 - a, axis=0), 1 - np.prod(1 - b, axis=0)
    tv = 0.5 * np.abs(a[:, None, :] - b[None, :, :]).sum(-1)
    _, cols = linear_sum_assignment(tv)
    pairs = []
    for i, k in enumerate(cols):
        pairs.append({**_dist(a[i], b[k]), "declared_ref": bool(lbf_a[i] >= MIN_LOG_BF),
                      "declared_arm": bool(lbf_b[k] >= MIN_LOG_BF),
                      "min_log_bf": float(min(lbf_a[i], lbf_b[k]))})
    return pairs, {"max_pip": float(np.abs(pa - pb).max()),
                   "n_decl_ref": int((lbf_a >= MIN_LOG_BF).sum()),
                   "n_decl_arm": int((lbf_b >= MIN_LOG_BF).sum())}


def load(sc: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(component-pair frame, per-fit frame) for one 025 supercollection, cached on the number of
    fits.parquet files present."""
    pairs = list(RC._pairs(sc))   # (bh, mh, arm, meta); arm in cavi/gibss/laplace/score
    present = [p for p in pairs if os.path.exists(f"{RC.RESULTS}/by_batch/{p[0]}/fits/{p[1]}/fits.parquet")]
    stem = os.path.join(RC.CACHE, f"rate_pip_v2_{sc}_{len(present)}")
    if os.path.exists(stem + "_fit.parquet"):
        return pl.read_parquet(stem + "_comp.parquet"), pl.read_parquet(stem + "_fit.parquet")
    by_batch: dict[str, dict] = {}
    for bh, mh, arm, meta in present:
        cell = by_batch.setdefault(bh, {"meta": meta, "fits": {}})
        cell["fits"][arm] = f"{RC.RESULTS}/by_batch/{bh}/fits/{mh}/fits.parquet"
    comp_rows, fit_rows = [], []
    cols = ["replicate", "single_effects", "q2_elbo"]
    for bh, cell in by_batch.items():
        if "cavi" not in cell["fits"]:
            continue
        ref = {r["replicate"]: r for r in pl.read_parquet(cell["fits"]["cavi"], columns=cols).iter_rows(named=True)}
        for arm, f in cell["fits"].items():
            if arm == "cavi":
                continue
            for r in pl.read_parquet(f, columns=cols).iter_rows(named=True):
                if r["replicate"] not in ref:
                    continue
                cp, fit = compare(ref[r["replicate"]]["single_effects"], r["single_effects"])
                key = {"method": arm, "batch_hash": bh, "rep": r["replicate"], **cell["meta"]}
                e_ref, e_arm = ref[r["replicate"]]["q2_elbo"], r["q2_elbo"]
                d_elbo = None if e_ref is None or e_arm is None else e_arm - e_ref
                fit_rows.append({**key, **fit, "n_pairs": len(cp), "d_elbo": d_elbo})
                comp_rows.extend({**key, **p} for p in cp)
    comp = pl.DataFrame(comp_rows, infer_schema_length=None)
    fits = pl.DataFrame(fit_rows, infer_schema_length=None)
    comp.write_parquet(stem + "_comp.parquet")
    fits.write_parquet(stem + "_fit.parquet")
    return comp, fits


def summarize(df: pl.DataFrame, value: str, by: list[str], n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Mean of `value` per group with a 95% CI from resampling whole replicates."""
    rng = np.random.default_rng(seed)
    out = []
    for key, g in df.group_by(by, maintain_order=True):
        x = g[value].to_numpy().astype(float)
        rep_id = (g["batch_hash"] + ":" + g["rep"].cast(pl.Utf8)).to_numpy()
        uniq, inv = np.unique(rep_id, return_inverse=True)
        sums, cnts = np.bincount(inv, weights=x), np.bincount(inv).astype(float)
        idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        means = sums[idx].sum(1) / cnts[idx].sum(1)
        out.append({**dict(zip(by, key)), "mean": float(x.mean()),
                    "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)), "n": len(x)})
    return pl.DataFrame(out)


def by_rate_figure(L: int):
    """Rows = TV, KL, max |dalpha| (declared component pairs), columns = designs; mean vs lambda0
    over non-null cells (pooled over T and gap) with a replicate-bootstrap 95% CI; log y."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.0), sharex=True, sharey="row")
    for j, d in enumerate(DESIGNS):
        comp, _ = load(f"025-rate-{d}" + ("" if L == 5 else "-ser"))
        sub = comp.filter(~pl.col("null"))
        for i, (metric, label) in enumerate(METRICS.items()):
            ax = axes[i, j]
            tab = summarize(sub, metric, ["method", "lambda0"]).sort("lambda0")
            for m in ARMS:
                t = tab.filter(pl.col("method") == m)
                if t.height == 0:
                    continue
                ax.errorbar(t["lambda0"].to_numpy(), t["mean"].to_numpy(),
                            yerr=[(t["mean"] - t["lo"]).to_numpy(), (t["hi"] - t["mean"]).to_numpy()],
                            color=COLOR[m], marker="o", ms=3.5, lw=1, capsize=0, label=LABEL[m])
            ax.set_xscale("log")
            ax.set_yscale("log")
            lams = sorted(tab["lambda0"].unique().to_list())
            ax.set_xticks(lams)
            ax.set_xticklabels([f"{v:g}" for v in lams], fontsize=7)
            ax.minorticks_off()
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title(DESIGN_SHORT[d], fontsize=8)
            if i == 2:
                ax.set_xlabel(r"$\lambda_0$", fontsize=8)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def survival_figure(floor: float = 1e-4):
    """Rows = L (1, 5), columns = designs, line style = lambda0 group: share of fits with
    max |dPIP| > x, one colour per arm, log-log. Values below `floor` are drawn at the floor."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.4), sharex=True, sharey=True)
    styles = {"low": "-", "high": "--"}
    for i, L in enumerate([1, 5]):
        for j, d in enumerate(DESIGNS):
            ax = axes[i, j]
            _, fits = load(f"025-rate-{d}" + ("" if L == 5 else "-ser"))
            fits = fits.filter(~pl.col("null"))
            for m in ARMS:
                for grp, filt in (("low", pl.col("lambda0") < 0.1), ("high", pl.col("lambda0") >= 1)):
                    x = np.sort(np.maximum(fits.filter((pl.col("method") == m) & filt)["max_pip"].to_numpy(), floor))
                    if x.size == 0:
                        continue
                    surv = 1.0 - np.arange(1, x.size + 1) / x.size
                    ax.step(x[:-1], surv[:-1], where="post", color=COLOR[m], lw=1.1, ls=styles[grp])
            for v in (0.01, 0.1):
                ax.axvline(v, color="0.6", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(floor, 1)
            ax.set_ylim(1e-3, 1.05)
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(DESIGN_SHORT[d], fontsize=8)
            if i == 1:
                ax.set_xlabel("max |ΔPIP| in the fit", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"L = {L}\nshare of fits above x", fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR[m], lw=1.2, label=LABEL[m]) for m in ARMS]
    handles += [Line2D([], [], color="0.3", ls="-", lw=1.2, label=r"$\lambda_0 = 0.01$"),
                Line2D([], [], color="0.3", ls="--", lw=1.2, label=r"$\lambda_0 \geq 1$")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def share_table() -> pl.DataFrame:
    """Per design, L, arm and rate group: share of fits with max |dPIP| <= 0.01 and > 0.1."""
    rows = []
    for d in DESIGNS:
        for L in (1, 5):
            _, fits = load(f"025-rate-{d}" + ("" if L == 5 else "-ser"))
            fits = fits.filter(~pl.col("null")).with_columns(
                pl.when(pl.col("lambda0") < 0.1).then(pl.lit("0.01")).otherwise(pl.lit(">= 1")).alias("lambda0"))
            for (m, g), h in fits.group_by("method", "lambda0"):
                rows.append({"design": DESIGN_SHORT[d], "L": L, "arm": LABEL[m], "lambda0": g,
                             "fits": h.height, "≤ 0.01": float((h["max_pip"] <= 0.01).mean()),
                             "> 0.1": float((h["max_pip"] > 0.1).mean())})
    order = {LABEL[m]: i for i, m in enumerate(ARMS)}
    return (pl.DataFrame(rows)
            .with_columns(pl.col("arm").replace_strict(order, return_dtype=pl.Int8).alias("_o"))
            .sort("design", "L", "_o", "lambda0").drop("_o"))


BARS = [0.0, 1.0, 2.0]


def tv_survival_figure(design: str, L: int = 5, bars=BARS, floor: float = 1e-4):
    """Rows = all matched pairs, then pairs whose smaller component log BF exceeds each bar;
    columns = lambda0. Share of matched pairs with TV(arm, CAVI-Q2) > x, one line per arm,
    log-log, non-null cells pooled over T and gap. Values below `floor` are drawn at the floor."""
    import matplotlib.pyplot as plt
    comp, _ = load(f"025-rate-{design}" + ("" if L == 5 else "-ser"))
    comp = comp.filter(~pl.col("null"))
    lams = sorted(comp["lambda0"].unique().to_list())
    rows = [("all pairs", pl.lit(True))] + [(f"min log BF > {t:g}", pl.col("min_log_bf") > t) for t in bars]
    fig, axes = plt.subplots(len(rows), len(lams), figsize=(6.5, 1.55 * len(rows) + 0.7),
                             sharex=True, sharey=True, squeeze=False)
    for i, (label, keep) in enumerate(rows):
        for j, lam in enumerate(lams):
            ax = axes[i, j]
            sub = comp.filter((pl.col("lambda0") == lam) & keep)
            for m in ARMS:
                x = sub.filter(pl.col("method") == m)["tv"].to_numpy()
                x = np.sort(np.maximum(x[~np.isnan(x)], floor))
                if x.size == 0:
                    continue
                surv = 1.0 - np.arange(1, x.size + 1) / x.size
                ax.step(x[:-1], surv[:-1], where="post", color=COLOR[m], lw=1.1, label=LABEL[m])
            for v in (0.01, 0.1):
                ax.axvline(v, color="0.6", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(floor, 1)
            ax.set_ylim(1e-3, 1.05)
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(rf"$\lambda_0 = {lam:g}$", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("TV, matched pair", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{label}\nshare above x", fontsize=7.5)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=len(handles), fontsize=7,
               frameon=False)
    fig_h = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig_h))
    top = max(ax.get_position().y1 for ax in axes[0])
    fig.suptitle(f"{DESIGN_SHORT[design]}, " + ("3 causals, L = 5" if L == 5 else "one causal, L = 1"),
                 fontsize=8, y=top + 0.25 / fig_h, va="bottom")
    return fig
