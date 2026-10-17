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

Conditionals q(b | gamma = j) = N(mu_j, var_j), arm vs CAVI-Q2, averaged over CAVI-Q2's alpha:
dmu = sum_j alpha_j |mu'_j - mu_j| / sd_j (shift in CAVI-Q2 posterior SDs) and
lsd = sum_j alpha_j log(sd'_j / sd_j) (signed, > 0 when the arm is wider).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG = os.path.dirname(os.path.dirname(_HERE))
for _p in (_TG, _HERE, os.path.join(os.path.dirname(_HERE), "logistic_laplace_simulations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from experiments import loader

import rate_cs as RC

MIN_LOG_BF = RC.MIN_LOG_BF
ARMS = ["smoothed", "taylor2", "gibss", "laplace", "score"]
LABEL = {"gibss": "gIBSS-Q2", "laplace": "gIBSS-Laplace", "smoothed": "gIBSS-GO", "taylor2": "Taylor2", "score": "score"}
COLOR = {"smoothed": "#000000", "taylor2": "#7F7F7F", "gibss": "#0072B2", "laplace": "#009E73", "score": "#CC79A7"}
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


def _moments(se: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(conditional mean [L, p], conditional variance [L, p])."""
    return (np.stack([np.asarray(e["mu"], dtype=float) for e in se]),
            np.stack([np.asarray(e["var"], dtype=float) for e in se]))


def _cond(a: np.ndarray, mu: np.ndarray, var: np.ndarray, mu_b: np.ndarray, var_b: np.ndarray) -> dict:
    lsd = float(np.sum(a * 0.5 * np.log(var_b / var)))
    return {"dmu": float(np.sum(a * np.abs(mu_b - mu) / np.sqrt(var))), "lsd": lsd, "abs_lsd": abs(lsd)}


PIP_FLOOR = 0.05   # feature frames keep features with PIP (alpha) >= this on either side
ETA_ROWS = 10      # observations per replicate kept for the linear-predictor scatters (same rows for every arm)


def separated(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per feature: True when the counts are all zero over the rows with x_j != 0, or over the
    (non-empty) rows with x_j == 0. The Poisson likelihood then has no maximum in b_j (it runs to
    -inf or +inf), so the conditional moments are set by the prior. Never true for a dense
    continuous column unless y is all zero."""
    on = X != 0
    y_on = y @ on
    n_off, y_off = len(y) - on.sum(0), y.sum() - y_on
    return (y_on == 0) | ((n_off > 0) & (y_off == 0))


def predict(fit: dict, Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and variance of each observation's linear predictor psi_i = b0 + x~_i . b,
    x~ the centered design (the fits' center=True convention), under the mean-field posterior:
    E = m0 + sum_l sum_j alpha_lj mu_lj x~_ij,
    Var = v0 + sum_l (sum_j alpha_lj (mu_lj^2 + var_lj) x~_ij^2 - (sum_j alpha_lj mu_lj x~_ij)^2)."""
    se, fs = fit["single_effects"], fit["family_state"]
    A = np.array([e["alpha"] for e in se])
    M = np.array([e["mu"] for e in se])
    V = np.array([e["var"] for e in se])
    Eb = A * M
    m = fs["intercept_value"] + Xc @ Eb.sum(0)
    v = fs["intercept_var"] + (Xc ** 2) @ (A * (M ** 2 + V)).sum(0) - ((Xc @ Eb.T) ** 2).sum(1)
    return m, v


def compare(ref_se: list[dict], arm_se: list[dict], sep: np.ndarray | None = None):
    """(component pairs, per-fit summary, PIP vectors (reference, arm), matched alphas
    (reference, arm) of the pairs either side declares at features with alpha >= PIP_FLOOR on
    either side). Pairs carry the conditional moments at CAVI-Q2's top feature, both prior
    variances and log BFs, and `sep` when either side's top feature separates y."""
    a, lbf_a = _components(ref_se)
    b, lbf_b = _components(arm_se)
    (mu_a, var_a), (mu_b, var_b) = _moments(ref_se), _moments(arm_se)
    pa, pb = 1 - np.prod(1 - a, axis=0), 1 - np.prod(1 - b, axis=0)
    tv = 0.5 * np.abs(a[:, None, :] - b[None, :, :]).sum(-1)
    _, cols = linear_sum_assignment(tv)
    pairs, a_ref, a_arm = [], [], []
    for i, k in enumerate(cols):
        dr, da = bool(lbf_a[i] >= MIN_LOG_BF), bool(lbf_b[k] >= MIN_LOG_BF)
        if dr or da:
            keep = np.maximum(a[i], b[k]) >= PIP_FLOOR
            a_ref.append(a[i][keep])
            a_arm.append(b[k][keep])
        j, jb = int(a[i].argmax()), int(b[k].argmax())
        pairs.append({**_dist(a[i], b[k]), **_cond(a[i], mu_a[i], var_a[i], mu_b[k], var_b[k]),
                      "declared_ref": dr, "declared_arm": da,
                      "sep": bool(sep is not None and (sep[j] or sep[jb])),
                      "mu_ref": float(mu_a[i, j]), "mu_arm": float(mu_b[k, j]),
                      "var_ref": float(var_a[i, j]), "var_arm": float(var_b[k, j]),
                      "pv_ref": float(ref_se[i]["prior_variance"]),
                      "pv_arm": float(arm_se[k]["prior_variance"]),
                      "lbf_ref": float(lbf_a[i]), "lbf_arm": float(lbf_b[k]),
                      "min_log_bf": float(min(lbf_a[i], lbf_b[k]))})
    fit = {"max_pip": float(np.abs(pa - pb).max()),
           "n_decl_ref": int((lbf_a >= MIN_LOG_BF).sum()),
           "n_decl_arm": int((lbf_b >= MIN_LOG_BF).sum())}
    alphas = (np.concatenate(a_ref), np.concatenate(a_arm)) if a_ref else (np.empty(0), np.empty(0))
    return pairs, fit, pa, pb, alphas


FRAMES = ("comp", "fit", "feat", "alpha", "eta")


def _cells(sc: str):
    """(batch hash, fit hash, arm, cell meta, simulation coord) per fit of a supercollection."""
    design = sc.split("-")[2]
    seen = set()
    for coll in loader.collection_method_pairs(loader.load_config(), sc).values():
        for bh, mh, mname, _mc, scoord in coll["pairs"]:
            if (bh, mh) not in seen:
                seen.add((bh, mh))
                yield bh, mh, mname.rsplit("_", 1)[-1], RC._meta(design, scoord), scoord


def load(sc: str) -> dict[str, pl.DataFrame]:
    """Frames for one 025 supercollection, cached on the number of fits.parquet files present:

    * comp: one row per matched component pair (all of them), distances and both sides' moments.
    * fit: one row per (arm, replicate): max |dPIP|, declared counts, q2 ELBO difference, and
      (signal cells) psi_dm / psi_lvar, the mean |dE psi_i| / sd_i and mean log variance ratio.
    * feat: per (arm, replicate, feature) with PIP >= PIP_FLOOR on either side.
    * alpha: per (arm, replicate, matched pair either side declares, feature) with alpha >=
      PIP_FLOOR on either side.
    * eta: ETA_ROWS observations per (arm, replicate), the same rows for every arm, with psi's
      posterior mean (eta_m_*) and variance (eta_v_*). Signal cells only.

    Signal cells re-run the pipeline's simulation per replicate for the design (separation
    flags, psi)."""
    import posterior_figure as PF
    os.makedirs(RC.CACHE, exist_ok=True)
    present = [c for c in _cells(sc)
               if os.path.exists(f"{RC.RESULTS}/by_batch/{c[0]}/fits/{c[1]}/fits.parquet")]
    stem = os.path.join(RC.CACHE, f"rate_pip_v4_{sc}_{len(present)}")
    if all(os.path.exists(f"{stem}_{k}.parquet") for k in FRAMES):
        return {k: pl.read_parquet(f"{stem}_{k}.parquet") for k in FRAMES}
    by_batch: dict[str, dict] = {}
    for bh, mh, arm, meta, scoord in present:
        cell = by_batch.setdefault(bh, {"meta": meta, "scoord": scoord, "fits": {}})
        cell["fits"][arm] = f"{RC.RESULTS}/by_batch/{bh}/fits/{mh}/fits.parquet"
    rows = {k: [] for k in FRAMES}
    cols = ["replicate", "single_effects", "family_state", "q2_elbo"]
    for bh, cell in by_batch.items():
        if "cavi" not in cell["fits"]:
            continue
        meta = cell["meta"]
        ref = {r["replicate"]: r for r in pl.read_parquet(cell["fits"]["cavi"], columns=cols).iter_rows(named=True)}
        sep, xc, pref, obs = {}, {}, {}, {}
        if not meta["null"]:
            spec = loader.resolve_simulation_from_coord(cell["scoord"])
            for rep, ref_fit in ref.items():
                sim = PF.simulate_dense(spec, rep)
                sep[rep] = separated(sim.X, np.asarray(sim.y_count, dtype=float))
                xc[rep] = sim.X - sim.X.mean(0)
                pref[rep] = predict(ref_fit, xc[rep])
                obs[rep] = np.random.default_rng(rep).choice(len(xc[rep]), ETA_ROWS, replace=False)
        for arm, f in cell["fits"].items():
            if arm == "cavi":
                continue
            for r in pl.read_parquet(f, columns=cols).iter_rows(named=True):
                rep = r["replicate"]
                if rep not in ref:
                    continue
                cp, fit, pa, pb, (ar, aa) = compare(ref[rep]["single_effects"], r["single_effects"], sep.get(rep))
                key = {"method": arm, "batch_hash": bh, "rep": rep, **meta}
                if rep in xc:
                    (m0, v0), (m1, v1) = pref[rep], predict(r, xc[rep])
                    fit["psi_dm"] = float(np.mean(np.abs(m1 - m0) / np.sqrt(v0)))
                    fit["psi_lvar"] = float(np.mean(np.log(v1 / v0)))
                    k = obs[rep]
                    rows["eta"].append(pl.DataFrame({
                        "method": arm, "lambda0": meta["lambda0"], "eta_m_ref": m0[k], "eta_m_arm": m1[k],
                        "eta_v_ref": v0[k], "eta_v_arm": v1[k]}))
                e_ref, e_arm = ref[rep]["q2_elbo"], r["q2_elbo"]
                d_elbo = None if e_ref is None or e_arm is None else e_arm - e_ref
                rows["fit"].append({**key, **fit, "n_pairs": len(cp), "d_elbo": d_elbo})
                rows["comp"].extend({**key, **p} for p in cp)
                keep = np.flatnonzero(np.maximum(pa, pb) >= PIP_FLOOR)
                rows["feat"].append(pl.DataFrame({
                    "method": arm, "lambda0": meta["lambda0"], "null": meta["null"],
                    "pip_ref": pa[keep], "pip_arm": pb[keep]}))
                rows["alpha"].append(pl.DataFrame({
                    "method": arm, "lambda0": meta["lambda0"], "null": meta["null"],
                    "alpha_ref": ar, "alpha_arm": aa}))
    out = {"comp": pl.DataFrame(rows["comp"], infer_schema_length=None),
           "fit": pl.DataFrame(rows["fit"], infer_schema_length=None),
           **{k: pl.concat(rows[k], how="vertical_relaxed") for k in ("feat", "alpha", "eta")}}
    for k, df in out.items():
        df.write_parquet(f"{stem}_{k}.parquet")
    return out


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
        comp = load(f"025-rate-{d}" + ("" if L == 5 else "-ser"))["comp"]
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
            fits = load(f"025-rate-{d}" + ("" if L == 5 else "-ser"))["fit"]
            fits = fits.filter(~pl.col("null"))
            for m in ARMS:
                for grp, filt in (("low", pl.col("lambda0") < 1), ("high", pl.col("lambda0") >= 1)):
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
    handles += [Line2D([], [], color="0.3", ls="-", lw=1.2, label=r"$\lambda_0 < 1$"),
                Line2D([], [], color="0.3", ls="--", lw=1.2, label=r"$\lambda_0 \geq 1$")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def share_table() -> pl.DataFrame:
    """Per design, L, arm and rate group: share of fits with max |dPIP| <= 0.01 and > 0.1."""
    rows = []
    for d in DESIGNS:
        for L in (1, 5):
            fits = load(f"025-rate-{d}" + ("" if L == 5 else "-ser"))["fit"]
            fits = fits.filter(~pl.col("null")).with_columns(
                pl.when(pl.col("lambda0") < 1).then(pl.lit("< 1")).otherwise(pl.lit(">= 1")).alias("lambda0"))
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
    comp = load(f"025-rate-{design}" + ("" if L == 5 else "-ser"))["comp"]
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
            ax.set_xticks([1e-4, 1e-2, 1.0])
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


COND = {"dmu": "|Δμ| / σ", "abs_lsd": "|log σ′/σ|"}


def cond_survival_figure(design: str, L: int = 5, bar: float = 2.0, floor: float = 1e-4,
                         xmax: float = 10.0):
    """Rows = conditional mean shift and |log SD ratio| (COND), columns = lambda0: share of
    matched pairs whose smaller component log BF exceeds `bar` with the metric > x, one line per
    arm, log-log, non-null cells pooled over T and gap. Values below `floor` are drawn at it."""
    import matplotlib.pyplot as plt
    comp = load(f"025-rate-{design}" + ("" if L == 5 else "-ser"))["comp"]
    comp = comp.filter(~pl.col("null") & (pl.col("min_log_bf") > bar))
    lams = sorted(comp["lambda0"].unique().to_list())
    fig, axes = plt.subplots(len(COND), len(lams), figsize=(6.5, 1.55 * len(COND) + 0.7),
                             sharex=True, sharey=True, squeeze=False)
    for i, (col, label) in enumerate(COND.items()):
        for j, lam in enumerate(lams):
            ax = axes[i, j]
            sub = comp.filter(pl.col("lambda0") == lam)
            for m in ARMS:
                x = sub.filter(pl.col("method") == m)[col].to_numpy()
                x = np.sort(np.maximum(x[~np.isnan(x)], floor))
                if x.size == 0:
                    continue
                surv = 1.0 - np.arange(1, x.size + 1) / x.size
                ax.step(x[:-1], surv[:-1], where="post", color=COLOR[m], lw=1.1, label=LABEL[m])
            for v in (0.01, 0.1, 1.0):
                ax.axvline(v, color="0.6", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(floor, xmax)
            ax.set_xticks([1e-4, 1e-2, 1.0])
            ax.set_ylim(1e-3, 1.05)
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(rf"$\lambda_0 = {lam:g}$", fontsize=8)
            if i == len(COND) - 1:
                ax.set_xlabel("value, matched pair", fontsize=7)
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
    fig.suptitle(f"{DESIGN_SHORT[design]}, " + ("3 causals, L = 5" if L == 5 else "one causal, L = 1")
                 + f", min log BF > {bar:g}", fontsize=8, y=top + 0.25 / fig_h, va="bottom")
    return fig


# ------------------------------------------------------------------------------ scatters
PV_FREEZE, PV_CAP = 0.01, 100.0   # freeze_prior_variance (an effect stops updating once its prior
#                                   variance drops below it) and max_prior_variance, every arm
DESIGN = {"binary": "Binary", "block": "Binary-Block", "gaussian": "Gaussian"}   # display order


def _sc(design: str, L: int) -> str:
    return f"025-rate-{design}" + ("" if L == 5 else "-ser")


# quantity -> (frame, column stem, scale, axis label). Scales: "unit" = linear on [0, 1];
# "linear" / "log" = linear / log10 with limits from each row's data; "pv" = log10 like "log",
# with dashed guides at the freeze threshold and the cap; "slog" = sign(x) log10(1 + |x|), with
# dashed guides at the declaration bar.
SCATTERS = {
    "pip": ("feat", "pip", "unit", "PIP"),
    "eta_m": ("eta", "eta_m", "linear", "posterior mean of ψ"),
    "eta_v": ("eta", "eta_v", "log", "posterior variance of ψ"),
    "alpha": ("alpha", "alpha", "unit", "α, matched component"),
    "lbf": ("comp", "lbf", "slog", "component log BF"),
    "mu": ("comp", "mu", "linear", "conditional posterior mean"),
    "var": ("comp", "var", "log", "conditional posterior variance"),
    "pv": ("comp", "pv", "pv", "prior variance"),
}


def _slog(x):
    return np.sign(x) * np.log10(1 + np.abs(x))


def _scatter_rows(design: str, L: int, quantity: str) -> pl.DataFrame:
    """The rows a scatter draws: non-null cells; conditional moments only for pairs both sides
    declare and neither side's top feature separates y; prior variance for every matched pair."""
    frame, _, _, _ = SCATTERS[quantity]
    df = load(_sc(design, L))[frame]
    if "null" in df.columns:
        df = df.filter(~pl.col("null"))
    if quantity in ("mu", "var"):
        df = df.filter(pl.col("declared_ref") & pl.col("declared_arm") & ~pl.col("sep"))
    return df


def dropped(design: str, L: int, quantity: str) -> pl.DataFrame:
    """The rows `scatter_figure` leaves out because a value is non-finite (gIBSS-GO's runaway
    fits), per arm and lambda0."""
    col = SCATTERS[quantity][1]
    df = _scatter_rows(design, L, quantity)
    bad = ~pl.col(f"{col}_ref").is_finite() | ~pl.col(f"{col}_arm").is_finite()
    return df.filter(bad).group_by("method", "lambda0").len().sort("method", "lambda0")


def _transform(v: np.ndarray, scale: str) -> np.ndarray:
    if scale in ("log", "pv"):
        return np.log10(v)
    if scale == "slog":
        return _slog(v)
    return v


def _limits(df: pl.DataFrame, col: str, scale: str) -> tuple[float, float]:
    """Shared axis limits for one panel row: [0, 1] for unit scales, else the 0.05-99.95%
    quantiles of both sides, padded."""
    if scale == "unit":
        return 0.0, 1.0
    v = np.concatenate([df[f"{col}_ref"].to_numpy(), df[f"{col}_arm"].to_numpy()])
    if scale in ("log", "pv"):
        v = v[v > 0]
    v = _transform(v, scale)
    lo, hi = np.quantile(v, [0.0005, 0.9995]) if v.size else (0.0, 1.0)
    pad = 0.08 * (hi - lo) or 0.5
    return lo - pad, hi + pad


def _hex_panel(ax, d: pl.DataFrame, col: str, scale: str, lo: float, hi: float, gridsize: int = 25,
               sparse: bool = False):
    """One hexbin (log counts) of arm against CAVI-Q2 with grey y = x, guides and ticks for
    `scale` (see SCATTERS). `sparse` keeps fewer tick labels, for narrow panels side by side."""
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator
    log = scale in ("log", "pv")
    x, y = _transform(d[f"{col}_ref"].to_numpy(), scale), _transform(d[f"{col}_arm"].to_numpy(), scale)
    ax.hexbin(x, y, gridsize=gridsize, extent=(lo, hi, lo, hi), norm=LogNorm(vmin=1, vmax=1e4),
              cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.5)
    if scale == "pv":
        for g in (PV_FREEZE, PV_CAP):
            if lo < np.log10(g) < hi:
                ax.axvline(np.log10(g), color="0.6", lw=0.5, ls="--")
                ax.axhline(np.log10(g), color="0.6", lw=0.5, ls="--")
    if scale == "slog":
        ax.axvline(_slog(MIN_LOG_BF), color="0.6", lw=0.5, ls="--")
        ax.axhline(_slog(MIN_LOG_BF), color="0.6", lw=0.5, ls="--")
    if scale == "unit":
        ticks, labels = [0, 0.5, 1], ["0", "", "1"] if sparse else ["0", "0.5", "1"]
    elif log:
        ticks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
        if len(ticks) < 2:   # under a decade: half decades
            ticks = list(np.arange(np.ceil(2 * lo), np.floor(2 * hi) + 1) / 2)
        if len(ticks) > (2 if sparse else 4):
            ticks = [ticks[0], ticks[-1]] if sparse else ticks[::2]
        labels = [f"$10^{{{t:g}}}$" for t in ticks]
    elif scale == "slog":
        grid = (-100, 0, 10, 1000) if sparse else (-100, -10, 0, 2, 10, 100, 1000, 10000)
        ticks = [t for t in grid if lo <= _slog(t) <= hi]
        ticks, labels = [_slog(t) for t in ticks], [f"{t:g}" for t in ticks]
    elif sparse:
        ticks = [t for t in MaxNLocator(2).tick_values(lo, hi) if lo <= t <= hi]
        labels = [f"{t:g}" for t in ticks]
    else:
        ticks = None
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(3))
    if ticks is not None:
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
    if sparse and ticks is not None and len(ticks) >= 2:   # end labels inward: no collisions
        xt = ax.get_xticklabels()
        xt[0].set_ha("left")
        xt[-1].set_ha("right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=4.5, length=2, pad=1)


def _finite(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.filter(pl.col(f"{col}_ref").is_finite() & pl.col(f"{col}_arm").is_finite())


def scatter_figure(design: str, L: int, quantity: str):
    """Rows = lambda0, columns = arms: `quantity` (SCATTERS), arm against CAVI-Q2, as a hexbin
    with log counts; grey y = x."""
    import matplotlib.pyplot as plt
    _, col, scale, label = SCATTERS[quantity]
    df = _finite(_scatter_rows(design, L, quantity), col)
    arms = [m for m in ARMS if m in set(df["method"].unique().to_list())]
    lams = sorted(df["lambda0"].unique().to_list())
    fig, axes = plt.subplots(len(lams), len(arms), figsize=(6.5 * max(len(arms), 4) / 5, 1.2 * len(lams) + 0.4),
                             squeeze=False)
    for i, lam in enumerate(lams):
        row = df.filter(pl.col("lambda0") == lam)
        lo, hi = _limits(row, col, scale)
        for k, m in enumerate(arms):
            ax = axes[i, k]
            _hex_panel(ax, row.filter(pl.col("method") == m), col, scale, lo, hi)
            if k:
                ax.set_yticklabels([])
            if i < len(lams) - 1 and scale == "unit":
                ax.set_xticklabels([])
            if i == 0:
                ax.set_title(LABEL[m], fontsize=6.5, color=COLOR[m])
            if k == 0:
                ax.set_ylabel(rf"$\lambda_0 = {lam:g}$", fontsize=6.5)
    fig.tight_layout(rect=(0.025, 0.03, 1, 1), w_pad=0.2, h_pad=0.4)
    fig.text(0.5, 0.006, f"{label}, CAVI-Q2", ha="center", fontsize=7)
    fig.text(0.004, 0.5, f"{label}, arm", va="center", rotation="vertical", fontsize=7)
    return fig


RATE_SPLIT = 1.0   # summary_figure bins lambda0 into <= RATE_SPLIT (low rate) and > RATE_SPLIT


def summary_figure(L: int, quantity: str):
    """Rows = designs; columns = arms, each split into lambda0 <= 1 and lambda0 > 1: `quantity`
    (SCATTERS), arm against CAVI-Q2, as a hexbin with log counts; grey y = x. Axis limits are
    shared across arms within each (design, rate bin): beta and so the effect scale shrink as
    the rate grows. The per-design scatter_figure has one row per lambda0."""
    import matplotlib.pyplot as plt
    _, col, scale, label = SCATTERS[quantity]
    frames = {d: _finite(_scatter_rows(d, L, quantity), col) for d in DESIGN}
    arms = [m for m in ARMS if any(m in set(f["method"].unique().to_list()) for f in frames.values())]
    bins = [(r"$\lambda_0 \leq 1$", pl.col("lambda0") <= RATE_SPLIT), (r"$\lambda_0 > 1$", pl.col("lambda0") > RATE_SPLIT)]
    n, gap = len(arms), 0.35                            # gap = spacer between arms, in panel widths
    units = 2 * n + gap * (n - 1)
    left, right, bottom, top = 0.55, 0.08, 0.42, 0.42   # inches
    w = min(0.62, (6.5 - left - right) / units)        # square panel side, inches
    W, H = left + right + w * units, bottom + top + w * len(DESIGN) * 1.3
    fig = plt.figure(figsize=(W, H))
    ratios = []
    for k in range(n):
        ratios += [1, 1] + ([gap] if k < n - 1 else [])
    gs = fig.add_gridspec(len(DESIGN), len(ratios), width_ratios=ratios, wspace=0.12, hspace=0.3,
                          left=left / W, right=1 - right / W, bottom=bottom / H, top=1 - top / H)
    lims = {(d, j): _limits(f.filter(cond), col, scale) for d, f in frames.items()
            for j, (_, cond) in enumerate(bins)}
    for k, m in enumerate(arms):
        for i, d in enumerate(DESIGN):
            f = frames[d].filter(pl.col("method") == m)
            for j, (blabel, cond) in enumerate(bins):
                ax = fig.add_subplot(gs[i, 3 * k + j])
                _hex_panel(ax, f.filter(cond), col, scale, *lims[d, j], gridsize=20, sparse=True)
                if k or j:   # square panels about y = x: every panel's x ticks give its scale
                    ax.set_yticklabels([])
                if i == 0:
                    ax.set_title(blabel, fontsize=5.5, pad=2)
                    if j == 0:   # arm name centred over its pair of panels
                        pos = ax.get_position()
                        fig.text(pos.x1 + 0.03 * w / W, 1 - 0.1 / H, LABEL[m], ha="center", va="top",
                                 fontsize=7, color=COLOR[m])
                if j == 0 and k == 0:
                    ax.set_ylabel(DESIGN[d], fontsize=6.5)
    fig.text((left + w * units / 2) / W, 0.04 / H, f"{label}, CAVI-Q2", ha="center", va="bottom", fontsize=7)
    fig.text(0.04 / W, (bottom + w * len(DESIGN) * 1.3 / 2) / H, f"{label}, arm", va="center",
             rotation="vertical", fontsize=7)
    return fig
