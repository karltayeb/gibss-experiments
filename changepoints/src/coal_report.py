"""Coal-example figures and tables, drawn from saved results. Each `fig_*` returns a
matplotlib Figure (and a table where one goes with it) without saving it, so the
summarize script and the notebook draw the same figures."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from coal import PROC
from stepbasis import cs_interval

RESULTS = ROOT / "results/coal"
FIG = RESULTS / "figures"
# Okabe-Ito, fixed order: component 1, 2, 3, ...; exact posterior is near-black.
COMP = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
EXACT = "#222222"
DATA = "#9a9a9a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})


def load_fit(res: str, tag: str) -> dict:
    with open(RESULTS / res / "fits" / f"{tag}.pkl", "rb") as fh:
        return pickle.load(fh)


def load_exact(res: str, shape: float, p: float) -> dict:
    z = np.load(RESULTS / res / "exact" / f"exact_shape{shape}_p{p}.npz")
    return {k: z[k] for k in z.files}


def series(res: str):
    d = pl.read_csv(PROC / f"{res}.csv")
    return d["label"].to_numpy(), d["count"].to_numpy().astype(float)


def shade_cs(ax, fit: dict, labels: np.ndarray, *, min_log_bf: float, alpha_declared=0.18,
             alpha_sub=0.07):
    """Shade each component's CS hull; declared components darker, sub-threshold faint."""
    for l in range(fit["L"]):
        lbf = fit["ser_log_bf"][l]
        if lbf < 0.5:
            continue
        lo, hi, _ = cs_interval(fit["cs"][l])
        # column j = jump between labels[j] and labels[j+1]: shade the boundary span
        x0, x1 = labels[lo] + 0.5 * (labels[lo + 1] - labels[lo]), labels[hi] + 0.5 * (labels[hi + 1] - labels[hi])
        ax.axvspan(x0, x1, color=COMP[l], alpha=alpha_declared if lbf >= min_log_bf else alpha_sub,
                   lw=0)


def fig_main(cfg: dict):
    """Counts, SuSiE fitted rate and CS hulls (L = max), exact mean rate; alphas vs the
    exact marginal boundary probabilities underneath."""
    years, y = series("year")
    L = cfg["fit"]["L"][-1]
    fit = load_fit("year", f"poisson_L{L}")
    ex = load_exact("year", cfg["exact"]["shape_default"], cfg["exact"]["p_default"])
    min_bf = cfg["fit"]["min_log_bf"]
    bnd = years[1:] - 0.5  # boundary between consecutive years, for the lower panel

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1.3], "hspace": 0.08})
    a0.bar(years, y, width=0.9, color=DATA, lw=0, label="disasters per year")
    a0.step(years, np.exp(fit["eta"]), where="mid", color=COMP[0], lw=1.6,
            label=f"SuSiE fitted rate (L={L})")
    a0.step(years, ex["mean_profile"], where="mid", color=EXACT, lw=1.2, ls="--",
            label=f"exact posterior mean rate (p={cfg['exact']['p_default']})")
    shade_cs(a0, fit, years, min_log_bf=min_bf)
    a0.set_ylabel("count / rate per year")
    a0.legend(loc="upper right", frameon=False, fontsize=8)
    a0.set_title("Coal-mining disasters 1851-1962: Poisson SuSiE on the step basis", fontsize=10)

    for l in range(fit["L"]):
        if fit["ser_log_bf"][l] < 0.5:
            continue
        lab = f"SuSiE effect {l + 1} alpha (log BF {fit['ser_log_bf'][l]:.1f})"
        a1.vlines(bnd, 0, fit["alpha"][l], color=COMP[l], lw=1.6, alpha=0.9, label=lab)
    a1.plot(bnd, ex["boundary_prob"], color=EXACT, lw=1.2, ls="--",
            label="exact P(changepoint between years)")
    a1.set_ylabel("probability")
    a1.set_xlabel("year")
    a1.legend(loc="upper left", frameon=False, fontsize=8)
    a1.set_xlim(years[0] - 1, years[-1] + 1)
    return fig


def fig_single(cfg: dict):
    """SuSiE L=1 alpha against the exact single-changepoint location posterior.
    Returns (fig, comparison dict)."""
    years, _ = series("year")
    fit1 = load_fit("year", "poisson_L1")
    bnd = years[1:] - 0.5
    sc = np.load(RESULTS / "year/exact/single_changepoint.npz")
    prob = sc["prob"]
    a_l1 = fit1["alpha"][0]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.vlines(bnd, 0, a_l1, color=COMP[0], lw=2.2, label=f"SuSiE L=1 alpha (CS {fit1['cs'][0].__len__()} years, log BF {fit1['ser_log_bf'][0]:.1f})")
    ax.plot(bnd, prob, "o-", color=EXACT, ms=3, lw=1, label=f"exact one-changepoint posterior (log BF {float(sc['log_bf10']):.1f})")
    ax.set_xlim(1875, 1910)
    ax.set_xlabel("year boundary")
    ax.set_ylabel("probability")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Single changepoint: SuSiE L=1 vs the exact posterior", fontsize=10)
    hs = np.argsort(prob)[::-1]
    k = int(np.searchsorted(np.cumsum(prob[hs]), 0.95)) + 1
    hpd = set(hs[:k].tolist())
    cs1 = set(fit1["cs"][0])
    return fig, {
        "L1_vs_exact_max_abs_diff": float(np.max(np.abs(a_l1 - prob))),
        "L1_vs_exact_total_variation": float(0.5 * np.abs(a_l1 - prob).sum()),
        "L1_cs": sorted(cs1), "exact_hpd": sorted(hpd),
        "L1_cs_eq_hpd": cs1 == hpd,
        "exact_mass_in_L1_cs": float(prob[sorted(cs1)].sum()),
        "L1_mode_year": int(years[int(np.argmax(a_l1)) + 1]),
        "exact_mode_year": int(years[int(np.argmax(prob)) + 1]),
    }


def fig_k_posterior(cfg: dict):
    """Exact P(k | y) for each prior changepoint rate, with SuSiE's declared counts.
    Returns (fig, exact grid table at the default Gamma shape)."""
    grid = pl.read_csv(RESULTS / "year/exact/grid_summary.csv")
    shape = cfg["exact"]["shape_default"]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for i, p in enumerate(cfg["exact"]["p"]):
        z = load_exact("year", shape, p)
        kp = z["k_prob"][:9]
        ax.plot(np.arange(len(kp)) + (i - 2) * 0.06, kp, "o-", ms=3.5, lw=1,
                color=plt.cm.Blues(0.35 + 0.15 * i), label=f"p = {p}")
    cs = pl.read_csv(RESULTS / "year/cs_table.csv")
    decl = cs.filter(pl.col("tag").str.starts_with("poisson_L") & pl.col("declared")) \
        .group_by("L").len().sort("L")
    for i, (_L, k) in enumerate(decl.iter_rows()):
        ax.axvline(k, color=COMP[0], lw=1.2, alpha=0.6,
                   label="SuSiE declared CSs, L=1,3,5" if i == 0 else None)
    ax.set_xlabel("number of changepoints k")
    ax.set_ylabel("exact posterior P(k | y)")
    ax.legend(frameon=False, fontsize=8, title="prior changepoint rate", title_fontsize=8)
    ax.set_title(f"How many changepoints? exact posterior (Gamma shape {shape}) vs SuSiE", fontsize=10)
    return fig, grid.filter(pl.col("shape") == shape).select(
        ["p_per_year", "prior_k_mean", "k_mean", "k_mode", "P_k1", "P_k2", "P_k_ge3",
         "boundary_mass_1885_1900", "boundary_mass_1925_1960"])


def fig_resolution(cfg: dict):
    """Yearly vs monthly alphas and exact boundary probabilities (L=5). Returns (fig, table)."""
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True,
                             gridspec_kw={"hspace": 0.1})
    for ax, res in zip(axes, ("year", "month")):
        labels, _y = series(res)
        fit = load_fit(res, "poisson_L5")
        step = labels[1] - labels[0]
        bnd = labels[1:] - 0.5 * step
        ex = load_exact(res, cfg["exact"]["shape_default"], cfg["exact"]["p_default"])
        for l in range(fit["L"]):
            if fit["ser_log_bf"][l] < 0.5:
                continue
            lo, hi, _ = cs_interval(fit["cs"][l])
            ax.vlines(bnd, 0, fit["alpha"][l], color=COMP[l], lw=1.2 if res == "year" else 0.6,
                      label=f"effect {l + 1} (log BF {fit['ser_log_bf'][l]:.1f})")
            rows.append({"resolution": res, "n": fit["n"], "component": l + 1,
                         "ser_log_bf": float(fit["ser_log_bf"][l]), "cs_size": len(fit["cs"][l]),
                         "cs_start": float(labels[lo + 1]), "cs_end": float(labels[hi + 1]),
                         "cs_calendar_width_years": float(labels[hi + 1] - labels[lo + 1] + step),
                         "top": float(labels[int(np.argmax(fit['alpha'][l])) + 1])})
        ax.plot(bnd, ex["boundary_prob"], color=EXACT, lw=0.9, ls="--", label="exact")
        ax.set_ylabel(f"{res}ly (n={fit['n']})")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[-1].set_xlabel("year")
    axes[0].set_title("Changepoint posteriors at yearly vs monthly resolution (L=5)", fontsize=10)
    return fig, pl.DataFrame(rows)


def fig_calibration_a():
    """Design A: declared CS hull widths and where the declared CS tops land."""
    cal = RESULTS / "year/calibration"
    a = pl.read_csv(cal / "design_a_components.csv").filter(pl.col("declared"))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for i, L in enumerate(sorted(a["L"].unique())):
        sub = a.filter(pl.col("L") == L)
        axes[0].hist(sub["cs_hull_width"].to_numpy(), bins=np.arange(0.5, 40, 1),
                     histtype="step", lw=1.4, color=COMP[i], label=f"L={L}")
        # where do the declared CS tops land
        axes[1].hist(1851 + sub["top"].to_numpy() + 1, bins=np.arange(1850, 1964, 2),
                     histtype="step", lw=1.4, color=COMP[i])
    axes[0].set_xlabel("declared CS hull width (years)")
    axes[0].set_ylabel("declared CSs")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("year of declared CS top (truth: 1892, 1948)")
    for t in (1892, 1948):
        axes[1].axvline(t, color=EXACT, lw=0.8, ls=":")
    fig.suptitle("Design A: simulated from the two-changepoint fit", fontsize=10)
    return fig


def fig_calibration_b():
    """Design B: coverage, set size and power against changepoint position and jump size."""
    cal = RESULTS / "year/calibration"
    b = pl.read_csv(cal / "design_b_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), layout="constrained")
    for i, rr in enumerate(sorted(b["rate_ratio"].unique())):
        sub = b.filter(pl.col("rate_ratio") == rr).sort("position")
        x = 1851 + sub["position"].to_numpy() + 1
        axes[0].plot(x, sub["L1_coverage"], "o-", ms=3, color=COMP[i], label=f"ratio {rr}")
        axes[0].plot(x, sub["exact_hpd_coverage"], "s--", ms=3, color=COMP[i], alpha=0.6)
        axes[1].plot(x, sub["L1_median_cs_size"], "o-", ms=3, color=COMP[i])
        axes[1].plot(x, sub["exact_median_hpd_size"], "s--", ms=3, color=COMP[i], alpha=0.6)
        axes[2].plot(x, sub["L1_power"], "o-", ms=3, color=COMP[i])
    axes[0].axhline(0.95, color=EXACT, lw=0.8, ls=":")
    axes[0].set_ylabel("coverage")
    axes[0].set_ylim(0.8, 1.01)
    axes[1].set_ylabel("median set size (years)")
    axes[2].set_ylabel("power")
    axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel("changepoint year")
    axes[0].legend(frameon=False, fontsize=7, title="solid SuSiE L=1\ndashed exact", title_fontsize=7)
    fig.suptitle("Design B: one changepoint, base rate 3/yr, n=112", fontsize=10)
    return fig


def _cs_blocks(cs, alpha):
    """Contiguous runs of a credible set as (lo, hi, alpha mass), heaviest first."""
    cols = np.array(sorted(cs))
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    return sorted(((int(r[0]), int(r[-1]), float(alpha[r].sum())) for r in runs),
                  key=lambda b: -b[2])


def fig_method_compare(cfg: dict, resolutions=("year",), methods=("poisson", "cf_cavi"),
                       L: int = 5):
    """Rows = resolutions, columns = methods (gIBSS default vs exact CAVI). Each panel shows
    the counts, the fitted rate, and every effect with log BF > 0.5: its credible set as a
    rug of member boundaries under the axis, and the contiguous block carrying most of its
    alpha shaded (strong for a declared effect, faint below the threshold)."""
    min_bf = cfg["fit"]["min_log_bf"]
    names = {"poisson": "gIBSS", "cf_cavi": "CAVI"}
    nr, nc = len(resolutions), len(methods)
    fig, axes = plt.subplots(nr, nc, figsize=(6.5, 3.3 * nr), squeeze=False,
                             sharex=True, sharey="row", layout="constrained")
    for i, res in enumerate(resolutions):
        labels, y = series(res)
        step = labels[1] - labels[0]
        bnd = labels[1:] - step / 2  # boundary between bin j and bin j+1
        unit = "year" if res == "year" else "month"
        ymax = max(y.max(), 1.0)
        for j, m in enumerate(methods):
            ax = axes[i, j]
            fit = load_fit(res, f"{m}_L{L}")
            ax.bar(labels, y, width=step * 0.9, color=DATA, lw=0, alpha=0.6 if res == "year" else 0.45)
            ax.step(labels, np.exp(fit["eta"]), where="mid", color=EXACT, lw=1.3,
                    label="fitted rate")
            shown = [l for l in range(fit["L"]) if fit["ser_log_bf"][l] >= 0.5]
            for r, l in enumerate(shown):
                a = fit["alpha"][l]
                cs = fit["cs"][l]
                lbf = float(fit["ser_log_bf"][l])
                declared = lbf >= min_bf
                lo, hi, mass = _cs_blocks(cs, a)[0]
                ax.axvspan(bnd[lo] - step / 2, bnd[hi] + step / 2, color=COMP[r],
                           alpha=0.25 if declared else 0.10, lw=0,
                           label=f"effect {l + 1}, log BF {lbf:.1f}"
                                 f"{'' if declared else ' (not declared)'}")
                y0 = -ymax * (0.05 + 0.06 * r)
                ax.vlines(bnd[list(cs)], y0 - ymax * 0.025, y0 + ymax * 0.025,
                          color=COMP[r], lw=0.8 if res == "year" else 0.4)
                ax.text(bnd[lo] - step / 2, ymax * 1.02, f"{mass:.0%}", color=COMP[r],
                        fontsize=7, va="bottom", ha="left")
            ax.axhline(0, color="#777777", lw=0.5)
            ax.set_ylim(-ymax * (0.05 + 0.06 * max(len(shown), 1)), ymax * 1.12)
            ax.set_title(f"{names.get(m, m)}, {res}ly (n={fit['n']})", fontsize=9)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16 if i == nr - 1 else -0.06),
                      frameon=False, fontsize=7, ncol=1)
            if j == 0:
                ax.set_ylabel(f"disasters per {unit}")
        for ax in axes[-1]:
            ax.set_xlabel("year")
    return fig
