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
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0}
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


def fig_model_compare(cfg: dict, res: str = "month", L: int = 5):
    """Rows = observation models on the same series: Poisson counts, and the Bernoulli
    indicator "at least one disaster in the bin". Columns = exact posterior (Poisson-Gamma,
    Beta-Bernoulli), exact CAVI, gIBSS. Each model row has a main panel (data, fitted
    rate or probability, SuSiE effects with log BF > 0.5) over a strip with the
    per-boundary changepoint probability: the exact boundary marginal, or the SuSiE PIP.
    Each effect's 95% credible set is one row of ticks under the axis (one tick per member
    column); the contiguous block carrying most of its alpha is shaded, with that alpha mass
    printed above (strong shading = declared, faint = below the threshold)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    min_bf = cfg["fit"]["min_log_bf"]
    ex = cfg["exact"]
    labels, counts = series(res)
    step = labels[1] - labels[0]
    bnd = labels[1:] - step / 2  # boundary between bin j and bin j+1
    per_year = 12 if res == "month" else 1
    unit = "month" if res == "month" else "year"
    models = [
        {"name": "Poisson", "y": counts, "link": np.exp,
         "exact": load_exact(res, ex["shape_default"], ex["p_default"]),
         "exact_name": "Poisson-Gamma", "exact_scale": 1.0 / per_year,
         "fits": {"CAVI": f"cf_cavi_L{L}", "gIBSS": f"poisson_L{L}"},
         "ylabel": f"Poisson\ndisasters per {unit}"},
        {"name": "Bernoulli", "y": (counts > 0).astype(float),
         "link": lambda e: 1.0 / (1.0 + np.exp(-e)),
         "exact": dict(np.load(RESULTS / res / "exact"
                               / f"exact_bernoulli_a{ex['shape_default']}_p{ex['p_default']}.npz")),
         "exact_name": "Beta-Bernoulli", "exact_scale": 1.0,
         "fits": {"CAVI": f"cf_cavi_bernoulli_L{L}", "gIBSS": f"bernoulli_L{L}"},
         "ylabel": f"Bernoulli\nany disaster in {unit}"},
    ]
    cols = ["exact", "CAVI", "gIBSS"]
    fig = plt.figure(figsize=(6.5, 6.6))
    gs = fig.add_gridspec(5, 3, height_ratios=[3, 1, 0.45, 3, 1], hspace=0.06, wspace=0.07,
                          bottom=0.15, top=0.96)
    first = None
    n_effects = 0
    for i, mod in enumerate(models):
        y = mod["y"]
        ymax = max(y.max(), 1.0)
        main_row, strip_row = 3 * i, 3 * i + 1
        strip_axes = []
        row_first = None
        for j, col in enumerate(cols):
            ax = fig.add_subplot(gs[main_row, j], sharex=first, sharey=row_first)
            first = first or ax
            row_first = row_first or ax
            sx = fig.add_subplot(gs[strip_row, j], sharex=first)
            strip_axes.append(sx)
            # one line per nonzero bin: monthly bars are sub-pixel wide and alias away
            nz = y > 0
            ax.vlines(labels[nz], 0, y[nz], color="#c4c4c4", lw=0.5)
            n_shown = 0
            if col == "exact":
                e = mod["exact"]
                ax.step(labels, e["mean_profile"] * mod["exact_scale"], where="mid",
                        color=EXACT, lw=1.3)
                kp = e["k_prob"]
                kmean = float(np.sum(np.arange(len(kp)) * kp))
                ax.text(0.99, 0.84, f"{mod['exact_name']}\nE[k] = {kmean:.1f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, bbox=LABEL_BOX)
                sx.vlines(bnd, 0, e["boundary_prob"], color=EXACT, lw=0.6)
            else:
                fit = load_fit(res, mod["fits"][col])
                ax.step(labels, mod["link"](fit["eta"]), where="mid", color=EXACT, lw=1.3)
                shown = [l for l in range(fit["L"]) if fit["ser_log_bf"][l] >= 0.5]
                n_shown = len(shown)
                n_effects = max(n_effects, n_shown)
                for r, l in enumerate(shown):
                    a = fit["alpha"][l]
                    cs = fit["cs"][l]
                    lbf = float(fit["ser_log_bf"][l])
                    declared = lbf >= min_bf
                    lo, hi, mass = _cs_blocks(cs, a)[0]
                    ax.axvspan(bnd[lo] - step / 2, bnd[hi] + step / 2, color=COMP[r],
                               alpha=0.25 if declared else 0.10, lw=0)
                    y0 = -ymax * (0.07 + 0.08 * r)
                    ax.vlines(bnd[list(cs)], y0 - ymax * 0.03, y0 + ymax * 0.03,
                              color=COMP[r], lw=0.4)
                    ax.text(bnd[lo] - step / 2, ymax * 1.01, f"{mass:.0%}", color=COMP[r],
                            fontsize=7, va="bottom", ha="left", bbox=LABEL_BOX)
                    ax.text(0.99, 0.84 - 0.09 * r,
                            f"log BF {lbf:.1f}" + ("" if declared else " (not declared)"),
                            transform=ax.transAxes, ha="right", va="top", fontsize=7,
                            color=COMP[r], bbox=LABEL_BOX)
                sx.vlines(bnd, 0, fit["pip"], color=EXACT, lw=0.6)
            ax.axhline(0, color="#777777", lw=0.5)
            ax.set_ylim(-ymax * (0.07 + 0.08 * max(n_shown, 2)), ymax * 1.14)
            ax.tick_params(labelbottom=False)
            ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= ymax])
            if i == 0:
                ax.set_title(col, fontsize=10)
            if j == 0:
                ax.set_ylabel(mod["ylabel"])
                sx.set_ylabel("P(change)", fontsize=7.5)
            else:
                ax.tick_params(labelleft=False)
                sx.tick_params(labelleft=False)
            if i == 0:
                sx.tick_params(labelbottom=False)
            else:
                sx.set_xlabel("year")
        top = max(s.get_ylim()[1] for s in strip_axes)
        for s in strip_axes:
            s.set_ylim(0, top)
            s.tick_params(labelsize=7)
    handles = [Line2D([], [], color=EXACT, lw=1.3, label="posterior mean rate / probability")]
    for r in range(n_effects):
        handles.append(Patch(color=COMP[r], alpha=0.5,
                             label=f"effect {r + 1}: CS members (ticks), main block (shaded)"))
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    return fig
