"""Lambda-example figures and tables, drawn from saved results: GC profile with the
SuSiE step fit, CS hulls, exact boundary and HMM switch probabilities, gene blocks; CSs by
bin width. `fig_*` return Figures without saving them (shared by script and notebook)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lam import PROC
from stepbasis import cs_interval

RESULTS = ROOT / "results/lambda"
FIG = RESULTS / "figures"
COMP = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
EXACT = "#222222"
HMM = "#D55E00"
DATA = "#9a9a9a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})


def load_fit(w: int, L: int) -> dict:
    with open(RESULTS / f"w{w}" / "fits" / f"binomial_L{L}.pkl", "rb") as fh:
        return pickle.load(fh)


def bins(w: int):
    d = pl.read_csv(PROC / f"bins_{w}.csv")
    return d["start"].to_numpy().astype(float), d["k"].to_numpy().astype(float), d["m"].to_numpy().astype(float)


def declared(fit: dict, min_bf: float):
    return [l for l in range(fit["L"]) if fit["ser_log_bf"][l] >= min_bf]


def fig_main(cfg: dict):
    """Primary width, L = max. Returns (fig, annotated declared-CS table)."""
    w = cfg["primary_width"]
    L = cfg["fit"]["L"][-1]
    min_bf = cfg["fit"]["min_log_bf"]
    start, k, m = bins(w)
    mid = start + m / 2
    bnd = start[1:]  # boundary between bin j and j+1 is the start of bin j+1
    fit = load_fit(w, L)
    ex = np.load(RESULTS / f"w{w}" / "exact" / f"exact_p{cfg['exact']['p_default']}.npz")
    hm = np.load(RESULTS / f"w{w}" / "hmm.npz")
    dec = declared(fit, min_bf)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.2, 1.2], "hspace": 0.08})
    a0, a1, a2 = axes
    a0.plot(mid / 1000, k / m, color=DATA, lw=0.7, label=f"GC fraction per {w} bp bin")
    a0.step(mid / 1000, 1 / (1 + np.exp(-fit["eta"])), where="mid", color=COMP[0], lw=1.5,
            label=f"SuSiE fitted GC (L={L}, {len(dec)} declared CSs)")
    a0.step(mid / 1000, ex["mean_profile"], where="mid", color=EXACT, lw=1.0, ls="--",
            label=f"exact posterior mean (p={cfg['exact']['p_default']} per bin)")
    for i, l in enumerate(dec):
        lo, hi, _ = cs_interval(fit["cs"][l])
        a0.axvspan(bnd[lo] / 1000, bnd[hi] / 1000 + w / 1000, color=COMP[i % len(COMP)], alpha=0.18, lw=0)
    a0.set_ylabel("GC fraction")
    a0.set_ylim(0.15, 0.86)
    a0.legend(loc="lower right", frameon=False, fontsize=7.5, ncol=1)
    a0.set_title("Lambda phage GC content: binomial SuSiE on the step basis", fontsize=10)
    # gene blocks as labelled spans at the top of the first panel
    for i, b in enumerate(cfg["blocks"]):
        a0.axvspan(b["start"] / 1000, b["end"] / 1000, ymin=0.96, ymax=1.0,
                   color=["#444444", "#999999"][i % 2], alpha=0.8, lw=0)
        a0.text((b["start"] + b["end"]) / 2000, 0.80, b["name"].split(" ")[0], ha="center",
                va="top", fontsize=6, rotation=90 if (b["end"] - b["start"]) < 3000 else 0)

    for i, l in enumerate(dec):
        a1.vlines(bnd / 1000, 0, fit["alpha"][l], color=COMP[i % len(COMP)], lw=0.9)
    a1.set_ylabel("SuSiE alpha\n(declared CSs)")
    a1.set_ylim(0, 1)
    a2.plot(bnd / 1000, ex["boundary_prob"], color=EXACT, lw=0.9, label="exact P(changepoint)")
    a2.plot(bnd / 1000, hm["switch_prob"], color=HMM, lw=0.9, ls=":", label="2-state HMM P(switch)")
    a2.set_ylabel("probability")
    a2.set_ylim(0, 1)
    a2.set_xlabel("genome position (kb)")
    a2.legend(loc="upper left", bbox_to_anchor=(0.13, 1.0), frameon=False, fontsize=7.5)
    for b in cfg["blocks"][1:]:
        for ax in axes:
            ax.axvline(b["start"] / 1000, color="#bbbbbb", lw=0.5, ls="-", zorder=0)

    # CS table in bp with nearest block boundary and nearest HMM/exact peaks
    block_bounds = sorted({b["start"] for b in cfg["blocks"][1:]} | {b["end"] for b in cfg["blocks"][:-1]})
    cds = pl.read_csv(PROC / "cds.csv")
    cds_bounds = np.sort(np.concatenate([cds["start"].to_numpy(), cds["end"].to_numpy()]))
    rows = []
    for i, l in enumerate(dec):
        c = fit["cs"][l]
        lo, hi, width = cs_interval(c)
        top = int(np.argmax(fit["alpha"][l]))
        top_bp = bnd[top]
        cs_lo_bp, cs_hi_bp = bnd[lo], bnd[hi] + w
        nb = min(block_bounds, key=lambda b: abs(b - top_bp))
        inside = any(cs_lo_bp - w <= b <= cs_hi_bp + w for b in block_bounds)
        ncds = cds_bounds[np.argmin(np.abs(cds_bounds - top_bp))]
        exact_mass = float(ex["boundary_prob"][lo:hi + 1].sum())
        hmm_mass = float(hm["switch_prob"][lo:hi + 1].sum())
        rows.append({"component": l + 1, "ser_log_bf": float(fit["ser_log_bf"][l]),
                     "cs_size": len(c), "cs_hull_width_bins": width,
                     "cs_start_bp": int(cs_lo_bp), "cs_end_bp": int(cs_hi_bp), "top_bp": int(top_bp),
                     "max_pip": float(fit["alpha"][l][top]),
                     "log_odds_jump": float(fit["mu"][l, top]),
                     "gc_before": float(1 / (1 + np.exp(-fit["eta"][top]))),
                     "gc_after": float(1 / (1 + np.exp(-fit["eta"][top + 1]))),
                     "nearest_block_boundary_bp": int(nb), "dist_to_block_boundary_bp": int(abs(nb - top_bp)),
                     "block_boundary_in_cs": inside,
                     "dist_to_nearest_cds_end_bp": int(abs(ncds - top_bp)),
                     "exact_mass_in_hull": exact_mass, "hmm_switch_mass_in_hull": hmm_mass})
    tbl = pl.DataFrame(rows).sort("top_bp")
    return fig, tbl


def table_scale(cfg: dict) -> pl.DataFrame:
    min_bf = cfg["fit"]["min_log_bf"]
    rows = []
    eg = pl.read_csv(RESULTS / "exact_grid_summary.csv")
    hs = pl.read_csv(RESULTS / "hmm_summary.csv")
    for w in cfg["bin_widths"]:
        for L in cfg["fit"]["L"]:
            fit = load_fit(w, L)
            dec = declared(fit, min_bf)
            widths = [cs_interval(fit["cs"][l])[2] * w for l in dec]
            e = eg.filter((pl.col("width") == w) & (pl.col("p_per_100bp") == cfg["exact"]["p_default"]))
            h = hs.filter(pl.col("width") == w)
            rows.append({"width": w, "n_bins": fit["n"], "L": L, "declared_cs": len(dec),
                         "median_cs_width_bp": float(np.median(widths)) if widths else np.nan,
                         "max_cs_width_bp": float(np.max(widths)) if widths else np.nan,
                         "exact_k_mean": float(e["k_mean"][0]), "exact_k_q05": int(e["k_q05"][0]),
                         "exact_k_q95": int(e["k_q95"][0]),
                         "hmm_expected_switches": float(h["expected_switches"][0]),
                         "seconds": fit["seconds"]})
    return pl.DataFrame(rows)


def fig_scale(cfg: dict):
    """Declared CS hulls at every width (L = max), stacked, with block boundaries."""
    min_bf = cfg["fit"]["min_log_bf"]
    L = cfg["fit"]["L"][-1]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for i, w in enumerate(cfg["bin_widths"]):
        start, _k, _m = bins(w)
        bnd = start[1:]
        fit = load_fit(w, L)
        for l in declared(fit, min_bf):
            lo, hi, _ = cs_interval(fit["cs"][l])
            top = int(np.argmax(fit["alpha"][l]))
            ax.plot([bnd[lo] / 1000, (bnd[hi] + w) / 1000], [i, i], color=COMP[0], lw=4, alpha=0.5,
                    solid_capstyle="butt")
            ax.plot(bnd[top] / 1000, i, "|", color=EXACT, ms=8, mew=1.2)
    for b in cfg["blocks"][1:]:
        ax.axvline(b["start"] / 1000, color="#bbbbbb", lw=0.6, zorder=0)
    ax.set_yticks(range(len(cfg["bin_widths"])))
    ax.set_yticklabels([f"{w} bp" for w in cfg["bin_widths"]])
    ax.set_xlabel("genome position (kb)")
    ax.set_title(f"Declared CS hulls (L={L}) by bin width", fontsize=10)
    return fig
