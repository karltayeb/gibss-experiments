"""Figure: every arm against a reference arm, one point per simulation scenario.

The 025 `vs_cavi_figure` layout with a configurable reference (default linear_int, the GTEx /
jaxQTL-linear pipeline). For each L (5, then 1) two column groups: the reference's own coverage,
power, mean declared-CS size and declared CSs per fit, then each arm minus the reference in the
same scenario (size: arm over reference, log axis). One point per scenario = one cell (n, mu0,
phi, |beta|, k); every arm sees the same simulated replicates. Tinted band and tick label =
arm, point colour = |beta| (left to right within a band), marker = k_causal. Declared = 95% CS
of a component with SER log BF >= 2. Null cells (k = 0) are excluded (no coverage / power).

    uv run python analysis/vs_reference.py [COLLECTION] [REF]
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
from eqtl.evaluate import SCENARIO  # noqa: E402

MIN_LOG_BF = 2.0
ARM_ORDER = ["linear_int", "linear_log", "nb_true", "nb_est", "nb_score_rss", "nb_score_rss_true", "nb_mle_rss", "nb_cavi", "nb_cavi_true", "nb_polya_gamma", "nb_poisson_gamma", "poisson", "poisson_cavi"]
LABEL = {"linear_int": "linear-INT", "linear_log": "linear-log", "nb_true": "NB gIBSS (true r)",
         "nb_est": "NB gIBSS", "nb_score_rss": "NB score RSS", "nb_score_rss_true": "NB score RSS (true r)", "nb_mle_rss": "NB MLE RSS",
         "nb_cavi": "NB CAVI-Q2", "nb_cavi_true": "NB CAVI-Q2 (true r)",
         "nb_polya_gamma": "NB Polya-Gamma", "nb_poisson_gamma": "NB Poisson-Gamma",
         "poisson": "Poisson gIBSS", "poisson_cavi": "Poisson CAVI-Q2"}
COLOR = {"linear_int": "#0072B2", "linear_log": "#56B4E9", "nb_true": "#D55E00", "nb_est": "#E69F00",
         "nb_score_rss": "#009E73", "nb_score_rss_true": "#CC79A7", "nb_mle_rss": "#117733", "nb_cavi": "#000000", "nb_cavi_true": "#7F7F7F",
         "nb_polya_gamma": "#882255", "nb_poisson_gamma": "#44AA99",
         "poisson": "#117733", "poisson_cavi": "#332288"}
K_MARKER = {1: "o", 2: "s", 3: "^"}
SCEN = ["L", *SCENARIO]
_ABS = [("coverage", "coverage", False), ("power", "power", False), ("size", "mean CS size", True), ("n_cs", "declared CSs / fit", False)]
_REL = [("d_coverage", "− ref", False), ("d_power", "− ref", False), ("r_size", "÷ ref", True), ("d_n_cs", "− ref", False)]


def per_rep(cs: pl.DataFrame, fit: pl.DataFrame) -> pl.DataFrame:
    """One row per (cell, L, arm, rep): declared sets, covering sets, size sum, causal hits."""
    decl = pl.col("ser_log_bf") >= MIN_LOG_BF
    d = (cs.group_by("cell", "L", "arm", "rep")
           .agg(decl.sum().alias("n_decl"), (decl & pl.col("covered")).sum().alias("n_cover"),
                pl.when(decl).then(pl.col("cs_size")).otherwise(0).sum().alias("size_sum"),
                pl.when(decl).then(pl.col("n_causal_in_cs")).otherwise(0).sum().alias("hits")))
    coords = fit.select("cell", "L", "arm", "rep", *SCENARIO).unique()
    out = coords.join(d, on=["cell", "L", "arm", "rep"], how="left").fill_null(0)
    return out.with_columns(pl.min_horizontal(pl.col("hits"), pl.col("k_causal")).alias("hits"))


def scenario_frame(cs: pl.DataFrame, fit: pl.DataFrame, ref: str = "linear_int") -> pl.DataFrame:
    """One row per (scenario, arm != ref): the arm's coverage / power / mean size / declared
    sets per fit, the reference's in the same scenario (`*_ref`), and the differences."""
    reps = per_rep(cs, fit).filter(pl.col("k_causal") > 0)
    df = (reps.group_by(SCEN + ["arm"])
          .agg((pl.col("n_cover").sum() / pl.col("n_decl").sum()).alias("coverage"),
               (pl.col("hits").sum() / pl.col("k_causal").sum()).alias("power"),
               (pl.col("size_sum").sum() / pl.col("n_decl").sum()).alias("size"),
               pl.col("n_decl").mean().alias("n_cs"))
          .with_columns(pl.col(c).fill_nan(None) for c in ("coverage", "size")))
    r = df.filter(pl.col("arm") == ref).select(SCEN + [pl.col(c).alias(f"{c}_ref") for c, _, _ in _ABS])
    return (df.filter(pl.col("arm") != ref).join(r, on=SCEN, how="inner", nulls_equal=True)
            .with_columns((pl.col("coverage") - pl.col("coverage_ref")).alias("d_coverage"),
                          (pl.col("power") - pl.col("power_ref")).alias("d_power"),
                          (pl.col("size") / pl.col("size_ref")).alias("r_size"),
                          (pl.col("n_cs") - pl.col("n_cs_ref")).alias("d_n_cs")))


def _log_axis(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def vs_reference_figure(sf: pl.DataFrame, ref: str = "linear_int", seed: int = 0, Ls=("5", "1")):
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    rng = np.random.default_rng(seed)
    sf = sf.with_columns(pl.col("L").cast(pl.Utf8))
    ms_by_L = {L: [m for m in ARM_ORDER if m in set(sf.filter(pl.col("L") == L)["arm"].to_list())] for L in Ls}
    Ls = [L for L in Ls if ms_by_L[L]]
    widths = []
    for L in Ls:
        widths += [1.25, len(ms_by_L[L]) + 0.6]
    fig, axes = plt.subplots(len(_REL), 2 * len(Ls), figsize=(6.5, 1.7 * len(_REL) + 1.5), squeeze=False,
                             layout="constrained", gridspec_kw={"width_ratios": widths})
    for i in range(len(_REL)):
        for c in range(2, 2 * len(Ls)):
            axes[i, c].sharey(axes[i, c % 2])
    betas = sorted(sf["beta_abs"].unique().to_list())
    offs = dict(zip(betas, np.linspace(-0.32, 0.32, len(betas))))
    half = 0.32 / max(len(betas) - 1, 1)
    b_color = dict(zip(betas, plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(betas)))))
    ks = sorted(sf["k_causal"].unique().to_list())

    def band(ax, x, color):
        ax.axvspan(x - 0.45, x + 0.45, color=to_rgba(color, 0.10), lw=0, zorder=0)

    def strip(ax, x, df, c):
        for b in betas:
            g = df.filter(pl.col("beta_abs") == b)
            for k in ks:
                y = g.filter(pl.col("k_causal") == k)[c].drop_nulls().to_numpy().astype(float)
                y = y[np.isfinite(y)]
                ax.scatter(x + offs[b] + rng.uniform(-0.45, 0.45, y.size) * half, y, s=9, marker=K_MARKER.get(k, "o"),
                           facecolor="none", edgecolor=b_color[b], linewidth=0.6, alpha=0.85, zorder=2)

    def color_ticks(ax, colors):
        for t, col in zip(ax.get_xticklabels(), colors):
            t.set_color(col)

    refrows = sf.unique(subset=SCEN, keep="first")
    last = len(_REL) - 1
    for bi, L in enumerate(Ls):
        ca, cr = 2 * bi, 2 * bi + 1
        dref, d_L, ms = refrows.filter(pl.col("L") == L), sf.filter(pl.col("L") == L), ms_by_L[L]
        for i, ((acol, alabel, alog), (rcol, rlabel, ratio)) in enumerate(zip(_ABS, _REL)):
            ax = axes[i, ca]
            band(ax, 0, COLOR[ref]); strip(ax, 0, dref, f"{acol}_ref")
            if alog and bi == 0:
                _log_axis(ax)
            if acol == "coverage":
                ax.axhline(0.95, color="0.5", lw=0.7, ls="--", zorder=1)
            ax.set_xlim(-0.5, 0.5); ax.set_xticks([0])
            ax.set_xticklabels([LABEL[ref]] if i == last else [], rotation=60, ha="right", fontsize=6.5)
            color_ticks(ax, [COLOR[ref]])
            if i == 0:
                ax.set_title(LABEL[ref], fontsize=7.5, color=COLOR[ref])
            ax = axes[i, cr]
            ax.axhline(1.0 if ratio else 0.0, color="0.5", lw=0.7, ls="--", zorder=1)
            for k, m in enumerate(ms):
                band(ax, k, COLOR[m]); strip(ax, k, d_L.filter(pl.col("arm") == m), rcol)
            if ratio and bi == 0:
                _log_axis(ax)
            ax.set_xlim(-0.5, len(ms) - 0.5); ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([LABEL[m] for m in ms] if i == last else [], rotation=60, ha="right", fontsize=6.5)
            color_ticks(ax, [COLOR[m] for m in ms])
            if i == 0:
                ax.set_title(f"L = {L}: arm vs {LABEL[ref]}", fontsize=7.5)
            for a_, lab in ((axes[i, ca], alabel), (axes[i, cr], rlabel)):
                a_.grid(True, axis="y", alpha=0.25); a_.tick_params(labelsize=6.5, labelleft=True)
                if bi == 0:
                    a_.set_ylabel(lab, fontsize=7.5)
    h = [Line2D([], [], color="0.3", marker=K_MARKER.get(k, "o"), mfc="none", ls="", ms=5, label=f"k = {k}") for k in ks]
    h += [Line2D([], [], color=b_color[b], marker="o", ls="", ms=5, label=f"|β| = {b:g}") for b in betas]
    fig.legend(handles=h, loc="outside lower center", ncol=min(len(h), 6), fontsize=6.5, frameon=False)
    return fig


def main(collection: str = "pilot", ref: str = "linear_int") -> None:
    root = os.path.join(os.path.dirname(_HERE), "results", "collections", collection)
    cs = pl.read_parquet(os.path.join(root, "cs.parquet")); fit = pl.read_parquet(os.path.join(root, "fit.parquet"))
    sf = scenario_frame(cs, fit, ref)
    out = os.path.join(_HERE, "figures"); os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"vs_{ref}_{collection}")
    sf.write_parquet(stem + ".parquet")
    vs_reference_figure(sf, ref).savefig(stem + ".png", dpi=200, bbox_inches="tight")
    print("wrote", stem + ".png")


if __name__ == "__main__":
    main(*sys.argv[1:])
