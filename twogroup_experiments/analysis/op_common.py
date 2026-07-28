"""Shared data + faceting layer for the 011 GO:BP operating-points notebook.

The 011 supercollection is 24 difficulty-matched cells (all on the iso-detectability
E[LRT]=25 curve; see analysis/detectability_level_curves.qmd). Each cell crosses:

  scenario in {loc, scale}   x   set size m in {50,100,200,400}   x   regime in {enrich,balanced,signal}

`enrich` sources detectability from the enrichment log-odds b (max b=4, weak per-gene
signal); `signal` from strong per-gene signal (max mu=4 / sigma=6, weak b); `balanced`
is the arc-length midpoint. This module maps the on-disk collection/batch names to that
(scenario, m, regime) grid, loads the `pip` and `cs` reductions off disk (bypassing the
live loader - same pattern as c2_discretization_example.qmd), and provides a 4x3
faceting helper (rows = size, cols = regime) that every figure in the notebook reuses.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

import viz_utils

# ---------------------------------------------------------------------------
# Methods (the 011 op_methods anchor, in display order: oracles, then models,
# then the thresholded cox sweep).
# ---------------------------------------------------------------------------
OP_METHODS = [
    "twogroup_oracle__L=1",
    "twogroup__L=1",
    "twogroup_loc_fam__L=1",
    "twogroup_scale_fam__L=1",
    "linear_fixed__L=1",
    "cox__threshold=1.00__L=1",
    "cox__threshold=2.00__L=1",
    "cox__threshold=3.00__L=1",
    "cox__threshold=4.00__L=1",
    "cox_reversed__L=1",
]

SIZES = [50, 100, 200, 400]           # facet rows
REGIMES = ["enrich", "balanced", "signal"]   # facet cols
SIZE_RANGE = {50: "40_60", 100: "80_120", 200: "160_240", 400: "320_480"}

# alias -> (b, signal_value); scenario/m/regime are parsed from the alias.
# Values transcribed from experiments/011_gobp_operating_points.yaml.
_CELL_PARAMS = {
    "loc_m50_enrich": (4.00, 0.96), "loc_m50_balanced": (2.15, 1.93), "loc_m50_signal": (1.62, 4.00),
    "loc_m100_enrich": (4.00, 0.66), "loc_m100_balanced": (1.80, 1.62), "loc_m100_signal": (1.21, 4.00),
    "loc_m200_enrich": (4.00, 0.46), "loc_m200_balanced": (1.52, 1.37), "loc_m200_signal": (0.91, 4.00),
    "loc_m400_enrich": (4.00, 0.32), "loc_m400_balanced": (1.29, 1.18), "loc_m400_signal": (0.68, 4.00),
    "scale_m50_enrich": (4.00, 1.70), "scale_m50_balanced": (2.36, 3.12), "scale_m50_signal": (1.94, 6.00),
    "scale_m100_enrich": (4.00, 1.22), "scale_m100_balanced": (1.91, 2.55), "scale_m100_signal": (1.45, 6.00),
    "scale_m200_enrich": (4.00, 0.94), "scale_m200_balanced": (1.57, 2.14), "scale_m200_signal": (1.09, 6.00),
    "scale_m400_enrich": (4.00, 0.75), "scale_m400_balanced": (1.31, 1.83), "scale_m400_signal": (0.81, 6.00),
}


def _batch_prefix(scenario: str, m: int, b: float, sig: float) -> str:
    """Reconstruct the on-disk batch-name prefix for a cell."""
    return f"gobp_10_500__op_b{b:.2f}_s{SIZE_RANGE[m]}__op_{scenario}_{sig:.2f}"


def cells() -> list[dict]:
    """The 24 cells, each with alias, scenario, m, regime, params, and batch prefix."""
    out = []
    for alias, (b, sig) in _CELL_PARAMS.items():
        scenario, mseg, regime = alias.split("_")  # loc/scale, m50, enrich
        m = int(mseg[1:])
        out.append({
            "alias": alias, "scenario": scenario, "m": m, "regime": regime,
            "b": b, "signal": sig, "batch_prefix": _batch_prefix(scenario, m, b, sig),
        })
    return out


def cell_alias(scenario: str, m: int, regime: str) -> str:
    return f"{scenario}_m{m}_{regime}"


B0 = -2.0  # fixed background membership log-odds across all 011 cells


def design_table(scenario: str) -> pl.DataFrame:
    """The 12 design points for one scenario, with derived membership quantities.

    p* = sigmoid(b0 + b) is the causal-set membership rate; E[active] = m * p* the
    expected number of planted active genes. All rows sit on E[LRT] ~ 25 by design.
    """
    sig_col = "mu" if scenario == "loc" else "sigma"
    rows = []
    for c in cells():
        if c["scenario"] != scenario:
            continue
        p_star = 1.0 / (1.0 + np.exp(-(B0 + c["b"])))
        rows.append({
            "m": c["m"], "regime": c["regime"], "b": c["b"], sig_col: c["signal"],
            "p*": round(p_star, 3), "E[active]": round(c["m"] * p_star, 1),
        })
    regime_rank = {r: i for i, r in enumerate(REGIMES)}
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("regime").replace_strict(regime_rank).alias("_r"))
        .sort(["m", "_r"]).drop("_r")
    )


# ---------------------------------------------------------------------------
# Reduction loading (read reductions/*.parquet straight off disk, tag with the
# cell alias). Mirrors c2_discretization_example.qmd's load_collection.
# ---------------------------------------------------------------------------
def _name_by_hash(results_root: str) -> dict[str, str]:
    manifest = json.load(open(f"{results_root}/manifest_cache.json"))
    return {h: b["name"] for h, b in manifest["batches"].items()}


def method_metadata(results_root: str = "results") -> pl.DataFrame:
    from experiments import loader as L
    manifest = json.load(open(f"{results_root}/manifest_cache.json"))
    specs = {m["name"]: m for m in manifest["methods"].values() if m["name"] in OP_METHODS}
    missing = set(OP_METHODS) - set(specs)
    if missing:
        raise AssertionError(f"methods missing from manifest: {missing}")
    return L.method_metadata(specs)


def null_prefix(cell: dict) -> str:
    """On-disk batch-name prefix of a cell's paired null (null_b0, same signal, b=0)."""
    return f"gobp_10_500__null_b0__op_{cell['scenario']}_{cell['signal']:.2f}"


def _load_prefix(kind, results_root, name_by_hash, prefix, alias, is_null):
    frames = []
    for bh, name in name_by_hash.items():
        if not (name == prefix or name.startswith(prefix + "__batch")):
            continue
        for path in glob.glob(f"{results_root}/by_batch/{bh}/fits/*/reductions/{kind}.parquet"):
            df = pl.read_parquet(path).filter(pl.col("method").is_in(OP_METHODS))
            if df.height:
                frames.append(df.with_columns(
                    pl.lit(alias).alias("collection_name"),
                    pl.lit(is_null).alias("is_null"),
                ))
    return frames


def load_reduction(kind: str, results_root: str = "results", *, include_null: bool = False) -> pl.DataFrame:
    """Concat the `kind` reduction (pip|cs) for every cell, tagged collection_name=alias
    and is_null. With include_null, also loads each cell's paired null_b0 batch (matched
    by signal); shared-signal nulls are tagged under each cell that references them.

    Missing reduction files (e.g. cs not yet materialized) are skipped per cell.
    """
    name_by_hash = _name_by_hash(results_root)
    frames = []
    for cell in cells():
        frames += _load_prefix(kind, results_root, name_by_hash, cell["batch_prefix"], cell["alias"], False)
        if include_null:
            frames += _load_prefix(kind, results_root, name_by_hash, null_prefix(cell), cell["alias"], True)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Method colour / ordering (reuse the twogroup Okabe-Ito palette).
# ---------------------------------------------------------------------------
def method_color(method: str) -> str:
    return viz_utils.method_color(method)


# The 4 cox methods share the green family colour; encode the censoring threshold
# as line style so they are distinguishable (same convention as c2_discretization).
THRESH_LS = {None: "-", 1.0: "-", 2.0: "--", 3.0: "-.", 4.0: (0, (1, 1))}


def linestyle_by_method(method_meta: pl.DataFrame) -> dict[str, object]:
    return {r["method"]: THRESH_LS.get(r["threshold"], "-") for r in method_meta.iter_rows(named=True)}


def method_order(method_meta: pl.DataFrame) -> list[str]:
    return (
        method_meta.sort(["is_oracle", "is_thresholded", "method"], descending=[True, False, False])
        .get_column("method").to_list()
    )


# ---------------------------------------------------------------------------
# 4x3 faceting: rows = size, cols = regime, one scenario per figure.
# ---------------------------------------------------------------------------
def facet_grid(scenario: str, cell_plot_fn, *, figsize_scale=(3.3, 3.0), sharex=True,
               sharey=True, suptitle=None):
    """Build a 4x3 grid for one scenario and call cell_plot_fn(ax, alias, m, regime)
    on each panel. Returns (fig, axes). Column titles = regime, row labels = size m.
    """
    import matplotlib.pyplot as plt
    nrow, ncol = len(SIZES), len(REGIMES)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(figsize_scale[0] * ncol, figsize_scale[1] * nrow),
        sharex=sharex, sharey=sharey, squeeze=False,
    )
    for i, m in enumerate(SIZES):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            cell_plot_fn(ax, cell_alias(scenario, m, regime), m, regime)
            if i == 0:
                ax.set_title(regime, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"m = {m}", fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    return fig, axes


def shared_method_legend(fig, method_meta, *, loc="center left", bbox=(0.9, 0.5)):
    """One figure-level legend keyed by method colour, in method_order."""
    import matplotlib.pyplot as plt
    order = method_order(method_meta)
    disp = {r["method"]: r["method_display"] for r in method_meta.iter_rows(named=True)}
    ls_by = linestyle_by_method(method_meta)
    handles = [
        plt.Line2D([], [], color=method_color(m), lw=2.0, ls=ls_by[m], label=disp[m])
        for m in order
    ]
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# Figure builders. Each takes the (already loaded) reduction frame + method_meta
# and returns a 4x3 fig for one scenario. Two are pip-driven (power-FDP,
# calibration), two are cs-driven (coverage-size operating curve, localization
# ROC). All overlay the 10 methods in the shared palette.
# ---------------------------------------------------------------------------
def _finish(fig, method_meta, right=0.86):
    fig.tight_layout(rect=[0, 0, right, 1])
    shared_method_legend(fig, method_meta, bbox=(right + 0.01, 0.5))
    return fig


def figure_power_fdp(scenario, pip, method_meta, *, max_fdp=0.5):
    """Power vs FDP per cell (PIP sweep). Rides top-left = efficient ordering of PIPs."""
    pf = viz_utils.expand_power_fdp_from_compact(pip, method_meta, selected_methods=set(OP_METHODS))
    order = method_order(method_meta)
    ls_by = linestyle_by_method(method_meta)

    def cell(ax, alias, m, regime):
        d = pf.filter(pl.col("simulation_name") == alias)
        for meth in order:
            # trace in PIP-threshold order (the sweep parameter); sorting by fdp
            # scrambles the parametric curve into a sawtooth.
            md = d.filter(pl.col("method") == meth).sort("pip_threshold", descending=True)
            if md.height:
                ax.plot(md["fdp"], md["power"], color=method_color(meth), lw=1.4, ls=ls_by[meth])
        ax.set_xlim(0, max_fdp)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        if m == SIZES[-1]:
            ax.set_xlabel("FDP")

    fig, _ = facet_grid(scenario, cell)
    return _finish(fig, method_meta)


def figure_pip_calibration(scenario, pip, method_meta):
    """Empirical causal frequency vs PIP bin midpoint per cell; diagonal = perfect."""
    cal = viz_utils.expand_pip_calibration_from_compact(pip, method_meta)
    order = method_order(method_meta)
    ls_by = linestyle_by_method(method_meta)

    def cell(ax, alias, m, regime):
        d = cal.filter(pl.col("simulation_name") == alias)
        for meth in order:
            md = d.filter(pl.col("method") == meth).sort("pip_mid")
            md = md.filter(pl.col("empirical_rate").is_not_null())
            if md.height:
                ax.plot(md["pip_mid"], md["empirical_rate"], color=method_color(meth),
                        lw=1.2, ls=ls_by[meth], marker="o", markersize=2.5)
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        if m == SIZES[-1]:
            ax.set_xlabel("PIP")

    fig, _ = facet_grid(scenario, cell)
    return _finish(fig, method_meta)


# ----- cs-driven: coverage-size operating curve + localization ROC -----
def _cs_beta_grid():
    from utils import CS_BETA_GRID
    return np.asarray(CS_BETA_GRID.tolist())


def _coverage_size_curves(cs, method_meta, *, nominal=0.95):
    """Per (collection, method), a coverage-vs-median-size curve swept over the beta
    grid, plus the (coverage, size) point at the nominal beta.

    coverage(beta) = mean over sims of [causal in beta-CS] = mean(mass_above_causal < beta).
    size(beta)     = MEDIAN CS size at beta (median, not mean: CS size is heavy-tailed
                     over ~5.3k overlapping GO:BP sets, so a few diffuse CSs blow up the mean).
    """
    grid = _cs_beta_grid()
    inom = int(np.argmin(np.abs(grid - nominal)))
    meta = {r["method"]: r for r in method_meta.iter_rows(named=True)}
    rows = []
    key = ["collection_name", "method"]
    for (coll, meth), g in cs.filter(pl.col("l") == 0).group_by(key):
        if meth not in meta:
            continue
        mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
        sizes = np.asarray(g["cs_sizes"].to_list(), dtype=float)   # (n_sims, 100)
        if sizes.ndim != 2 or sizes.shape[0] == 0:
            continue
        cov = np.array([np.mean(mass < b) for b in grid])
        med = np.median(sizes, axis=0)
        rows.append({
            "collection_name": coll, "method": meth,
            "coverage": cov, "size": med,
            "nom_coverage": float(cov[inom]), "nom_size": float(med[inom]),
        })
    return rows, float(grid[inom])


def figure_coverage_size(scenario, cs, method_meta, *, nominal=0.95, log_y=True):
    """Coverage vs median CS size, swept over nominal beta (my recommended operating
    curve). Marked point = the nominal beta-CS; its x-position vs the dashed line at
    `nominal` reads as calibration (on the line = calibrated, left = anti-conservative).
    Lower curve at a given coverage = tighter (more efficient) method.
    """
    curves, nomv = _coverage_size_curves(cs, method_meta, nominal=nominal)
    by_cell = {}
    for r in curves:
        by_cell.setdefault(r["collection_name"], {})[r["method"]] = r
    order = method_order(method_meta)
    ls_by = linestyle_by_method(method_meta)

    def cell(ax, alias, m, regime):
        cc = by_cell.get(alias, {})
        for meth in order:
            r = cc.get(meth)
            if r is None:
                continue
            col = method_color(meth)
            ax.plot(r["coverage"], r["size"], color=col, lw=1.3, ls=ls_by[meth])
            ax.plot(r["nom_coverage"], r["nom_size"], color=col, marker="o",
                    markersize=5, markeredgecolor="k", markeredgewidth=0.4, zorder=5)
        ax.axvline(nomv, color="grey", ls="--", lw=0.8)
        ax.set_xlim(0, 1.02)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        if m == SIZES[-1]:
            ax.set_xlabel("empirical coverage")

    fig, _ = facet_grid(scenario, cell, sharey=True,
                        suptitle=None)
    return _finish(fig, method_meta)


def _roc(scores, labels):
    """Sweep score high->low; return (fpr, tpr, auc). labels in {0,1}."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, float)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-scores)
    ls = labels[order]
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ls) / n_neg])
    ranks = scores.argsort(kind="mergesort").argsort() + 1
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return fpr, tpr, float(auc)


def _covered_at(g, beta_val, grid):
    """Boolean localization label per replicate: causal set inside the beta-CS."""
    mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
    return (mass < beta_val).astype(float)


def figure_logbf_roc(scenario, cs, method_meta, *, nominal=0.95):
    """Null-free 'are the BFs well ordered?' ROC, POOLED over the 12 cells of one
    scenario (single panel). Positive = causal set landed inside the nominal beta-CS
    (correct localization), negative = it did not; score = the SER log Bayes factor.
    AUC = P(a correctly-localized replicate outranks a mis-localized one by log-BF).

    Pooled, not 4x3: at matched detectability the strong methods localize almost
    deterministically, so per-cell the label is ~all-positive and a per-cell ROC is
    degenerate. Pooling easy+hard cells balances the label and answers the global
    'is the BF a trustworthy confidence signal?' question. Null-free analogue of the
    built-in log_bf_roc (which needs the null collections 011 does not have).
    """
    import matplotlib.pyplot as plt
    grid = _cs_beta_grid()
    inom = int(np.argmin(np.abs(grid - nominal)))
    order = method_order(method_meta)
    disp = {r["method"]: r["method_display"] for r in method_meta.iter_rows(named=True)}
    ls_by = linestyle_by_method(method_meta)
    aliases = [cell_alias(scenario, m, r) for m in SIZES for r in REGIMES]
    base = cs.filter((pl.col("l") == 0) & pl.col("collection_name").is_in(aliases))

    # A method that (mis)localizes only a handful of replicates has a near-degenerate
    # label; its AUC would rest on a few points and read as noise. Require at least
    # MIN_CLASS in each class before drawing/annotating a curve.
    MIN_CLASS = 15
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    handles = []
    for meth in order:
        g = base.filter(pl.col("method") == meth)
        if g.height == 0:
            continue
        covered = _covered_at(g, grid[inom], grid)
        rate = float(np.nanmean(covered))
        n_pos, n_neg = int(covered.sum()), int((1 - covered).sum())
        res = _roc(g["ser_log_bf"].to_numpy(), covered)
        if res is None or min(n_pos, n_neg) < MIN_CLASS:
            # too few of one class to order meaningfully -> report the rate, not an AUC
            label = f"{disp[meth]} (loc.rate {rate:.2f}, n/a)"
            handles.append(plt.Line2D([], [], color=method_color(meth), ls=ls_by[meth], label=label))
            continue
        fpr, tpr, auc = res
        ax.plot(fpr, tpr, color=method_color(meth), lw=1.6, ls=ls_by[meth])
        handles.append(plt.Line2D([], [], color=method_color(meth), ls=ls_by[meth],
                                  label=f"{disp[meth]} (AUC {auc:.2f}, loc.rate {rate:.2f})"))
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="FPR (mis-localized replicates)",
           ylabel="TPR (correctly-localized replicates)",
           title=f"{scenario}: logBF vs 95%-CS localization (pooled over 12 cells)")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


# ===========================================================================
# 1x3 aggregated views (cols = regime, pooled over the 4 set sizes) + a 4x3
# faceted ROC + per-method calibration facets. These complement the 4x3 detail
# figures with condensed summaries (requested layout: "I want the 4x3 plots, but
# I ALSO want the 1x3 plots").
# ===========================================================================
REGIME_COLORS = {"enrich": "#0072B2", "balanced": "#009E73", "signal": "#D55E00"}
SIZE_COLORS = {50: "#F0E442", 100: "#56B4E9", 200: "#E69F00", 400: "#CC79A7"}


def facet_1x3(scenario, cell_plot_fn, *, figsize=(5.2, 4.4), sharex=True, sharey=True,
              suptitle=None):
    """One row, 3 panels (cols = regime). cell_plot_fn(ax, regime, aliases) receives
    the 4 size-aliases of that (scenario, regime) to pool over."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(figsize[0] * len(REGIMES), figsize[1]),
                             sharex=sharex, sharey=sharey, squeeze=False)
    for j, regime in enumerate(REGIMES):
        ax = axes[0, j]
        aliases = [cell_alias(scenario, m, regime) for m in SIZES]
        cell_plot_fn(ax, regime, aliases)
        ax.set_title(regime, fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    return fig, axes


def figure_power_fdp_1x3(scenario, pip, method_meta, *, max_fdp=0.5):
    """Power vs FDP, pooled over the 4 set sizes within each regime (1x3)."""
    order = method_order(method_meta)
    ls_by = linestyle_by_method(method_meta)

    def cell(ax, regime, aliases):
        sub = pip.filter(pl.col("collection_name").is_in(aliases))
        pf = viz_utils.expand_power_fdp_from_compact(
            sub, method_meta, selected_methods=set(OP_METHODS), aggregate_across_collections=True)
        for meth in order:
            md = pf.filter(pl.col("method") == meth).sort("pip_threshold", descending=True)
            if md.height:
                ax.plot(md["fdp"], md["power"], color=method_color(meth), lw=1.5, ls=ls_by[meth])
        ax.set_xlim(0, max_fdp); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        ax.set_xlabel("FDP")
    fig, _ = facet_1x3(scenario, cell)
    axes0 = fig.axes[0]; axes0.set_ylabel("power")
    return _finish(fig, method_meta, right=0.88)


# ----- shared cs cores (pooled over whatever rows are passed) -----
def _cov_size_core(g, grid, inom):
    mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
    sizes = np.asarray(g["cs_sizes"].to_list(), dtype=float)
    if sizes.ndim != 2 or sizes.shape[0] == 0:
        return None
    cov = np.array([np.mean(mass < b) for b in grid])
    med = np.median(sizes, axis=0)
    return cov, med, float(cov[inom]), float(med[inom])


def _roc_core(g, grid, inom, *, min_class=15):
    mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
    covered = (mass < grid[inom]).astype(float)
    rate = float(np.nanmean(covered))
    n_pos, n_neg = int(covered.sum()), int((1 - covered).sum())
    res = _roc(g["ser_log_bf"].to_numpy(), covered)
    if res is None or min(n_pos, n_neg) < min_class:
        return None, rate
    return res, rate


def figure_coverage_size_1x3(scenario, cs, method_meta, *, nominal=0.95, log_y=True):
    """Coverage vs median CS size, pooled over the 4 set sizes within each regime (1x3)."""
    grid = _cs_beta_grid(); inom = int(np.argmin(np.abs(grid - nominal))); nomv = float(grid[inom])
    order = method_order(method_meta); ls_by = linestyle_by_method(method_meta)
    base = cs.filter(pl.col("l") == 0)

    def cell(ax, regime, aliases):
        sub = base.filter(pl.col("collection_name").is_in(aliases))
        for meth in order:
            g = sub.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            r = _cov_size_core(g, grid, inom)
            if r is None:
                continue
            cov, med, nc, ns = r
            col = method_color(meth)
            ax.plot(cov, med, color=col, lw=1.4, ls=ls_by[meth])
            ax.plot(nc, ns, color=col, marker="o", markersize=5, markeredgecolor="k",
                    markeredgewidth=0.4, zorder=5)
        ax.axvline(nomv, color="grey", ls="--", lw=0.8)
        ax.set_xlim(0, 1.02)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("empirical coverage")
    fig, _ = facet_1x3(scenario, cell)
    fig.axes[0].set_ylabel("median CS size")
    return _finish(fig, method_meta, right=0.88)


def figure_logbf_roc_4x3(scenario, cs, method_meta, *, nominal=0.95):
    """Localization ROC faceted per cell (4x3). Sparser than the pooled panel - many
    strong-method cells are all-correct - but it exposes per-cell scenario differences."""
    grid = _cs_beta_grid(); inom = int(np.argmin(np.abs(grid - nominal)))
    order = method_order(method_meta); ls_by = linestyle_by_method(method_meta)
    base = cs.filter(pl.col("l") == 0)

    def cell(ax, alias, m, regime):
        g0 = base.filter(pl.col("collection_name") == alias)
        for meth in order:
            g = g0.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            res, rate = _roc_core(g, grid, inom)
            if res is None:
                continue
            fpr, tpr, auc = res
            ax.plot(fpr, tpr, color=method_color(meth), lw=1.3, ls=ls_by[meth])
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        if m == SIZES[-1]:
            ax.set_xlabel("FPR (mis-localized)")
    fig, _ = facet_grid(scenario, cell)
    return _finish(fig, method_meta)


def figure_logbf_roc_1x3(scenario, cs, method_meta, *, nominal=0.95):
    """Localization ROC pooled over the 4 set sizes within each regime (1x3). AUC per
    method annotated in a compact per-panel legend."""
    import matplotlib.pyplot as plt
    grid = _cs_beta_grid(); inom = int(np.argmin(np.abs(grid - nominal)))
    order = method_order(method_meta); ls_by = linestyle_by_method(method_meta)
    disp = {r["method"]: r["method_display"] for r in method_meta.iter_rows(named=True)}
    base = cs.filter(pl.col("l") == 0)

    def cell(ax, regime, aliases):
        sub = base.filter(pl.col("collection_name").is_in(aliases))
        handles = []
        for meth in order:
            g = sub.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            res, rate = _roc_core(g, grid, inom)
            if res is None:
                handles.append(plt.Line2D([], [], color=method_color(meth), ls=ls_by[meth],
                                          label=f"{disp[meth]} (n/a)"))
                continue
            fpr, tpr, auc = res
            ax.plot(fpr, tpr, color=method_color(meth), lw=1.5, ls=ls_by[meth])
            handles.append(plt.Line2D([], [], color=method_color(meth), ls=ls_by[meth],
                                      label=f"{disp[meth]} ({auc:.2f})"))
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        ax.set_xlabel("FPR (mis-localized)")
        ax.legend(handles=handles, fontsize=6.5, loc="lower right", frameon=False)
    fig, _ = facet_1x3(scenario, cell)
    fig.axes[0].set_ylabel("TPR (correctly localized)")
    fig.tight_layout()
    return fig


def figure_pip_calibration_grid(scenario, pip, method_meta, *, facet_by="regime"):
    """2D calibration facet grid: rows = the aggregation axis (regime or size), cols =
    method. Each panel is ONE calibration curve (empirical causal freq vs PIP midpoint)
    coloured by its method, aggregated over the OTHER design axis. facet_by='regime'
    aggregates over size (3 x 10); facet_by='size' aggregates over regime (4 x 10).
    """
    import matplotlib.pyplot as plt
    cal = viz_utils.expand_pip_calibration_from_compact(pip, method_meta)
    order = method_order(method_meta)
    disp = {r["method"]: r["method_display"] for r in method_meta.iter_rows(named=True)}
    scen_aliases = {cell_alias(scenario, m, r) for m in SIZES for r in REGIMES}
    parsed = (
        cal.filter(pl.col("collection_name").is_in(list(scen_aliases)))
        .with_columns(pl.col("collection_name").str.split("_").alias("_p"))
        .with_columns(
            pl.col("_p").list.get(1).str.strip_chars("m").cast(pl.Int64).alias("size"),
            pl.col("_p").list.get(2).alias("regime"),
        )
    )
    rowvals = REGIMES if facet_by == "regime" else SIZES
    keycol = "regime" if facet_by == "regime" else "size"

    nrow, ncol = len(rowvals), len(order)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.65 * ncol, 1.8 * nrow), squeeze=False,
                             sharex=True, sharey=True)
    for i, rv in enumerate(rowvals):
        for j, meth in enumerate(order):
            ax = axes[i][j]
            gg = (parsed.filter((pl.col("method") == meth) & (pl.col(keycol) == rv))
                  .group_by("pip_mid").agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
                  .sort("pip_mid"))
            gg = gg.filter(pl.col("n_total") > 0).with_columns(
                (pl.col("n_causal") / pl.col("n_total")).alias("rate"))
            ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.7)
            if gg.height:
                ax.plot(gg["pip_mid"], gg["rate"], color=method_color(meth), lw=1.3,
                        marker="o", markersize=2.2)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_box_aspect(1)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(disp[meth], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{keycol}={rv}", fontsize=9)
    fig.suptitle(f"{scenario}: PIP calibration - {keycol} (rows) x method (cols), "
                 f"aggregated over {'size' if facet_by=='regime' else 'regime'}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def figure_pip_calibration_by_method(scenario, pip, method_meta, *, color_by="regime", ncols=5):
    """Each method gets its own square calibration panel; lines are aggregated over the
    OTHER design axis and colour-coded. color_by='regime' -> aggregate over size (3 lines);
    color_by='size' -> aggregate over regime (4 lines). Replaces the unreadable
    all-methods-overlaid calibration."""
    import matplotlib.pyplot as plt
    cal = viz_utils.expand_pip_calibration_from_compact(pip, method_meta)
    order = method_order(method_meta)
    disp = {r["method"]: r["method_display"] for r in method_meta.iter_rows(named=True)}
    scen_aliases = {cell_alias(scenario, m, r) for m in SIZES for r in REGIMES}
    cal = cal.filter(pl.col("collection_name").is_in(list(scen_aliases)))
    # tag each row with its size + regime (parse alias) and the grouping key/colour
    parsed = cal.with_columns(
        pl.col("collection_name").str.split("_").alias("_parts"),
    ).with_columns(
        pl.col("_parts").list.get(1).str.strip_chars("m").cast(pl.Int64).alias("size"),
        pl.col("_parts").list.get(2).alias("regime"),
    )
    if color_by == "regime":
        groups, cmap, glabel = REGIMES, REGIME_COLORS, "regime"
        keycol = "regime"
    else:
        groups, cmap, glabel = SIZES, SIZE_COLORS, "size"
        keycol = "size"

    nrow = (len(order) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrow, ncols, figsize=(2.7 * ncols, 2.7 * nrow), squeeze=False)
    for idx, meth in enumerate(order):
        ax = axes[idx // ncols][idx % ncols]
        md = parsed.filter(pl.col("method") == meth)
        for gval in groups:
            gg = (md.filter(pl.col(keycol) == gval)
                  .group_by("pip_mid").agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
                  .sort("pip_mid"))
            gg = gg.filter(pl.col("n_total") > 0).with_columns(
                (pl.col("n_causal") / pl.col("n_total")).alias("rate"))
            if gg.height:
                ax.plot(gg["pip_mid"], gg["rate"], color=cmap[gval], lw=1.3,
                        marker="o", markersize=2.5, label=str(gval))
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3); ax.set_title(disp[meth], fontsize=9)
    # blank any unused panels
    for idx in range(len(order), nrow * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    # one legend keyed by the colour dimension
    handles = [plt.Line2D([], [], color=cmap[g], lw=2, label=str(g)) for g in groups]
    fig.legend(handles=handles, title=glabel, loc="lower right",
               bbox_to_anchor=(0.99, 0.02), fontsize=8, frameon=False)
    fig.suptitle(f"{scenario}: PIP calibration per method (colour = {glabel}, "
                 f"aggregated over {'size' if color_by=='regime' else 'regime'})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ===========================================================================
# Detection ROC (enriched vs paired null). Positive = enriched replicate,
# negative = its matched null_b0 replicate; score = SER logBF. Requires the
# cs frame loaded with include_null=True (carries the is_null column).
# ===========================================================================
def _detection_roc_for(sub, meth):
    """pos = enriched logBFs, neg = de-duplicated null logBFs, for one method."""
    g = sub.filter((pl.col("method") == meth) & (pl.col("l") == 0))
    pos = g.filter(~pl.col("is_null"))["ser_log_bf"].to_numpy()
    neg = (g.filter(pl.col("is_null"))
           .unique(subset=["sample_id", "batch_hash"])["ser_log_bf"].to_numpy())
    if len(pos) == 0 or len(neg) == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return _roc(scores, labels)


def figure_detection_roc(scenario, cs_pn, method_meta, *, aliases=None, title=None):
    """Pooled detection ROC over the 12 cells of a scenario (single panel, AUC per method)."""
    import matplotlib.pyplot as plt
    order = method_order(method_meta); ls_by = linestyle_by_method(method_meta)
    disp = {r["method"]: r["method_display"] for r in method_meta.iter_rows(named=True)}
    if aliases is None:
        aliases = [cell_alias(scenario, m, r) for m in SIZES for r in REGIMES]
    base = cs_pn.filter((pl.col("l") == 0) & pl.col("collection_name").is_in(aliases))
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    handles = []
    for meth in order:
        res = _detection_roc_for(base, meth)
        if res is None:
            continue
        fpr, tpr, auc = res
        ax.plot(fpr, tpr, color=method_color(meth), lw=1.6, ls=ls_by[meth])
        handles.append(plt.Line2D([], [], color=method_color(meth), ls=ls_by[meth],
                                  label=f"{disp[meth]} (AUC {auc:.2f})"))
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="false-positive rate (nulls flagged)",
           ylabel="true-positive rate (enriched flagged)",
           title=title or f"{scenario}: detection ROC (enriched vs null, pooled)")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def figure_detection_roc_4x3(scenario, cs_pn, method_meta):
    """Detection ROC faceted per cell (4x3): enriched vs that cell's matched null."""
    order = method_order(method_meta); ls_by = linestyle_by_method(method_meta)
    base = cs_pn.filter(pl.col("l") == 0)

    def cell(ax, alias, m, regime):
        sub = base.filter(pl.col("collection_name") == alias)
        for meth in order:
            res = _detection_roc_for(sub, meth)
            if res is None:
                continue
            fpr, tpr, auc = res
            ax.plot(fpr, tpr, color=method_color(meth), lw=1.3, ls=ls_by[meth])
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        if m == SIZES[-1]:
            ax.set_xlabel("FPR")
    fig, _ = facet_grid(scenario, cell)
    return _finish(fig, method_meta)
