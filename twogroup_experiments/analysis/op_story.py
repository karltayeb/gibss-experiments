"""Lightweight data + figure layer for the *distilled* GO:BP operating-points talk figures.

The full landscape notebook (`op_landscape.py`) reloads every cell's pip/cs reductions off
disk - fine for the exhaustive view, too heavy for a presentation notebook we want to render
snappily. This module keeps the presentation cut fast in two ways:

  * the localization CSV figures read *only* the small `covsize__{scenario}_m{m}.csv`
    tables (95%-CS size + coverage per strength/regime/method), and
  * the reduction-backed 3x3 figures (power-FDP, detection ROC, CS-size forest) load
    pip/cs reductions **scoped to a single (scenario, m)** - a handful of cells, not the
    whole 24-cell cube - via `load_pip` / `load_cs`.

The 3x3 figures are drawn in **layers**: `methods` is a reveal-ordered list and `n_layers`
truncates it, so a talk can build the panel up one method at a time.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import polars as pl

import op_landscape as ol
import viz_utils

CSV_DIR = Path("analysis/figures/011-gobp-operating-points")
SC = "011-gobp-operating-points"
STRENGTHS = ["weak", "intermediate", "strong"]
REGIMES = ["enrich", "balanced", "signal"]
COV_OK = 0.93  # coverage >= this => filled marker; below => hollow (under-covers)

# Talk reveal order (thresholded methods kept only at tau=2; distinct colours so no
# line-style needed). Pass a truncation via n_layers to build the panel up in a talk.
#
# loc: oracle sets the ceiling, then the estimated CORRECT (loc) family costs you in the
# enrich regime, then the MISSPECIFIED (scale) family, then linear (sidesteps f1, tracks the
# oracle here), then the misspecified-but-analysable rank/binary models.
LOC_LAYERS = [
    "twogroup_oracle__L=1",
    "twogroup_loc_fam__L=1",       # correct family for loc
    "twogroup_scale_fam__L=1",     # misspecified family for loc
    "linear_fixed__L=1",
    "logistic_threshold__threshold=2.00__L=1",   # binary @ tau=2 ...
    "cox__threshold=2.00__L=1",                  # ... and rank-thresholded @ tau=2 (revealed together)
    "cox_reversed__L=1",
]

# scale: f1 = N(0, sigma^2) (heavier tails, NO mean shift). Now the CORRECT family (scale)
# works, the WRONG family (loc, hunting a mean that isn't there) fails everywhere, and
# linear - with no mean shift to see - returns tight credible sets that don't cover
# (confidently wrong). Same misspecified rank/binary models close it out.
SCALE_LAYERS = [
    "twogroup_oracle__L=1",
    "twogroup_scale_fam__L=1",     # correct family for scale
    "twogroup_loc_fam__L=1",       # misspecified family for scale
    "linear_fixed__L=1",
    "logistic_threshold__threshold=2.00__L=1",   # binary @ tau=2 ...
    "cox__threshold=2.00__L=1",                  # ... and rank-thresholded @ tau=2 (revealed together)
    "cox_reversed__L=1",
]

# Method display-name (as it appears in the covsize CSV) -> (short label, colour).
# Colours are the repo twogroup Okabe-Ito palette (viz_utils.method_color), pinned here by
# the raw method key so the talk figures match the landscape notebook.
_C = viz_utils.method_color
METHOD_STYLE = {
    "Linear SER":            ("Linear",               _C("linear_fixed")),
    "Twogroup SER (Oracle)": ("Twogroup (oracle f1)", _C("twogroup_oracle")),
    "Twogroup Loc SER":      ("Twogroup (est. f1)",   _C("twogroup_loc_fam")),
    "Twogroup Scale SER":    ("Twogroup (scale fam)", _C("twogroup_scale_fam")),
    "Logistic SER (@2)":     ("Logistic @2",          _C("logistic_threshold")),
    "Cox SER (@2)":          ("Cox @2",               _C("cox")),
    "Cox (reversed) SER":    ("Cox reversed",         _C("cox_reversed")),
}


def load_covsize(scenario: str, m: int, *, nominal: float = 0.95) -> dict:
    """(strength, regime, method) -> (size_q50, coverage) at the given nominal, read from the
    small exported covsize CSV (no reduction loading). Also returns nothing else - snappy."""
    path = CSV_DIR / f"covsize__{scenario}_m{m}.csv"
    tag = f"{nominal:g}"
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["nominal"] != tag and float(r["nominal"]) != nominal:
                continue
            out[(r["strength"], r["regime"], r["method"])] = (
                float(r["size_q50"]), float(r["coverage"]))
    return out


def _whole_collection(rows: dict) -> float:
    """Visual reference for 'CS = the whole collection' = the largest size seen in the file
    (a diffuse method returns ~every set), so it self-calibrates per scenario/size."""
    return max(sz for sz, _ in rows.values())


def figure_localization_story(scenario: str, m: int, methods: list[str], *,
                              strengths: list[str] | None = None, title: str | None = None):
    """95%-CS size (log) vs regime; columns = strength; `methods` overlaid. Filled marker =
    covers (>=COV_OK), hollow = under-covers. Dashed line = whole collection, dotted = <=10."""
    import matplotlib.pyplot as plt

    rows = load_covsize(scenario, m)
    strengths = strengths or STRENGTHS
    n_sets = _whole_collection(rows)
    x = range(len(REGIMES))

    fig, axes = plt.subplots(1, len(strengths), figsize=(3.7 * len(strengths), 3.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, strength in zip(axes, strengths):
        for meth in methods:
            label, color = METHOD_STYLE[meth]
            ys, covs = [], []
            for reg in REGIMES:
                sz, cov = rows[(strength, reg, meth)]
                ys.append(sz); covs.append(cov)
            ax.plot(x, ys, color=color, lw=1.8, zorder=2, label=label)
            for xi, (yi, ci) in enumerate(zip(ys, covs)):
                filled = ci >= COV_OK
                ax.plot(xi, yi, marker="o", ms=9, zorder=3,
                        mfc=color if filled else "white",
                        mec=color if filled else "#B00020", mew=1.8)
        ax.axhline(n_sets, color="0.6", ls="--", lw=0.8)
        ax.axhline(10, color="0.6", ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(0.7, n_sets * 1.5)
        ax.set_xticks(list(x)); ax.set_xticklabels(REGIMES, fontsize=9)
        ax.set_title(strength, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, which="both")
    axes[0].set_ylabel("95%-CS size (gene-sets)", fontsize=10)
    axes[0].text(0.03, n_sets, "whole collection", fontsize=7.5, color="0.5", va="bottom")
    axes[0].text(0.03, 10, "localized (≤10)", fontsize=7.5, color="0.5", va="bottom")
    axes[-1].legend(fontsize=8.5, loc="center right", frameon=False)
    fig.suptitle(title or f"{scenario}  ·  m={m}  —  95%-CS size across the regime "
                 f"(hollow = under-covers <{COV_OK})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ===========================================================================
# Reduction-backed 3x3 talk figures (rows = strength, cols = regime), scoped to
# a single (scenario, m) so they load only a handful of cells. Methods are drawn
# in reveal order; n_layers truncates the list for progressive talk builds.
# ===========================================================================
from functools import lru_cache


@lru_cache(maxsize=1)
def method_meta():
    """method_display / colour / threshold table for the 011 methods (cached)."""
    return ol.method_meta(SC)


def _scoped(kind: str, scenario: str, m: int, *, include_null: bool = False) -> pl.DataFrame:
    """Load the `kind` reduction (pip|cs) for ONLY the (scenario, m) cells (+ their paired
    nulls if include_null), tagged collection_name=alias and is_null. Reuses op_landscape's
    per-prefix loader but restricts the cell list, so a talk figure never touches the other
    set sizes."""
    name_by_hash = ol._name_by_hash("results")
    keep = ol.methods(SC)
    frames = []
    for cell in ol.cells(SC):
        if cell["scenario"] != scenario or cell["m"] != m:
            continue
        frames += ol._load_prefix(kind, "results", name_by_hash, cell["batch_prefix"],
                                  cell["alias"], False, keep)
        if include_null and cell["null_prefix"]:
            frames += ol._load_prefix(kind, "results", name_by_hash, cell["null_prefix"],
                                      cell["alias"], True, keep)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def load_pip(scenario: str, m: int) -> pl.DataFrame:
    return _scoped("pip", scenario, m)


def load_cs(scenario: str, m: int, *, include_null: bool = False) -> pl.DataFrame:
    return _scoped("cs", scenario, m, include_null=include_null)


def _layers(methods, n_layers):
    return methods[:n_layers] if n_layers else methods


def _facet_3x3(scenario, m, cell_fn, *, sharex=True, sharey=True, suptitle=None,
               figsize_scale=(3.3, 3.0)):
    """3x3 grid, rows = strength, cols = regime, for one (scenario, m). cell_fn(ax, alias,
    strength, regime); alias = the collection_name '{scenario}_m{m}_{strength}_{regime}'."""
    import matplotlib.pyplot as plt
    nrow, ncol = len(STRENGTHS), len(REGIMES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize_scale[0] * ncol, figsize_scale[1] * nrow),
                             sharex=sharex, sharey=sharey, squeeze=False)
    for i, s in enumerate(STRENGTHS):
        for j, r in enumerate(REGIMES):
            ax = axes[i, j]
            cell_fn(ax, f"{scenario}_m{m}_{s}_{r}", s, r)
            if i == 0:
                ax.set_title(r, fontsize=11)
            if j == 0:
                ax.set_ylabel(s, fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    return fig, axes


def _disp(mm):
    return {r["method"]: r["method_display"] for r in mm.iter_rows(named=True)}


# Concise, family-labelled legend names (the meta display names are long and clipped a
# right-side legend). "loc-family" vs "scale-family" is exactly the correct-vs-misspecified
# axis; the prose says which is correct per scenario.
SHORT_LABEL = {
    "twogroup_oracle__L=1": "TG oracle",
    "twogroup_loc_fam__L=1": "TG loc-family",
    "twogroup_scale_fam__L=1": "TG scale-family",
    "linear_fixed__L=1": "Linear",
    "logistic_threshold__threshold=2.00__L=1": "Logistic @2",
    "cox__threshold=2.00__L=1": "Cox @2",
    "cox_reversed__L=1": "Cox reversed",
}


def _method_legend(fig, mm, methods):
    """Single-row legend BELOW the grid (avoids the right-edge clipping the old vertical
    legend hit with long labels + many methods). Concise family labels; bottom margin
    reserved so it never overlaps the last-row x-axis labels."""
    import matplotlib.pyplot as plt
    disp = _disp(mm)
    handles = [plt.Line2D([], [], color=ol.method_color(x), lw=2.6,
                          label=SHORT_LABEL.get(x, disp.get(x, x))) for x in methods]
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(methods), fontsize=9.5, frameon=False, handlelength=1.6,
               columnspacing=1.4)


def figure_power_fdp(scenario, m, pip, mm, methods, *, n_layers=None, max_fdp=0.5):
    """Power vs FDP per cell (PIP sweep), 3x3, methods in reveal order (n_layers truncates)."""
    import matplotlib.pyplot as plt
    methods = _layers(methods, n_layers)
    pf = viz_utils.expand_power_fdp_from_compact(pip, mm, selected_methods=set(methods))

    def cell(ax, alias, s, r):
        d = pf.filter(pl.col("simulation_name") == alias)
        for meth in methods:
            md = d.filter(pl.col("method") == meth).sort("pip_threshold", descending=True)
            if md.height:
                ax.plot(md["fdp"], md["power"], color=ol.method_color(meth), lw=1.7)
        ax.set_xlim(0, max_fdp); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        if s == STRENGTHS[-1]:
            ax.set_xlabel("FDP")

    fig, _ = _facet_3x3(scenario, m, cell,
                        suptitle=f"{scenario} · m={m} — power vs FDP (rows=strength, cols=regime)")
    _method_legend(fig, mm, methods)
    return fig


def figure_detection_roc(scenario, m, cs_pn, mm, methods, *, n_layers=None):
    """logBF detection ROC per cell (enriched vs paired null, score = ser_log_bf), 3x3,
    methods in reveal order. `cs_pn` must be a cs frame loaded with include_null=True."""
    methods = _layers(methods, n_layers)
    base = cs_pn.filter(pl.col("l") == 0)

    def cell(ax, alias, s, r):
        sub = base.filter(pl.col("collection_name") == alias)
        for meth in methods:
            res = ol._detection_roc_for(sub, meth)
            if res is None:
                continue
            fpr, tpr, _ = res
            ax.plot(fpr, tpr, color=ol.method_color(meth), lw=1.7)
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        if s == STRENGTHS[-1]:
            ax.set_xlabel("FPR (nulls flagged)")

    fig, _ = _facet_3x3(scenario, m, cell,
                        suptitle=f"{scenario} · m={m} — logBF detection ROC (rows=strength, cols=regime)")
    _method_legend(fig, mm, methods)
    return fig


def _cs_size_stats(g, inom, nominal):
    """(q25, q50, q75 of the 95%-CS size, coverage) for one (cell, method) group, or None."""
    mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
    sizes = np.asarray(g["cs_sizes"].to_list(), dtype=float)
    if sizes.ndim != 2 or sizes.shape[0] == 0:
        return None
    s = sizes[:, inom]
    q25, q50, q75 = np.quantile(s, [0.25, 0.5, 0.75])
    return float(q25), float(q50), float(q75), float(np.mean(mass < nominal))


def figure_cs_size(scenario, m, cs, mm, methods, *, n_layers=None, nominal=0.95):
    """CS-size 'forest' per cell: one horizontal row per method (reveal order, top -> bottom),
    dot = median 95%-CS size on a LOG axis, whisker = q25-q75 over replicates, marker filled
    if the method covers (>= COV_OK) else hollow. Dashed vline = whole collection, dotted = 10.
    Replaces the unreadable overlaid coverage-vs-size curves. 3x3, methods layer in."""
    methods = _layers(methods, n_layers)
    grid = ol._cs_beta_grid(); inom = int(np.argmin(np.abs(grid - nominal)))
    base = cs.filter(pl.col("l") == 0)
    n_sets = float(np.nanmax(np.asarray(  # whole-collection reference = largest CS seen
        [np.asarray(v, float).max() if len(v) else np.nan
         for v in base["cs_sizes"].to_list()])))

    def cell(ax, alias, s, r):
        sub = base.filter(pl.col("collection_name") == alias)
        for k, meth in enumerate(methods):
            g = sub.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            stat = _cs_size_stats(g, inom, nominal)
            if stat is None:
                continue
            q25, q50, q75, cov = stat
            y = len(methods) - 1 - k                      # first method at the top
            color = ol.method_color(meth); filled = cov >= COV_OK
            ax.plot([max(q25, 0.8), max(q75, 0.8)], [y, y], color=color, lw=2.4, alpha=0.7,
                    zorder=2, solid_capstyle="round")
            ax.plot(max(q50, 0.8), y, marker="o", ms=8, zorder=3,
                    mfc=color if filled else "white",
                    mec=color if filled else "#B00020", mew=1.8)
        ax.axvline(n_sets, color="0.6", ls="--", lw=0.8)
        ax.axvline(10, color="0.6", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_xlim(0.8, n_sets * 1.6)
        ax.set_ylim(-0.6, len(methods) - 0.4); ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3, which="both")
        if s == STRENGTHS[-1]:
            ax.set_xlabel("95%-CS size")

    fig, _ = _facet_3x3(scenario, m, cell, sharey=True,
                        suptitle=f"{scenario} · m={m} — 95%-CS size (median · IQR; "
                                 f"hollow = under-covers <{COV_OK}); rows=strength, cols=regime")
    _method_legend(fig, mm, methods)
    return fig


# ===========================================================================
# High-quality PDF export, following the slides/ deck standard (scripts/_common.save):
# vector PDF at dpi 200 / bbox tight, each with a <name>.prov.json provenance sidecar
# (script, git commit, timestamp, params). Decoupled from the quarto render so the
# figures can drop straight into the beamer deck.
# ===========================================================================
def _git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _save_pdf(fig, out: Path, params: dict) -> None:
    """Save `fig` as a vector PDF (dpi 200, bbox tight) + a .prov.json sidecar, matching
    slides/scripts/_common.save (the deck's provenance standard)."""
    import datetime as dt
    import json
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", dpi=200, bbox_inches="tight")
    prov = {
        "output": out.name,
        "script": "analysis/op_story.py",
        "git_commit": _git_hash(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "params": params,
    }
    out.with_suffix(".prov.json").write_text(json.dumps(prov, indent=2) + "\n")
    plt.close(fig)


# scenario -> its reveal-ordered method list (for the export loop).
LAYERS_BY_SCENARIO = {"loc": LOC_LAYERS, "scale": SCALE_LAYERS}


def export_figures(outdir=str(CSV_DIR), *, scenarios=("loc", "scale"), m=100,
                   per_layer=False):
    """Write the three talk figures (power-FDP, detection ROC, CS-size forest) for each
    scenario to `outdir` as deck-grade PDFs (`talk_{metric}__{scenario}_m{m}.pdf`) with a
    .prov.json sidecar each. With per_layer=True also emits the progressive-reveal stages
    `..._n{k}.pdf` (k = 1..len(methods)) for building the panel up in a talk. Returns the
    list of written PDF paths."""
    out = Path(outdir)
    mm = method_meta()
    written = []
    for scen in scenarios:
        methods = LAYERS_BY_SCENARIO[scen]
        pip = load_pip(scen, m)
        cs = load_cs(scen, m)
        cs_pn = load_cs(scen, m, include_null=True)
        # (metric name, builder, args) so full + per-layer share one path.
        specs = [
            ("powerfdp", figure_power_fdp, (pip, mm)),
            ("roc", figure_detection_roc, (cs_pn, mm)),
            ("cssize", figure_cs_size, (cs, mm)),
        ]
        ks = range(1, len(methods) + 1) if per_layer else [len(methods)]
        for metric, builder, args in specs:
            for k in ks:
                suffix = "" if k == len(methods) and not per_layer else f"_n{k}"
                path = out / f"talk_{metric}__{scen}_m{m}{suffix}.pdf"
                fig = builder(scen, m, *args, methods, n_layers=k)
                _save_pdf(fig, path, {
                    "scenario": scen, "m": m, "metric": metric,
                    "methods": methods[:k], "n_layers": k,
                    "supercollection": SC,
                    "source": "reductions (pip/cs) scoped to (scenario, m)",
                })
                written.append(str(path))
    return written
