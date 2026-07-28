"""Config-driven data + faceting layer for the GO:BP operating-points landscape notebook.

Works for ANY 011-family supercollection (the tiny `011-pilot` or the full
`011-gobp-operating-points`): cells, methods, and their on-disk batch names are all
discovered from `experiments/*.yaml` via the loader + manifest, so nothing is
transcribed and a partial run (the pilot) renders as a partial grid.

The design cube has, per scenario in {loc, scale}:

    set size m in {50,100,200,400}
    strength (localization-evidence tier) in {weak, intermediate, strong}   (feature_log_bf 10/15/20)
    regime (arc-length position on the iso-evidence curve) in {enrich, balanced, signal}

Primary figure (chosen layout): a 3x3 grid, rows = strength, cols = regime, one figure
per (m, scenario); m and scenario are paged across figures. `facet_sr` builds that grid
and renders only the cells present on disk (blank panels otherwise), so the pilot's
loc/m=100/{weak,strong} slice fills 6 of 9 panels and the rest come in with the full run.
"""
from __future__ import annotations

import glob
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

import viz_utils
from experiments import loader as L

# Canonical axis orders (top->bottom = increasing evidence; left->right = enrichment->signal).
STRENGTHS = ["weak", "intermediate", "strong"]
REGIMES = ["enrich", "balanced", "signal"]
SIZES = [50, 100, 200, 400]
SCENARIOS = ["loc", "scale"]
B0 = -2.0


# ---------------------------------------------------------------------------
# Cell + method discovery (from the experiment config, not hardcoded).
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _config():
    return L.load_config()


def supercollection(sc_name: str) -> dict:
    cfg = _config()
    if sc_name not in cfg["supercollections"]:
        raise KeyError(f"{sc_name!r} not in config; have {sorted(cfg['supercollections'])}")
    return cfg["supercollections"][sc_name]


def _parse_alias(alias: str) -> dict | None:
    """`{scenario}_m{size}_{strength}_{regime}` -> parsed dict, or None if it doesn't match."""
    parts = alias.split("_")
    if len(parts) != 4 or parts[0] not in SCENARIOS or not parts[1].startswith("m"):
        return None
    try:
        m = int(parts[1][1:])
    except ValueError:
        return None
    if parts[2] not in STRENGTHS or parts[3] not in REGIMES:
        return None
    return {"scenario": parts[0], "m": m, "strength": parts[2], "regime": parts[3]}


def cells(sc_name: str) -> list[dict]:
    """One dict per collection: alias, (scenario, m, strength, regime), the enriched
    batch-name prefix, and the paired-null batch-name prefix. Batch names are
    `{design}__{enrichment}__{signal}` (loader.resolve_simulation), so the prefixes
    are read straight from the collection's two simulations."""
    out = []
    for coll in supercollection(sc_name)["collections"]:
        parsed = _parse_alias(coll["name"])
        if parsed is None:
            continue
        sims = coll["simulations"]
        enr = next(s for s in sims if s["enrichment"] != "null_b0")
        nul = next((s for s in sims if s["enrichment"] == "null_b0"), None)
        out.append({
            "alias": coll["name"], **parsed,
            "batch_prefix": f"{enr['design']}__{enr['enrichment']}__{enr['signal']}",
            "null_prefix": None if nul is None else f"{nul['design']}__null_b0__{nul['signal']}",
        })
    return out


def methods(sc_name: str) -> list[str]:
    """Ordered method names for the SC (its `methods:` anchor)."""
    return list(supercollection(sc_name)["methods"])


def method_meta(sc_name: str, results_root: str = "results") -> pl.DataFrame:
    """method_display / is_oracle / is_thresholded / threshold table for the SC's methods."""
    manifest = json.load(open(f"{results_root}/manifest_cache.json"))
    want = set(methods(sc_name))
    specs = {m["name"]: m for m in manifest["methods"].values() if m["name"] in want}
    missing = want - set(specs)
    if missing:
        raise AssertionError(f"methods missing from manifest: {sorted(missing)}")
    return L.method_metadata(specs)


# ---------------------------------------------------------------------------
# Reduction loading (read reductions/*.parquet off disk, tag with cell alias + is_null).
# ---------------------------------------------------------------------------
def _name_by_hash(results_root: str) -> dict[str, str]:
    manifest = json.load(open(f"{results_root}/manifest_cache.json"))
    return {h: b["name"] for h, b in manifest["batches"].items()}


def _load_prefix(kind, results_root, name_by_hash, prefix, alias, is_null, keep_methods):
    frames = []
    for bh, name in name_by_hash.items():
        if not (name == prefix or name.startswith(prefix + "__batch")):
            continue
        for path in glob.glob(f"{results_root}/by_batch/{bh}/fits/*/reductions/{kind}.parquet"):
            df = pl.read_parquet(path).filter(pl.col("method").is_in(keep_methods))
            if df.height:
                frames.append(df.with_columns(
                    pl.lit(alias).alias("collection_name"),
                    pl.lit(is_null).alias("is_null"),
                ))
    return frames


def load_reduction(kind: str, sc_name: str, results_root: str = "results",
                   *, include_null: bool = False) -> pl.DataFrame:
    """Concat the `kind` reduction (pip|cs) for every cell of the SC, tagged
    collection_name=alias and is_null. With include_null, also load each cell's paired
    null_b0 batch (needed for the detection ROC). Cells/files absent on disk are skipped,
    so a partial (pilot) run returns only what ran."""
    name_by_hash = _name_by_hash(results_root)
    keep = methods(sc_name)
    frames = []
    for cell in cells(sc_name):
        frames += _load_prefix(kind, results_root, name_by_hash, cell["batch_prefix"],
                               cell["alias"], False, keep)
        if include_null and cell["null_prefix"]:
            frames += _load_prefix(kind, results_root, name_by_hash, cell["null_prefix"],
                                   cell["alias"], True, keep)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def available(sc_name: str) -> pl.DataFrame:
    """(scenario, m) pairs that have at least one cell defined, for paging figures."""
    return (pl.DataFrame(cells(sc_name)).select("scenario", "m").unique()
            .sort(["scenario", "m"]))


# ---------------------------------------------------------------------------
# Method colour / order / linestyle (twogroup Okabe-Ito palette; cox threshold -> ls).
# ---------------------------------------------------------------------------
THRESH_LS = {None: "-", 1.0: "-", 2.0: "--", 3.0: "-.", 4.0: (0, (1, 1))}


def method_color(m: str) -> str:
    return viz_utils.method_color(m)


def method_order(mm: pl.DataFrame) -> list[str]:
    return (mm.sort(["is_oracle", "is_thresholded", "method"], descending=[True, False, False])
            .get_column("method").to_list())


def linestyle_by_method(mm: pl.DataFrame) -> dict:
    return {r["method"]: THRESH_LS.get(r["threshold"], "-") for r in mm.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# 3x3 faceting: rows = strength, cols = regime, one (scenario, m) per figure.
# ---------------------------------------------------------------------------
def cell_alias(scenario: str, m: int, strength: str, regime: str) -> str:
    return f"{scenario}_m{m}_{strength}_{regime}"


def present_axes(sc_name, scenario, m) -> tuple[list[str], list[str]]:
    """The strength rows and regime cols actually present on disk for (scenario, m), in
    canonical order. The grid uses only these, so the pilot is 2x3 (weak, strong) and the
    full run is 3x3 (weak, intermediate, strong) with no code change."""
    have = {(c["strength"], c["regime"]) for c in cells(sc_name)
            if c["scenario"] == scenario and c["m"] == m}
    rows = [s for s in STRENGTHS if any((s, r) in have for r in REGIMES)]
    cols = [r for r in REGIMES if any((s, r) in have for s in STRENGTHS)]
    return rows, cols


def facet_sr(sc_name, scenario, m, cell_fn, *, figsize_scale=(3.3, 3.0), sharex=True,
             sharey=True, suptitle=None):
    """Grid (rows = strength, cols = regime) for one (scenario, m), sized to the tiers
    actually present: 2x3 for the pilot (weak/strong), 3x3 for the full run. Calls
    cell_fn(ax, alias, strength, regime); a (strength, regime) hole inside the present
    rows/cols is left labelled. Row labels = strength, column titles = regime."""
    import matplotlib.pyplot as plt
    strengths, regimes = present_axes(sc_name, scenario, m)
    if not strengths or not regimes:
        raise ValueError(f"no cells on disk for {scenario} m={m}")
    have = {(c["strength"], c["regime"]) for c in cells(sc_name)
            if c["scenario"] == scenario and c["m"] == m}
    nrow, ncol = len(strengths), len(regimes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize_scale[0] * ncol, figsize_scale[1] * nrow),
                             sharex=sharex, sharey=sharey, squeeze=False)
    for i, strength in enumerate(strengths):
        for j, regime in enumerate(regimes):
            ax = axes[i, j]
            if (strength, regime) in have:
                cell_fn(ax, cell_alias(scenario, m, strength, regime), strength, regime)
            else:
                ax.text(0.5, 0.5, "(not run)", ha="center", va="center", fontsize=8,
                        color="0.6", transform=ax.transAxes)
            if i == 0:
                ax.set_title(regime, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{strength}", fontsize=10)
    fig.suptitle(suptitle or f"{scenario}  ·  m = {m}", fontsize=13)
    return fig, axes


def _shared_method_legend(fig, mm, *, bbox=(0.87, 0.5)):
    import matplotlib.pyplot as plt
    order = method_order(mm)
    disp = {r["method"]: r["method_display"] for r in mm.iter_rows(named=True)}
    ls_by = linestyle_by_method(mm)
    handles = [plt.Line2D([], [], color=method_color(m), lw=2.0, ls=ls_by[m], label=disp[m])
               for m in order]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=bbox, fontsize=8, frameon=False)


def _finish(fig, mm, right=0.85):
    fig.tight_layout(rect=[0, 0, right, 1])
    _shared_method_legend(fig, mm, bbox=(right + 0.01, 0.5))
    return fig


# ---------------------------------------------------------------------------
# Headline metrics (3x3 per (scenario, m)): power-FDP, coverage-vs-CS-size, calibration.
# ---------------------------------------------------------------------------
def figure_power_fdp(sc_name, scenario, m, pip, mm, *, max_fdp=0.5):
    pf = viz_utils.expand_power_fdp_from_compact(pip, mm, selected_methods=set(methods(sc_name)))
    order, ls_by = method_order(mm), linestyle_by_method(mm)

    def cell(ax, alias, strength, regime):
        d = pf.filter(pl.col("simulation_name") == alias)
        for meth in order:
            md = d.filter(pl.col("method") == meth).sort("pip_threshold", descending=True)
            if md.height:
                ax.plot(md["fdp"], md["power"], color=method_color(meth), lw=1.4, ls=ls_by[meth])
        ax.set_xlim(0, max_fdp); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        if strength == STRENGTHS[-1]:
            ax.set_xlabel("FDP")

    fig, _ = facet_sr(sc_name, scenario, m, cell,
                      suptitle=f"{scenario} · m={m} — power vs FDP (rows=strength, cols=regime)")
    return _finish(fig, mm)


# Two consistent method subsets used by ALL calibration plots (shared ordering):
#   CROSS   - cross-method comparison: twogroup oracle/loc/scale, linear, cox-reversed, cox@2, logistic@2
#   THRESH  - the cox/logistic threshold sweep at tau = 1..4
CROSS_METHODS = [
    "twogroup_oracle__L=1", "twogroup_loc_fam__L=1", "twogroup_scale_fam__L=1",
    "linear_fixed__L=1", "cox_reversed__L=1", "cox__threshold=2.00__L=1",
    "logistic_threshold__threshold=2.00__L=1",
]
THRESHOLD_METHODS = [
    "cox__threshold=1.00__L=1", "cox__threshold=2.00__L=1",
    "cox__threshold=3.00__L=1", "cox__threshold=4.00__L=1",
    "logistic_threshold__threshold=1.00__L=1", "logistic_threshold__threshold=2.00__L=1",
    "logistic_threshold__threshold=3.00__L=1", "logistic_threshold__threshold=4.00__L=1",
]
SPOTLIGHT_METHODS = CROSS_METHODS  # back-compat alias


def _present_methods(methods, mm):
    have = set(mm.get_column("method").to_list())
    return [x for x in methods if x in have]


def _calibration_panel(ax, gg, color):
    """One calibration panel: sized-by-bin-count scatter (method colour) + diagonal."""
    import numpy as _np
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
    if gg.height:
        x = gg["pip_mid"].to_numpy(); y = gg["rate"].to_numpy(); n = gg["n_total"].to_numpy()
        ax.scatter(x, y, s=8 + 120 * n / max(n.max(), 1), color=color, alpha=0.85,
                   edgecolor="k", linewidth=0.3, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_box_aspect(1); ax.grid(True, alpha=0.3)


def _cal_agg(cal, **filters):
    q = cal
    for col, val in filters.items():
        q = q.filter(pl.col(col) == val)
    return (q.group_by("pip_mid").agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
            .sort("pip_mid").filter(pl.col("n_total") > 0)
            .with_columns((pl.col("n_causal") / pl.col("n_total")).alias("rate")))


def _calibration_grid(cal, regimes, methods, mm, suptitle, *, colw=2.2):
    """Shared calibration grid: rows = regime, cols = the given `methods` (kept in order).
    `cal` must already be scoped to the cells of interest and carry a `regime` column. One
    method per panel, method colour, dots sized by #sets in the PIP bin, dotted diagonal."""
    import matplotlib.pyplot as plt
    disp = {r["method"]: r["method_display"] for r in mm.iter_rows(named=True)}
    nrow, ncol = len(regimes), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(colw * ncol, 2.4 * nrow), squeeze=False,
                             sharex=True, sharey=True)
    for i, r in enumerate(regimes):
        for j, meth in enumerate(methods):
            ax = axes[i][j]
            _calibration_panel(ax, _cal_agg(cal, method=meth, regime=r), method_color(meth))
            if i == 0:
                ax.set_title(disp[meth], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{r}\nemp. fraction", fontsize=8)
            if i == nrow - 1:
                ax.set_xlabel("PIP", fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def figure_pip_calibration(sc_name, scenario, m, pip, mm, *, methods=None):
    """PIP calibration for one (scenario, set size): rows = **regime**, cols = the chosen
    `methods` subset (default CROSS_METHODS), **aggregated over strength**. Repo convention:
    one method per panel, method colour, dots sized by #sets in the PIP bin, diagonal =
    calibrated. Pass `methods=THRESHOLD_METHODS` for the cox/logistic tau-sweep view."""
    methods = _present_methods(methods or CROSS_METHODS, mm)
    cal = viz_utils.expand_pip_calibration_from_compact(pip, mm)
    _, regimes = present_axes(sc_name, scenario, m)
    cal = (cal.filter(pl.col("collection_name").str.starts_with(f"{scenario}_m{m}_"))
           .with_columns(pl.col("collection_name").str.split("_").list.get(3).alias("regime")))
    return _calibration_grid(
        cal, regimes, methods, mm,
        f"{scenario} · m={m} — PIP calibration (rows = regime, cols = method; "
        f"aggregated over strength; dot size ∝ #sets in bin)")


def figure_pip_calibration_spotlight(sc_name, scenario, pip, mm, *, methods=None):
    """Spotlight PIP calibration for one scenario: rows = **regime**, cols = the chosen
    `methods` subset (default CROSS_METHODS - the 7-method cross comparison; pass
    THRESHOLD_METHODS for the cox/logistic tau-sweep), **pooled over strength AND set
    size**. One method per panel, method colour, dots sized by #sets in the PIP bin."""
    methods = _present_methods(methods or CROSS_METHODS, mm)
    cal = viz_utils.expand_pip_calibration_from_compact(pip, mm)
    cal = (cal.filter(pl.col("collection_name").str.starts_with(f"{scenario}_"))
           .with_columns(pl.col("collection_name").str.split("_").list.get(3).alias("regime")))
    return _calibration_grid(
        cal, REGIMES, methods, mm,
        f"{scenario} — PIP calibration (rows = regime, cols = method; pooled over "
        f"strength + set size; dot size ∝ #sets in bin)", colw=1.95)


def _cs_beta_grid():
    from utils import CS_BETA_GRID
    return np.asarray(CS_BETA_GRID.tolist())


def _cov_size(g, grid, inom):
    mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
    sizes = np.asarray(g["cs_sizes"].to_list(), dtype=float)
    if sizes.ndim != 2 or sizes.shape[0] == 0:
        return None
    cov = np.array([np.mean(mass < b) for b in grid])
    med = np.median(sizes, axis=0)
    return cov, med, float(cov[inom]), float(med[inom])


# nominal CS levels marked on the coverage-vs-size curve (and reported in the table below it)
NOMINALS = [0.95, 0.80, 0.50]
NOMINAL_MARKERS = {0.95: "o", 0.80: "s", 0.50: "D"}


def figure_coverage_size(sc_name, scenario, m, cs, mm, *, nominals=NOMINALS, log_y=True):
    """Coverage vs MEDIAN CS size, swept over the beta grid. Marked points = the CS at each
    nominal beta (0.95 circle, 0.80 square, 0.50 diamond); a marked point's x vs the dashed
    line at its nominal reads as calibration (on the line = calibrated). Lower curve at a
    given coverage = tighter (more efficient) method."""
    import matplotlib.pyplot as plt
    grid = _cs_beta_grid()
    inoms = [(nv, int(np.argmin(np.abs(grid - nv)))) for nv in nominals]
    order, ls_by = method_order(mm), linestyle_by_method(mm)
    base = cs.filter(pl.col("l") == 0)

    def cell(ax, alias, strength, regime):
        sub = base.filter(pl.col("collection_name") == alias)
        for meth in order:
            g = sub.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            r = _cov_size(g, grid, inoms[0][1])
            if r is None:
                continue
            cov, med, _, _ = r
            col = method_color(meth)
            ax.plot(cov, med, color=col, lw=1.3, ls=ls_by[meth])
            for nv, i in inoms:
                ax.plot(cov[i], med[i], color=col, marker=NOMINAL_MARKERS[nv], markersize=5,
                        markeredgecolor="k", markeredgewidth=0.4, zorder=5)
        for nv, _ in inoms:
            ax.axvline(nv, color="grey", ls="--", lw=0.6)
        ax.set_xlim(0, 1.02)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        if strength == STRENGTHS[-1]:
            ax.set_xlabel("empirical coverage")

    fig, _ = facet_sr(sc_name, scenario, m, cell,
                      suptitle=f"{scenario} · m={m} — coverage vs median CS size (rows=strength, cols=regime)")
    fig = _finish(fig, mm)
    # second legend: marker shape -> nominal CS level
    mk = [plt.Line2D([], [], color="0.3", marker=NOMINAL_MARKERS[nv], ls="none", ms=6,
                     markeredgecolor="k", label=f"{nv:.2f} CS") for nv in nominals]
    fig.legend(handles=mk, loc="center left", bbox_to_anchor=(0.86, 0.16), fontsize=8,
               frameon=False, title="nominal")
    return fig


def coverage_size_table(sc_name, scenario, m, cs, mm, *, nominals=NOMINALS) -> pl.DataFrame:
    """Long table behind the coverage-vs-size figure: one row per (strength, regime, method,
    nominal). `coverage` = fraction of replicates whose causal set is inside the nominal-beta
    CS; `size_q25/q50/q75` = quartiles of that cell's CS size at the nominal beta (over the
    50 replicates). Ordered strength -> regime -> method -> nominal(desc)."""
    grid = _cs_beta_grid()
    inoms = [(float(nv), int(np.argmin(np.abs(grid - nv)))) for nv in nominals]
    by = {c["alias"]: c for c in cells(sc_name)}
    disp = {r["method"]: r["method_display"] for r in mm.iter_rows(named=True)}
    order = method_order(mm)
    srank = {s: i for i, s in enumerate(STRENGTHS)}; rrank = {r: i for i, r in enumerate(REGIMES)}
    mrank = {mth: i for i, mth in enumerate(order)}
    base = cs.filter(pl.col("l") == 0)
    rows = []
    for (coll, meth), g in base.group_by(["collection_name", "method"]):
        c = by.get(coll)
        if c is None or c["scenario"] != scenario or c["m"] != m or meth not in mrank:
            continue
        mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
        sizes = np.asarray(g["cs_sizes"].to_list(), dtype=float)  # (n_rep, n_beta)
        if sizes.ndim != 2 or sizes.shape[0] == 0:
            continue
        for nv, i in inoms:
            q25, q50, q75 = np.quantile(sizes[:, i], [0.25, 0.5, 0.75])
            rows.append({
                "strength": c["strength"], "regime": c["regime"], "method": disp[meth],
                "nominal": nv, "coverage": float(np.mean(mass < nv)),
                "size_q25": float(q25), "size_q50": float(q50), "size_q75": float(q75),
                "_s": srank[c["strength"]], "_r": rrank[c["regime"]], "_m": mrank[meth],
            })
    if not rows:
        return pl.DataFrame()
    return (pl.DataFrame(rows).sort(["_s", "_r", "_m", "nominal"], descending=[False, False, False, True])
            .drop("_s", "_r", "_m"))


def coverage_size_table_md(sc_name, scenario, m, cs, mm, *, nominals=NOMINALS) -> str:
    """Compact single markdown table (emit via R `cat` to dodge asis dropout): one row per
    (strength, regime, method), with a coverage + size column per nominal level. Size is
    reported as `median [q25, q75]` of the CS size at that level."""
    df = coverage_size_table(sc_name, scenario, m, cs, mm, nominals=nominals)
    hdr = ["strength", "regime", "method"]
    for nv in nominals:
        hdr += [f"cov {nv*100:.0f}", f"size {nv*100:.0f} (med [q25,q75])"]
    lines = [f"**{scenario} · m={m}**", "", "| " + " | ".join(hdr) + " |",
             "| " + " | ".join("---" for _ in hdr) + " |"]
    # group the long df back to one row per (strength, regime, method), keeping order
    seen = []
    for r in df.iter_rows(named=True):
        key = (r["strength"], r["regime"], r["method"])
        if key not in seen:
            seen.append(key)
    by = {(r["strength"], r["regime"], r["method"], r["nominal"]): r for r in df.iter_rows(named=True)}
    for key in seen:
        cells_ = [key[0], key[1], key[2]]
        for nv in nominals:
            r = by.get((key[0], key[1], key[2], float(nv)))
            if r is None:
                cells_ += ["-", "-"]
            else:
                cells_ += [f"{r['coverage']:.2f}",
                           f"{r['size_q50']:.0f} [{r['size_q25']:.0f}, {r['size_q75']:.0f}]"]
        lines.append("| " + " | ".join(cells_) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Localization ROC: coverage of the causal set vs the CS "footprint" (mean CS size as a
# fraction of all gene-sets), swept over the CS confidence level beta. A random set of
# footprint f covers the causal set with prob f, so the y=x diagonal is the EXACT chance
# baseline; a method that truly localizes bows to the top-left (high coverage, tiny
# footprint). This is the coverage-vs-size curve made an ROC with an absolute reference.
# ---------------------------------------------------------------------------
def _loc_roc_curve(g, grid):
    """(footprint x, coverage y) over the beta grid for one (cell, method) group.
    footprint(beta) = mean CS size at beta / #sets; coverage(beta) = P(causal in beta-CS)."""
    mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
    sizes = np.asarray(g["cs_sizes"].to_list(), dtype=float)  # (n_rep, n_beta)
    if sizes.ndim != 2 or sizes.shape[0] == 0:
        return None
    p = float(g["n_features"][0])
    cov = np.array([np.mean(mass < b) for b in grid])
    footprint = sizes.mean(axis=0) / p
    return footprint, cov


def figure_localization_roc(sc_name, scenario, m, cs, mm, *, nominal=0.95):
    """Localization ROC per cell (rows = strength, cols = regime), methods overlaid: y =
    coverage of the causal set, x = mean CS footprint (fraction of all sets), swept over the
    CS level. Dotted y=x diagonal = chance; the marked point is the nominal (95%) CS.
    Top-left = localizes; on the diagonal = the CS is no better than a random set that size."""
    grid = _cs_beta_grid(); inom = int(np.argmin(np.abs(grid - nominal)))
    order, ls_by = method_order(mm), linestyle_by_method(mm)
    base = cs.filter(pl.col("l") == 0)

    def cell(ax, alias, strength, regime):
        sub = base.filter(pl.col("collection_name") == alias)
        for meth in order:
            g = sub.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            r = _loc_roc_curve(g, grid)
            if r is None:
                continue
            x, y = r
            col = method_color(meth)
            ax.plot(x, y, color=col, lw=1.3, ls=ls_by[meth])
            ax.plot(x[inom], y[inom], color=col, marker="o", markersize=4,
                    markeredgecolor="k", markeredgewidth=0.4, zorder=5)
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        if strength == STRENGTHS[-1]:
            ax.set_xlabel("CS footprint (frac. of sets)")

    fig, _ = facet_sr(sc_name, scenario, m, cell,
                      suptitle=f"{scenario} · m={m} — localization ROC "
                               f"(coverage vs CS footprint; rows=strength, cols=regime)")
    return _finish(fig, mm)


def figure_localization_roc_spotlight(sc_name, scenario, cs, mm, *, methods=None, nominal=0.95):
    """Pooled localization ROC: 1×3 (cols = regime), methods overlaid, pooled over strength
    AND set size for one scenario. y = causal-set coverage, x = mean CS footprint; y=x = chance
    (random set); the marked point is the nominal (95%) CS. Default methods = CROSS_METHODS."""
    import matplotlib.pyplot as plt
    methods = _present_methods(methods or CROSS_METHODS, mm)
    ls_by = linestyle_by_method(mm)
    disp = {r["method"]: r["method_display"] for r in mm.iter_rows(named=True)}
    grid = _cs_beta_grid(); inom = int(np.argmin(np.abs(grid - nominal)))
    base = (cs.filter(pl.col("l") == 0)
            .filter(pl.col("collection_name").str.starts_with(f"{scenario}_"))
            .with_columns(pl.col("collection_name").str.split("_").list.get(3).alias("regime")))
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(4.6 * len(REGIMES), 4.4),
                             sharex=True, sharey=True, squeeze=False)
    for j, r in enumerate(REGIMES):
        ax = axes[0][j]
        sub = base.filter(pl.col("regime") == r)
        for meth in methods:
            g = sub.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            res = _loc_roc_curve(g, grid)
            if res is None:
                continue
            x, y = res
            col = method_color(meth)
            ax.plot(x, y, color=col, lw=1.6, ls=ls_by[meth])
            ax.plot(x[inom], y[inom], color=col, marker="o", markersize=5,
                    markeredgecolor="k", markeredgewidth=0.4, zorder=5)
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        ax.set_title(r, fontsize=11); ax.set_xlabel("CS footprint (frac. of sets)")
    axes[0][0].set_ylabel("coverage of causal set")
    handles = [plt.Line2D([], [], color=method_color(x), lw=2, ls=ls_by[x], label=disp[x])
               for x in methods]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=8,
               frameon=False)
    fig.suptitle(f"{scenario} — localization ROC (pooled over strength + set size)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    return fig


# ---------------------------------------------------------------------------
# Log-BF detection ROC: is ser_log_bf a trustworthy DETECTION statistic? Enriched
# replicate vs its paired null_b0 replicate, scored by the SER log Bayes factor.
# Broken out per cell across the strength×regime landscape (that breakout IS the point:
# detection should improve with strength and depend on regime). cs frame must be loaded
# with include_null=True (carries the is_null column).
# ---------------------------------------------------------------------------
def _roc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, float)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-scores); ls = labels[order]
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ls) / n_neg])
    ranks = scores.argsort(kind="mergesort").argsort() + 1
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return fpr, tpr, float(auc)


def _detection_roc_for(sub, meth):
    g = sub.filter((pl.col("method") == meth) & (pl.col("l") == 0))
    pos = g.filter(~pl.col("is_null"))["ser_log_bf"].to_numpy()
    # de-duplicate nulls by replicate (a signal-shared null_b0 batch is tagged under every
    # cell that references it, so pooling would otherwise count it many times).
    neg = g.filter(pl.col("is_null")).unique(subset=["sample_id"])["ser_log_bf"].to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return None
    return _roc(np.concatenate([pos, neg]),
               np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))


def figure_detection_roc(sc_name, scenario, m, cs_pn, mm):
    """logBF detection ROC broken out across the strength×regime landscape (rows = strength,
    cols = regime), one figure per (scenario, m). Per cell: positive = enriched replicate,
    negative = its paired null_b0 replicate, score = `ser_log_bf`; methods overlaid. This
    breakout is the point - detection should sharpen as strength grows and vary by regime."""
    order, ls_by = method_order(mm), linestyle_by_method(mm)
    base = cs_pn.filter(pl.col("l") == 0)

    def cell(ax, alias, strength, regime):
        sub = base.filter(pl.col("collection_name") == alias)
        for meth in order:
            res = _detection_roc_for(sub, meth)
            if res is None:
                continue
            fpr, tpr, _ = res
            ax.plot(fpr, tpr, color=method_color(meth), lw=1.4, ls=ls_by[meth])
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        if strength == STRENGTHS[-1]:
            ax.set_xlabel("FPR (nulls flagged)")

    fig, _ = facet_sr(sc_name, scenario, m, cell,
                      suptitle=f"{scenario} · m={m} — logBF detection ROC (rows=strength, cols=regime)")
    return _finish(fig, mm)


# ---------------------------------------------------------------------------
# Birds-eye landscape summary: per-method m x strength heatmap of one scalar,
# pooled over regime (recovers the size/evidence trend the 3x3 detail buries).
# ---------------------------------------------------------------------------
def landscape_scalar(sc_name, cs, mm, *, scenario, nominal=0.95, stat="cs_size"):
    """DataFrame (method, m, strength, value). stat:
       'cs_size'  -> median nominal-beta CS size (log-scaled in the heatmap);
       'loc_rate' -> fraction of replicates with the causal set inside the nominal-beta CS."""
    grid = _cs_beta_grid(); inom = int(np.argmin(np.abs(grid - nominal)))
    by = {c["alias"]: c for c in cells(sc_name)}
    base = cs.filter(pl.col("l") == 0)
    rows = []
    for (coll, meth), g in base.group_by(["collection_name", "method"]):
        c = by.get(coll)
        if c is None or c["scenario"] != scenario:
            continue
        mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
        if stat == "loc_rate":
            val = float(np.nanmean((mass < grid[inom]).astype(float)))
        else:
            sizes = np.asarray(g["cs_sizes"].to_list(), dtype=float)
            val = float(np.median(sizes[:, inom])) if sizes.ndim == 2 and sizes.shape[0] else np.nan
        rows.append({"method": meth, "m": c["m"], "strength": c["strength"], "value": val})
    # pool over regime: mean of the per-cell scalar across regimes present
    df = pl.DataFrame(rows)
    if df.height == 0:
        return df
    return df.group_by(["method", "m", "strength"]).agg(pl.col("value").mean()).sort(["method", "m", "strength"])


def figure_landscape_heatmap(sc_name, cs, mm, *, scenario, nominal=0.95, stat="cs_size"):
    """One small m×strength heatmap per method (pooled over regime): the birds-eye
    'method performance across the detectability landscape' summary."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    tab = landscape_scalar(sc_name, cs, mm, scenario=scenario, nominal=nominal, stat=stat)
    order = [m for m in method_order(mm) if tab.filter(pl.col("method") == m).height]
    disp = {r["method"]: r["method_display"] for r in mm.iter_rows(named=True)}
    ncol = min(len(order), 5) or 1
    nrow = (len(order) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.5 * nrow), squeeze=False)
    label = "median 95%-CS size" if stat == "cs_size" else "localization rate"
    vals = tab["value"].to_numpy()
    if stat == "cs_size":
        norm = LogNorm(vmin=max(np.nanmin(vals), 1.0), vmax=max(np.nanmax(vals), 2.0)); cmap = "viridis_r"
    else:
        norm = None; cmap = "viridis"
    im = None
    for idx, meth in enumerate(order):
        ax = axes[idx // ncol][idx % ncol]
        grid = np.full((len(STRENGTHS), len(SIZES)), np.nan)
        for r in tab.filter(pl.col("method") == meth).iter_rows(named=True):
            grid[STRENGTHS.index(r["strength"]), SIZES.index(r["m"])] = r["value"]
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                       vmin=None if stat == "cs_size" else 0, vmax=None if stat == "cs_size" else 1)
        ax.set_xticks(range(len(SIZES))); ax.set_xticklabels(SIZES, fontsize=7)
        ax.set_yticks(range(len(STRENGTHS))); ax.set_yticklabels(STRENGTHS, fontsize=7)
        ax.set_title(disp[meth], fontsize=8)
        if idx % ncol == 0:
            ax.set_ylabel("strength", fontsize=8)
        if idx // ncol == nrow - 1:
            ax.set_xlabel("m", fontsize=8)
    for idx in range(len(order), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=f"{label} (pooled over regime)")
    fig.suptitle(f"{scenario}: {label} across the m×strength landscape", fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# Export: regenerate every notebook figure to descriptively-named files (vector
# PDF by default) for dropping into slides. Decoupled from the quarto render.
# ---------------------------------------------------------------------------
def export_figures(sc_name, outdir, *, fmt="pdf", dpi=200, tables=False,
                   results_root="results"):
    """Write one file per figure to `outdir`, named `{metric}__{scenario}_m{m}.{fmt}`
    (+ `landscape_cssize__{scenario}` per scenario). fmt: pdf (vector, default) / png / svg.
    With `tables=True` also dumps the coverage-size table as `covsize__{scenario}_m{m}.csv`.
    Returns the list of written paths."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    mm = method_meta(sc_name, results_root)
    pip = load_reduction("pip", sc_name, results_root)
    cs = load_reduction("cs", sc_name, results_root)
    cs_pn = load_reduction("cs", sc_name, results_root, include_null=True)
    written = []

    def _save(fig, name):
        p = out / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    cells_ = [(r[0], r[1]) for r in available(sc_name).rows()]
    for scen, m in cells_:
        _save(figure_power_fdp(sc_name, scen, m, pip, mm), f"power_fdp__{scen}_m{m}")
        _save(figure_coverage_size(sc_name, scen, m, cs, mm), f"coverage_size__{scen}_m{m}")
        _save(figure_pip_calibration(sc_name, scen, m, pip, mm), f"calibration__{scen}_m{m}")
        _save(figure_detection_roc(sc_name, scen, m, cs_pn, mm), f"detection_roc__{scen}_m{m}")
        if tables:
            p = out / f"covsize__{scen}_m{m}.csv"
            coverage_size_table(sc_name, scen, m, cs, mm).write_csv(p)
            written.append(str(p))
    for scen in sorted({s for s, _ in cells_}):
        _save(figure_landscape_heatmap(sc_name, cs, mm, scenario=scen, stat="cs_size"),
              f"landscape_cssize__{scen}")
    return written


# ---------------------------------------------------------------------------
# Compact credible-set summary tables. rows = method, cols = regime × size (2-level
# header), one table per (scenario, metric), at the nominal CS level, POOLED over
# strength (each cell = all strength tiers x replicates). Emitted as HTML (real
# 2-level header) via `cs_summary_html`.
# ---------------------------------------------------------------------------
LOC_POWER_SIZE_CAP = 10  # "localized" = 95%-CS with at most this many sets
_cov = lambda v: "-" if v is None else f"{v:.2f}"
_siz = lambda v: "-" if v is None else f"{v:.0f}"
# key -> (title, formatter, direction "max"/"min" for the per-column best).
# "small" = 95%-CS with at most LOC_POWER_SIZE_CAP gene-sets.
_C = LOC_POWER_SIZE_CAP
CS_METRICS = {
    "coverage": ("CS coverage  (P[causal in 95%-CS])", _cov, "max"),
    "median_size": ("median CS size", _siz, "min"),
    "loc_power": (f"small CS power  (P[causal in CS and |CS|<={_C}])", _cov, "max"),
    "localized_coverage": (f"small CS coverage  (P[causal in CS | |CS|<={_C}])", _cov, "max"),
    "small_median_size": (f"small CS median size  (median |CS| where |CS|<={_C})", _siz, "min"),
    # null-simulation metrics (no causal set; use CS size only). "small CS" on a null is a
    # false confident-localization, so a low rate / diffuse size is good.
    "null_small_rate": (f"null small-CS rate  (P[|CS|<={_C} | null])", _cov, "min"),
    "null_median_size": ("null median CS size  (diffuse ⇒ large)", _siz, "max"),
}
NULL_METRICS = {"null_small_rate", "null_median_size"}


def _cs_metric_value(g, inom, metric, size_cap):
    """Pooled CS metric over the replicate rows in `g` at beta index `inom`."""
    mass = np.array([v[0] if len(v) else np.nan for v in g["mass_above_causal"].to_list()])
    sizes = np.asarray(g["cs_sizes"].to_list(), dtype=float)
    if sizes.ndim != 2 or sizes.shape[0] == 0:
        return None
    from utils import CS_BETA_GRID
    nominal = float(np.asarray(CS_BETA_GRID.tolist())[inom])
    covered = mass < nominal
    s = sizes[:, inom]
    small = s <= size_cap
    if metric == "coverage":
        return float(np.mean(covered))
    if metric == "median_size":
        return float(np.median(s))
    if metric == "loc_power":
        return float(np.mean(covered & small))
    if metric == "localized_coverage":
        return None if small.sum() == 0 else float(np.mean(covered[small]))
    if metric in ("small_median_size", "null_small_median_size"):
        return None if small.sum() == 0 else float(np.median(s[small]))
    if metric == "null_small_rate":
        return float(np.mean(small))
    if metric == "null_median_size":
        return float(np.median(s))
    raise ValueError(f"unknown metric {metric!r}")


def cs_summary_frame(sc_name, scenario, cs, mm, *, metric, nominal=0.95, methods=None,
                     size_cap=LOC_POWER_SIZE_CAP, strength=None, is_null=False):
    """DataFrame (method, regime, size, value): one CS `metric` per (method, regime, size).
    `strength=None` pools over all strength tiers; `strength="weak"` (etc.) uses just that
    tier. `is_null=True` uses each cell's paired null_b0 rows (needs a cs frame loaded with
    include_null=True). Default methods = CROSS_METHODS."""
    methods = _present_methods(methods or CROSS_METHODS, mm)
    grid = _cs_beta_grid(); inom = int(np.argmin(np.abs(grid - nominal)))
    cs_cells = [c for c in cells(sc_name) if c["scenario"] == scenario]
    sizes = [s for s in SIZES if any(c["m"] == s for c in cs_cells)]
    regimes = [r for r in REGIMES if any(c["regime"] == r for c in cs_cells)]
    alias_set = {c["alias"] for c in cs_cells}
    use_strengths = [strength] if strength else STRENGTHS
    base = cs.filter(pl.col("l") == 0)
    if "is_null" in cs.columns:
        base = base.filter(pl.col("is_null") == is_null)
    rows = []
    for meth in methods:
        gm = base.filter(pl.col("method") == meth)
        for regime in regimes:
            for size in sizes:
                aliases = [cell_alias(scenario, size, s, regime) for s in use_strengths
                           if cell_alias(scenario, size, s, regime) in alias_set]
                g = gm.filter(pl.col("collection_name").is_in(aliases))
                rows.append({"method": meth, "regime": regime, "size": size,
                             "value": _cs_metric_value(g, inom, metric, size_cap)})
    return pl.DataFrame(rows)


def cs_summary_html(sc_name, scenario, cs, mm, *, metric, nominal=0.95, methods=None,
                    size_cap=LOC_POWER_SIZE_CAP, undercover_margin=0.02, strength=None,
                    is_null=False, shade_high=None):
    """Compact HTML table for one (scenario, metric): rows = method, 2-level column header
    regime × size, cells = the metric. **Best method per column** is bolded. Shading:
    causal tables shade a cell light red when it **under-covers** (coverage < nominal -
    undercover_margin) - flagging attractive size/power that isn't calibrated; null tables
    (`is_null=True`) instead shade when the value **exceeds `shade_high`** (e.g. a null small-CS
    rate over 0.05 = too-frequent false confident localization). Emit via R `cat` in the qmd."""
    title, fmt, direction = CS_METRICS[metric]
    mdf = cs_summary_frame(sc_name, scenario, cs, mm, metric=metric, nominal=nominal,
                           methods=methods, size_cap=size_cap, strength=strength, is_null=is_null)
    shade_cov = (not is_null) and shade_high is None
    cov = {}
    if shade_cov:
        cdf = (mdf if metric == "coverage" else
               cs_summary_frame(sc_name, scenario, cs, mm, metric="coverage", nominal=nominal,
                                methods=methods, size_cap=size_cap, strength=strength))
        cov = {(r["method"], r["regime"], r["size"]): r["value"] for r in cdf.iter_rows(named=True)}
    disp = {r["method"]: r["method_display"] for r in mm.iter_rows(named=True)}
    methods = _present_methods(methods or CROSS_METHODS, mm)
    regimes = [r for r in REGIMES if mdf.filter(pl.col("regime") == r).height]
    sizes = sorted(mdf["size"].unique().to_list())
    val = {(r["method"], r["regime"], r["size"]): r["value"] for r in mdf.iter_rows(named=True)}

    best_fmt = {}  # per-column best (formatted-value match so ties bold together)
    for r in regimes:
        for s in sizes:
            vals = [val[(m, r, s)] for m in methods if val.get((m, r, s)) is not None]
            if vals:
                best_fmt[(r, s)] = fmt(max(vals) if direction == "max" else min(vals))

    th = "border:1px solid #ccc;padding:2px 8px;text-align:center"
    td = "border:1px solid #eee;padding:2px 8px;text-align:right"
    red = "background:#ffdddd"
    scope = ("pooled over strength tiers" if strength is None else f"strength = {strength}") \
        + (" · NULLS" if is_null else "")
    flag = (f'<span style="{red};padding:0 4px">red</span> = value &gt; {shade_high:.2f}'
            if shade_high is not None else
            f'<span style="{red};padding:0 4px">red</span> = under-covers '
            f'(&lt;{nominal - undercover_margin:.2f})')
    h = [f'<p><b>{scenario} - {title}</b> (95% CS; {scope}; <b>bold</b> = best per column, {flag})</p>',
         '<table style="border-collapse:collapse;font-size:90%">', "<thead>",
         f'<tr><th rowspan="2" style="{th}">method</th>'
         + "".join(f'<th colspan="{len(sizes)}" style="{th}">{r}</th>' for r in regimes) + "</tr>",
         "<tr>" + "".join(f'<th style="{th}">m={s}</th>' for _ in regimes for s in sizes) + "</tr>",
         "</thead>", "<tbody>"]
    for meth in methods:
        tds = []
        for r in regimes:
            for s in sizes:
                v = val.get((meth, r, s))
                txt = fmt(v)
                if v is not None and best_fmt.get((r, s)) == txt:
                    txt = f"<b>{txt}</b>"
                if shade_high is not None:
                    shade = v is not None and v > shade_high
                else:
                    c = cov.get((meth, r, s))
                    shade = c is not None and c < nominal - undercover_margin
                tds.append(f'<td style="{td}{";" + red if shade else ""}">{txt}</td>')
        h.append(f'<tr><td style="{td};text-align:left">{disp[meth]}</td>{"".join(tds)}</tr>')
    h += ["</tbody>", "</table>"]
    return "\n".join(h)
