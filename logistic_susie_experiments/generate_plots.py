from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
_parent = str(Path(__file__).parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import plot_ready
import viz_utils
from analyses._common import foreground_methods as _foreground_methods_impl
from analyses.hooks import HOOKS
from viz_facet import apply_filter, assign_groups

_LINESTYLES = ["-", "--", ":", "-."]


def _foreground_methods(method_metadata: pl.DataFrame, settings: dict) -> set[str]:
    """Filter method_filter to methods present in method_metadata."""
    return _foreground_methods_impl(method_metadata, settings)


def _method_order(method_metadata: pl.DataFrame, foreground: set[str]) -> list[str]:
    from analyses._common import method_order
    return method_order(method_metadata, foreground)


def _shade(hex_color, frac: float):
    """Blend a hex color toward white. frac=0 -> original, frac=1 -> white.
    Used so the linestyle dim (e.g. step) reads as light/dark of the color dim's
    hue (e.g. family) — distinguishable on marker-only plots where linestyle
    is invisible."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    f = max(0.0, min(1.0, frac))
    return (r + (1.0 - r) * f, g + (1.0 - g) * f, b + (1.0 - b) * f)


def _label_for(color_key, ls_key) -> str:
    return " · ".join(str(x) for x in (color_key, ls_key) if x is not None)


def _distinct(rows, dim):
    if dim is None:
        return [None]
    return sorted({r.get(dim) for r in rows}, key=str)


def _method_label(row) -> str:
    """Readable per-method panel title from the method dims on a row."""
    fam, step, prior = row.get("family"), row.get("step"), row.get("prior")
    ctr = "local" if row.get("center") else "global"
    return f"{fam} · {step}\n{prior} · {ctr}"


def _aesthetics(rows, spec):
    """Shared color/linestyle maps. color dim -> hue; linestyle dim -> light/dark
    SHADE of that hue. Lines stay solid: hue + shade already separate the series
    (e.g. family + step), so a dash pattern is redundant."""
    color_vals = _distinct(rows, spec.get("color"))
    ls_vals = _distinct(rows, spec.get("linestyle"))
    base = viz_utils.dim_palette(color_vals)
    ls_dash = {v: "-" for v in ls_vals}  # solid; step is encoded by shade, not dashes
    n_ls = max(1, len(ls_vals))
    ls_shade = {v: (0.0 if n_ls == 1 else 0.55 * i / (n_ls - 1)) for i, v in enumerate(ls_vals)}

    def series_color(s):
        return _shade(base[s.color_key], ls_shade[s.linestyle_key])

    return base, ls_dash, ls_shade, series_color


def _draw_cell(ax, cell, hook, series_color, ls_dash, seen):
    # expose series position so dodge-style draws (e.g. operating-points) can
    # offset each method within a categorical x-group; other draws ignore it.
    ax._n_series = len(cell)
    for i, s in enumerate(cell):
        ax._series_index = i
        stats = hook.aggregate(s.rows)
        hook.draw(ax, stats, color=series_color(s), linestyle=ls_dash[s.linestyle_key],
                  label=_label_for(s.color_key, s.linestyle_key) or None)
        seen.setdefault((s.color_key, s.linestyle_key), s)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def _add_legend(fig, seen, base, ls_dash, ls_shade, spec):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=_shade(base[ck], ls_shade[lk]),
               linestyle=ls_dash[lk], marker="o", markersize=4, linewidth=1.6)
        for (ck, lk) in seen
    ]
    labels = [_label_for(ck, lk) for (ck, lk) in seen]
    if any(labels):
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   fontsize=8, frameon=False, title=_legend_title(spec))


def render_plot(bundle: dict, spec: dict, output_path: str) -> None:
    """Generic renderer: filtering, faceting, aesthetics owned here; geometry
    delegated to the analysis hook. Two layouts:

    - facet_wrap: one panel per value of a single dim (e.g. ``method``), wrapped
      into an ``ncol`` grid — for per-method plots like calibration where
      overlaying methods is illegible.
    - facet_row x facet_col grid (default): methods overlaid per cell, separated
      by color (hue) + linestyle (dash + light/dark shade).
    """
    layout = spec.get("layout")
    if layout:
        _LAYOUTS[layout](bundle, spec, output_path)
        return

    seen: dict = {}
    if spec.get("panels") or spec.get("columns"):
        # dashboard: one panel per analysis (spans reductions); aesthetics shared
        analyses = list(spec.get("panels") or [])
        for col in (spec.get("columns") or []):
            analyses += list(col)
        if spec.get("bottom_panels"):
            analyses += spec["bottom_panels"]
        if spec.get("wrap_panel"):
            analyses.append(spec["wrap_panel"])
        all_rows = []
        for analysis in analyses:
            df = bundle[f"{HOOKS[analysis].requires}_plot_data"]
            rws = df.to_dicts() if hasattr(df, "to_dicts") else list(df)
            all_rows.extend(apply_filter(rws, spec.get("filter")))
        base, ls_dash, ls_shade, series_color = _aesthetics(all_rows, spec)
        _render_dashboard(bundle, spec, series_color, ls_dash, seen)
    else:
        hook = HOOKS[spec["analysis"]]
        df = bundle[f"{hook.requires}_plot_data"]
        rows = df.to_dicts() if hasattr(df, "to_dicts") else list(df)
        rows = apply_filter(rows, spec.get("filter"))
        base, ls_dash, ls_shade, series_color = _aesthetics(rows, spec)
        if spec.get("facet_wrap"):
            _render_wrapped(rows, spec, hook, series_color, ls_dash, seen)
        else:
            _render_grid(rows, spec, hook, series_color, ls_dash, seen)

    fig = plt.gcf()
    _add_legend(fig, seen, base, ls_dash, ls_shade, spec)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _panel_rows(bundle, analysis, spec):
    df = bundle[f"{HOOKS[analysis].requires}_plot_data"]
    rows = df.to_dicts() if hasattr(df, "to_dicts") else list(df)
    return apply_filter(rows, spec.get("filter"))


def _cell_for(bundle, analysis, spec):
    hook = HOOKS[analysis]
    rows = _panel_rows(bundle, analysis, spec)
    g = assign_groups(rows, color=spec.get("color"), linestyle=spec.get("linestyle"))
    return hook, [s for c in g.cells.values() for s in c]


def _draw_overlaid(ax, analysis, rows, spec, series_color, ls_dash, seen):
    hook = HOOKS[analysis]
    g = assign_groups(rows, color=spec.get("color"), linestyle=spec.get("linestyle"))
    cell = [s for c in g.cells.values() for s in c]
    _draw_cell(ax, cell, hook, series_color, ls_dash, seen)
    ax.set_title(analysis, fontsize=10)


def _draw_top(ax, analysis, bundle, spec, series_color, ls_dash, seen, inset_map, title_size=9):
    """One overlaid analysis panel: square, titled, with an optional zoom inset."""
    hook, cell = _cell_for(bundle, analysis, spec)
    _draw_cell(ax, cell, hook, series_color, ls_dash, seen)
    ax.set_title(analysis, fontsize=title_size)
    ax.set_box_aspect(1)
    if analysis in inset_map:
        _add_zoom_inset(ax, hook, cell, series_color, ls_dash, float(inset_map[analysis]))


def _add_zoom_inset(ax, hook, cell, series_color, ls_dash, lo):
    """Inset re-drawing the same cell zoomed to [lo,1]^2 — for calibration, where
    the action lives near the top-right corner."""
    inset = ax.inset_axes([0.56, 0.10, 0.40, 0.40])
    _draw_cell(inset, cell, hook, series_color, ls_dash, {})
    inset.set_xlim(lo, 1.0)
    inset.set_ylim(lo, 1.0)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.set_title("")
    inset.tick_params(labelsize=6)
    try:
        ax.indicate_inset_zoom(inset, edgecolor="0.5")
    except Exception:
        pass


def _render_columns(bundle, spec, series_color, ls_dash, seen, inset_map) -> None:
    """Column-major ragged grid: spec['columns'] is a list of columns, each a list
    of analyses (top->bottom). Panel (col c, row r) -> axes[r][c]; short columns
    leave the lower cells blank."""
    columns = spec["columns"]
    ncols = len(columns)
    nrows = max(len(c) for c in columns)
    side = 3.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(side * ncols, side * nrows),
                             squeeze=False, constrained_layout=True)
    for c, col in enumerate(columns):
        for r in range(nrows):
            ax = axes[r][c]
            if r < len(col):
                _draw_top(ax, col[r], bundle, spec, series_color, ls_dash, seen, inset_map)
            else:
                ax.set_visible(False)


def _render_dashboard(bundle, spec, series_color, ls_dash, seen) -> None:
    """Top row: overlaid square panels (spec['panels']). Bottom row (if any):
    overlaid square panels (spec['bottom_panels'], optionally with a zoom inset
    via spec['inset']) followed by per-method square panels for spec['wrap_panel']
    — left-justified with tight spacing. cs_* hooks self-filter to causal, so a
    shared filter (no signal:) keeps nulls for pip/power-fdp."""
    import math

    inset_map = spec.get("inset") or {}

    if spec.get("columns"):
        _render_columns(bundle, spec, series_color, ls_dash, seen, inset_map)
        return

    panels = spec["panels"]
    bottom_panels = spec.get("bottom_panels") or []
    wrap = spec.get("wrap_panel")

    has_bottom = bool(bottom_panels or wrap)
    if not has_bottom:
        ncol = int(spec.get("ncol") or len(panels))
        n = len(panels)
        nrow = max(1, math.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow),
                                 squeeze=False, constrained_layout=True)
        flat = [ax for row in axes for ax in row]
        for idx, analysis in enumerate(panels):
            _draw_top(flat[idx], analysis, bundle, spec, series_color, ls_dash, seen, inset_map)
        for ax in flat[n:]:
            ax.set_visible(False)
        return

    wrap_rows = _panel_rows(bundle, wrap, spec) if wrap else []
    methods = _distinct(wrap_rows, "method") if wrap else []
    n_top = len(panels)
    n_bot = len(bottom_panels) + len(methods)

    top_side, bot_side = 3.3, 2.3
    # figure width spans whichever row is wider; both rows left-justified (fill
    # left cells of a grid sized to the width, hide the rest) so squares keep a
    # consistent size regardless of how many panels each row has.
    width = max(n_top * top_side, n_bot * bot_side)
    ncol_top = max(n_top, int(round(width / top_side)))
    ncol_bot = max(n_bot, int(round(width / bot_side)))
    fig = plt.figure(figsize=(width, top_side + bot_side + 0.9), constrained_layout=True)
    sub_top, sub_bot = fig.subfigures(2, 1, height_ratios=[top_side, bot_side])

    top_axes = sub_top.subplots(1, ncol_top, squeeze=False)[0]
    for j, analysis in enumerate(panels):
        _draw_top(top_axes[j], analysis, bundle, spec, series_color, ls_dash, seen, inset_map)
    for ax in top_axes[n_top:]:
        ax.set_visible(False)

    bot_axes = sub_bot.subplots(1, ncol_bot, squeeze=False)[0]
    j = 0
    for analysis in bottom_panels:  # overlaid panels (+ optional inset) in bottom row
        _draw_top(bot_axes[j], analysis, bundle, spec, series_color, ls_dash, seen, inset_map, title_size=8)
        j += 1
    wrap_hook = HOOKS[wrap] if wrap else None
    for method in methods:  # per-method calibration squares
        ax = bot_axes[j]
        sub = [r for r in wrap_rows if r.get("method") == method]
        g = assign_groups(sub, color=spec.get("color"), linestyle=spec.get("linestyle"))
        cell = [s for c in g.cells.values() for s in c]
        _draw_cell(ax, cell, wrap_hook, series_color, ls_dash, seen)
        ax.set_box_aspect(1)
        ax.set_title(_method_label(sub[0]) if sub else str(method), fontsize=7)
        j += 1
    for ax in bot_axes[n_bot:]:
        ax.set_visible(False)


def _render_grid(rows, spec, hook, series_color, ls_dash, seen) -> None:
    facet_row, facet_col = spec.get("facet_row"), spec.get("facet_col")
    grid = assign_groups(rows, facet_row=facet_row, facet_col=facet_col,
                         color=spec.get("color"), linestyle=spec.get("linestyle"))
    nr, nc = max(1, len(grid.row_keys)), max(1, len(grid.col_keys))
    fig, axes = plt.subplots(nr, nc, figsize=(3.8 * nc, 3.2 * nr),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    for i, rk in enumerate(grid.row_keys):
        for j, ck in enumerate(grid.col_keys):
            ax = axes[i][j]
            _draw_cell(ax, grid.cell(rk, ck), hook, series_color, ls_dash, seen)
            if i != nr - 1:
                ax.set_xlabel("")
            if j != 0:
                ax.set_ylabel("")
            if facet_col is not None and i == 0:
                ax.set_title(f"{facet_col} = {ck}", fontsize=9)
            if facet_row is not None and j == 0:
                ax.annotate(f"{facet_row} = {rk}", xy=(0, 0.5), xytext=(-42, 0),
                            xycoords="axes fraction", textcoords="offset points",
                            rotation=90, va="center", ha="center", fontsize=9,
                            fontweight="bold")


def _render_wrapped(rows, spec, hook, series_color, ls_dash, seen) -> None:
    import math

    wrap = spec["facet_wrap"]
    ncol = int(spec.get("ncol") or 4)
    color, linestyle = spec.get("color"), spec.get("linestyle")
    vals = _distinct(rows, wrap)
    n = len(vals)
    nrow = max(1, math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    flat = [ax for row in axes for ax in row]
    for idx, val in enumerate(vals):
        ax = flat[idx]
        sub = [r for r in rows if r.get(wrap) == val]
        cell = []
        g = assign_groups(sub, color=color, linestyle=linestyle)
        for c in g.cells.values():
            cell.extend(c)
        _draw_cell(ax, cell, hook, series_color, ls_dash, seen)
        title = _method_label(sub[0]) if (wrap == "method" and sub) else str(val)
        ax.set_title(title, fontsize=8)
        if idx % ncol != 0:
            ax.set_ylabel("")
        if idx < n - ncol:  # not in the last populated row
            ax.set_xlabel("")
    for ax in flat[n:]:  # hide unused panels
        ax.set_visible(False)


def _legend_title(spec: dict) -> str:
    parts = [spec.get("color"), spec.get("linestyle")]
    return " · ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Bespoke composite layouts (hand-built gridspecs). Register one per name; the
# plot spec selects it with `layout: <name>`. More flexible than a generic grid
# DSL for one-off figures — adjust the function directly.
# ---------------------------------------------------------------------------
# analyses each layout consumes (loader -> reductions via HOOKS[a].requires)
LAYOUT_ANALYSES = {
    "cs_summary": ["cs_calibration", "cs_power_fdp", "cs_roc",
                   "cs_size_power_nom", "cs_size_power_cal"],
}


def _layout_cs_summary(bundle, spec, output_path) -> None:
    """Headline = two (CS size, power) tradeoff scatters — nominal and calibrated
    side by side (one point + IQR whisker per method, no connectors; compare
    across the two). Calibration + detection stacked at left:
        col1 (small):  calibration | power_fdp | roc
        col2 (large):  size-power @ NOMINAL
        col3 (large):  size-power @ CALIBRATED
    """
    all_rows = []
    for a in LAYOUT_ANALYSES["cs_summary"]:
        df = bundle[f"{HOOKS[a].requires}_plot_data"]
        rws = df.to_dicts() if hasattr(df, "to_dicts") else list(df)
        all_rows.extend(apply_filter(rws, spec.get("filter")))
    base, ls_dash, ls_shade, series_color = _aesthetics(all_rows, spec)
    seen: dict = {}
    inset_map = spec.get("inset") or {}

    fig = plt.figure(figsize=(11.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.5, 1.5])
    for r, a in enumerate(("cs_calibration", "cs_power_fdp", "cs_roc")):
        _draw_top(fig.add_subplot(gs[r, 0]), a, bundle, spec, series_color,
                  ls_dash, seen, inset_map)
    ax_nom = fig.add_subplot(gs[:, 1])
    _draw_top(ax_nom, "cs_size_power_nom", bundle, spec, series_color, ls_dash, seen, inset_map)
    ax_nom.set_title("CS size vs power — nominal", fontsize=10)
    ax_cal = fig.add_subplot(gs[:, 2], sharey=ax_nom)
    _draw_top(ax_cal, "cs_size_power_cal", bundle, spec, series_color, ls_dash, seen, inset_map)
    ax_cal.set_title("CS size vs power — calibrated (95%)", fontsize=10)

    _add_legend(fig, seen, base, ls_dash, ls_shade, spec)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _layout_thematic(bundle, spec, output_path) -> None:
    """Generic thematic summary: analyses (spec['rows']) as ROWS, designs as
    COLUMNS. Each cell overlays all methods (color dim), pooled over every other
    dim under spec['filter'] (e.g. rho x logbf at a fixed b0). Design columns are
    derived from the data unless spec['designs'] is given."""
    rows = list(spec["rows"])
    all_rows: list = []
    for a in rows:
        all_rows.extend(_panel_rows(bundle, a, spec))
    designs = spec.get("designs") or sorted(
        {r.get("design") for r in _panel_rows(bundle, rows[0], spec) if r.get("design") is not None}
    )
    base, ls_dash, ls_shade, series_color = _aesthetics(all_rows, spec)
    seen: dict = {}
    base_filter = spec.get("filter") or {}
    nr, nc = len(rows), max(len(designs), 1)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 3.0 * nr),
                             squeeze=False, constrained_layout=True)
    for i, analysis in enumerate(rows):
        for j, design in enumerate(designs):
            ax = axes[i][j]
            cell_spec = {**spec, "filter": {**base_filter, "design": design}}
            hook, cell = _cell_for(bundle, analysis, cell_spec)
            _draw_cell(ax, cell, hook, series_color, ls_dash, seen)
            ax.set_box_aspect(1)
            if i == 0:
                ax.set_title(str(design), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{analysis}\n{ax.get_ylabel()}", fontsize=8)
            else:
                ax.set_ylabel("")
    _add_legend(fig, seen, base, ls_dash, ls_shade, spec)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


_LAYOUTS = {"cs_summary": _layout_cs_summary, "thematic": _layout_thematic}
