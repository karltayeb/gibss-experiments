import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import base64
    import marimo as mo
    from pathlib import Path

    PLOTS_ROOT = Path(__file__).parent.parent / "results" / "plots"

    PLOT_TYPES = [
        "pip_calibration",
        "power_fdp",
        "causal_pip",
        "causal_rank",
        "mass_above_causal",
        "cs_dot_summary",
        "cs_power_fdp",
        "cs_beta_trace",
    ]
    AGG_PLOT_TYPES = [
        "agg_pip_calibration",
        "agg_power_fdp",
        "agg_causal_pip",
        "agg_causal_rank",
        "agg_mass_above_causal",
        "agg_cs_power_fdp",
        "agg_cs_beta_trace",
    ]


@app.cell(hide_code=True)
def title_cell():
    mo.md("# Plot Explorer")
    return


@app.cell
def supercollection_cell():
    _sc_names = (
        sorted(p.name for p in PLOTS_ROOT.iterdir() if p.is_dir())
        if PLOTS_ROOT.exists()
        else []
    )
    supercollection_dropdown = mo.ui.dropdown(
        options=_sc_names,
        value=_sc_names[0] if _sc_names else None,
        label="Supercollection",
    )
    supercollection_dropdown
    return (supercollection_dropdown,)


@app.cell
def plot_settings_cell(supercollection_dropdown):
    _sc = supercollection_dropdown.value
    if _sc:
        _ps_names = sorted(
            p.name for p in (PLOTS_ROOT / _sc).iterdir() if p.is_dir()
        )
    else:
        _ps_names = []
    plot_settings_dropdown = mo.ui.dropdown(
        options=_ps_names,
        value=_ps_names[0] if _ps_names else None,
        label="Plot settings",
    )
    plot_settings_dropdown
    return (plot_settings_dropdown,)


@app.cell
def plots_cell(supercollection_dropdown, plot_settings_dropdown):
    _sc = supercollection_dropdown.value
    _ps = plot_settings_dropdown.value

    mo.stop(not _sc or not _ps, mo.md("Select supercollection and plot settings above."))

    _plot_dir = PLOTS_ROOT / _sc / _ps

    def _panel(plot_type: str):
        path = _plot_dir / f"{plot_type}.pdf"
        if not path.exists():
            return mo.md(f"**{plot_type}**\n\n_not found_")
        _data = base64.b64encode(path.read_bytes()).decode()
        _src = f"data:application/pdf;base64,{_data}"
        return mo.vstack([
            mo.md(f"**{plot_type}**"),
            mo.pdf(src=_src, width="100%", height="500px"),
        ])

    mo.vstack([
        mo.md("## Faceted Plots"),
        mo.hstack([_panel(pt) for pt in PLOT_TYPES[:4]], justify="start"),
        mo.hstack([_panel(pt) for pt in PLOT_TYPES[4:]], justify="start"),
        mo.md("## Aggregate Plots"),
        mo.hstack([_panel(pt) for pt in AGG_PLOT_TYPES[:4]], justify="start"),
        mo.hstack([_panel(pt) for pt in AGG_PLOT_TYPES[4:]], justify="start"),
    ])
    return
