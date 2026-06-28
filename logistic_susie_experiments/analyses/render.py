"""Generic Snakemake analyze entrypoint.

Resolves a plot spec by (supercollection, plot_name), loads the bundle for every
reduction the plot consumes (one analysis, or several for a dashboard), and
renders via generate_plots.render_plot. One script for all plot kinds.
"""
import sys
from pathlib import Path

if "snakemake" in globals():
    _parent = str(Path(__file__).parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    import generate_plots
    from experiments import loader as _loader

    _wc = snakemake.wildcards
    _cfg = _loader.load_config()
    _spec = _loader.resolve_plot_spec(_cfg, _wc.supercollection, _wc.plot_name)
    _reductions = _loader.plot_reductions(_cfg, _wc.supercollection, _wc.plot_name)
    _bundle = _loader.load_sc_bundle(_cfg, _wc.supercollection, _reductions)
    generate_plots.render_plot(_bundle, _spec, snakemake.output[0])
