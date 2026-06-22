from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import loader


def test_loader_suffix_aware_for_paired():
    lib = loader.load_library()
    cfg = {"library": lib}
    # family resolves to "paired" by suffix
    assert loader.analysis_family("causal_pip_paired") == "paired"
    # requires + simulation_filter resolve to the base analysis
    assert loader.analysis_requires(cfg, "causal_pip_paired") == loader.analysis_requires(cfg, "causal_pip")
    assert (loader.analysis_simulation_filter(lib, "causal_pip_paired")
            == loader.analysis_simulation_filter(lib, "causal_pip"))
    # flatten accepts a paired name (validates via base)
    assert loader.flatten_analyses(lib, ["causal_pip_paired"]) == ["causal_pip_paired"]


import matplotlib
matplotlib.use("Agg")
import polars as pl


def test_pair_reciprocal_transform():
    from analyses.paired import pair_reciprocal
    pip_plot = pl.DataFrame(
        {
            "collection_name": ["lambda=2/3", "lambda=3/2"],
            "method": ["cox_reversed__L=1", "cox_reversed__L=1"],
            "threshold": [None, None],
            "causal_pips": [[0.3, 0.4], [0.6, 0.7]],
        },
        schema_overrides={"threshold": pl.Float64},
    )
    bundle = {"pip_plot_data": pip_plot, "method_metadata": pl.DataFrame(),
              "collection_names": ["lambda=2/3", "lambda=3/2"]}
    out = pair_reciprocal(bundle)
    df = out["pip_plot_data"].sort("method")
    # reciprocals collapse to one pair label
    assert df["collection_name"].n_unique() == 1
    # sign assignment
    assert df["method"].to_list() == ["depletion", "enrichment"]
    # method_metadata rebuilt with the two sign rows
    assert set(out["method_metadata"]["method"].to_list()) == {"depletion", "enrichment"}
    assert len(out["collection_names"]) == 1  # 2/3 and 3/2 collapse to one pair


def test_paired_renderer_registered_and_renders():
    import generate_plots
    assert "causal_pip_paired" in generate_plots.ANALYSIS_RENDERERS
    pip_plot = pl.DataFrame(
        {
            "collection_name": ["lambda=2/3", "lambda=3/2"],
            "method": ["cox_reversed__L=1", "cox_reversed__L=1"],
            "threshold": [None, None],
            "causal_pips": [[0.3, 0.4], [0.6, 0.7]],
        },
        schema_overrides={"threshold": pl.Float64},
    )
    bundle = {"pip_plot_data": pip_plot, "method_metadata": pl.DataFrame(),
              "collection_names": ["lambda=2/3", "lambda=3/2"]}
    fig = generate_plots.ANALYSIS_RENDERERS["causal_pip_paired"](bundle, {})
    labels = set(fig.axes[0].get_legend_handles_labels()[1])
    assert labels and labels <= {"Depletion", "Enrichment"}


def test_009_uses_paired_analyses():
    cfg = loader.load_config()
    pairs = loader.resolve_sc_analyses(cfg, "009-hallmark-cox-well-specified")
    analyses = {a for a, _ in pairs}
    assert analyses, "009 resolved no analyses"
    assert all(a.endswith("_paired") for a in analyses)
    assert "causal_pip_paired" in analyses
    assert "agg_causal_pip_paired" in analyses
