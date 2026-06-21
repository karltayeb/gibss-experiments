from __future__ import annotations
import inspect
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import polars as pl
import generate_plots
import analyses.pip
import analyses.cs
import analyses.logbf
import analyses.f1


def test_foreground_methods_is_name_membership():
    meta = pl.from_dicts([{"method": "twogroup__L=1"}, {"method": "cox_reversed__L=1"}])
    fg = generate_plots._foreground_methods(meta, {"method_filter": ["twogroup__L=1", "absent__L=1"]})
    assert fg == {"twogroup__L=1"}


def test_analysis_registry_has_pip_calibration():
    assert "pip_calibration" in generate_plots.ANALYSIS_RENDERERS
    assert "agg_pip_calibration" in generate_plots.ANALYSIS_RENDERERS


def test_renderers_resolve_to_family_modules():
    """Each renderer should resolve (via inspect.getfile) to its family module."""
    family_map = {
        "pip_calibration": analyses.pip,
        "agg_pip_calibration": analyses.pip,
        "power_fdp": analyses.pip,
        "agg_power_fdp": analyses.pip,
        "causal_pip": analyses.pip,
        "agg_causal_pip": analyses.pip,
        "mass_above_causal": analyses.pip,
        "agg_mass_above_causal": analyses.pip,
        "causal_rank": analyses.cs,
        "cs_calibration": analyses.cs,
        "log_bf_roc": analyses.logbf,
        "agg_log_bf_roc": analyses.logbf,
        "log_bf_ser_ecdf": analyses.logbf,
        "agg_log_bf_ser_ecdf": analyses.logbf,
        "f1_boxplot": analyses.f1,
        "f1_scatter": analyses.f1,
        "f1_enrich_scatter": analyses.f1,
    }
    for key, expected_module in family_map.items():
        renderer = generate_plots.ANALYSIS_RENDERERS[key]
        renderer_file = Path(inspect.getfile(renderer))
        expected_file = Path(inspect.getfile(expected_module))
        assert renderer_file == expected_file, (
            f"{key}: expected {expected_file}, got {renderer_file}"
        )
