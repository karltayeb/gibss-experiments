from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import polars as pl
import generate_plots


def test_foreground_methods_is_name_membership():
    meta = pl.from_dicts([{"method": "twogroup__L=1"}, {"method": "cox_heavy__L=1"}])
    fg = generate_plots._foreground_methods(meta, {"method_filter": ["twogroup__L=1", "absent__L=1"]})
    assert fg == {"twogroup__L=1"}


def test_analysis_registry_has_pip_calibration():
    assert "pip_calibration" in generate_plots.ANALYSIS_RENDERERS
    assert "agg_pip_calibration" in generate_plots.ANALYSIS_RENDERERS
