from __future__ import annotations

import polars as pl
import numpy as np
import json
from twogroup_experiments import viz2_logic, viz2_metadata

def mock_method_spec(name: str, L: int = 1) -> str:
    return json.dumps({
        "fields": {
            "name": name,
            "kwargs": {"L": L}
        }
    })

def test_pip_calibration_summary_generation():
    # Create mock data
    method_spec = mock_method_spec("logistic_threshold_L1", L=1)
    data = pl.DataFrame({
        "replicate": [0, 0, 1, 1],
        "method": ["logistic_threshold_L1"] * 4,
        "threshold": [1.0] * 4,
        "method_spec": [method_spec] * 4,
        "simulation_spec": ["sim"] * 4,
        "pip_threshold": [0.1, 0.2, 0.5, 0.6],
        "n_exact": [10, 10, 10, 10],
        "n_causal_exact": [1, 2, 5, 6],
        "batch_hash": ["batch1"] * 4,
        "batch_name": ["batch1"] * 4,
        "simulation_name": ["sim1"] * 4,
    })
    
    # Add metadata columns
    data_with_meta = viz2_logic.add_plot_metadata_columns(data)
    assert "method_family" in data_with_meta.columns
    assert "L" in data_with_meta.columns
    assert "series_label" not in data_with_meta.columns # add_method_display_labels adds this
    
    # Add display labels
    data_with_labels = viz2_logic.add_method_display_labels(data_with_meta, selected_threshold=1.0)
    assert "series_label" in data_with_labels.columns
    
    # Generate summary
    summary = viz2_logic.make_pip_calibration_summary(data_with_labels)
    assert not summary.is_empty()
    assert "empirical_rate" in summary.columns
    
    # Bootstrap summary
    boot_summary = viz2_logic.summarize_calibration_with_bootstrap(
        summary, group_cols=["method_display", "series_label"]
    )
    assert "ci_lower" in boot_summary.columns
    assert "ci_upper" in boot_summary.columns

def test_power_fdp_summary_generation():
    method_spec = mock_method_spec("twogroup_L1", L=1)
    data = pl.DataFrame({
        "simulation_name": ["sim1"] * 5,
        "method": ["twogroup_L1"] * 5,
        "threshold": [None] * 5,
        "method_spec": [method_spec] * 5,
        "pip_threshold": [0.1, 0.2, 0.3, 0.4, 0.5],
        "power": [0.5, 0.6, 0.7, 0.8, 0.9],
        "fdp": [0.05, 0.1, 0.15, 0.2, 0.25],
        "is_thresholded": [False] * 5,
        "method_label_base": ["Twogroup SER"] * 5,
        "method_display": ["Twogroup SER"] * 5,
    })
    
    # Prep for plot
    prepared = viz2_logic.prepare_power_fdp_plot_data_frame(
        data,
        selected_threshold=2.0,
        selected_methods={"twogroup_L1"},
        show_background_threshold_traces=False
    )
    assert "trace_label" in prepared.columns
    
    summary = viz2_logic.make_power_fdp_summary(prepared)
    assert not summary.is_empty()
    assert "power" in summary.columns
    assert "fdp" in summary.columns

def test_empty_dataframes_dont_crash():
    # Test that logic handles empty inputs gracefully
    empty_pip = viz2_logic.empty_pip_threshold_plot_data()
    summary = viz2_logic.make_pip_calibration_summary(empty_pip)
    assert summary.is_empty()
    
    boot_summary = viz2_logic.summarize_calibration_with_bootstrap(summary, group_cols=["method"])
    assert boot_summary.is_empty()

def test_causal_pip_summary():
    method_spec = mock_method_spec("logistic_threshold_L1", L=1)
    data = pl.DataFrame({
        "simulation_name": ["sim1", "sim1"],
        "method": ["logistic_threshold_L1", "logistic_threshold_L1"],
        "threshold": [1.0, 2.0],
        "method_spec": [method_spec, method_spec],
        "causal_pip": [0.8, 0.9],
    })
    summary = viz2_logic.make_causal_pip_summary(data)
    assert len(summary) == 2
    assert "mean_causal_pip" in summary.columns

def test_cs_summaries():
    # Just verify these can be called without crash for now
    comp_empty = viz2_logic.empty_cs_component_plot_data()
    truth_empty = viz2_logic.empty_cs_truth_plot_data()
    
    agg, by_sim = viz2_logic.make_conditional_cs_summary(
        comp_empty, truth_empty, nominal_coverage=0.95, max_cs_size=10, min_ser_log_bf=2.0
    )
    assert agg.is_empty()
    assert by_sim.is_empty()
