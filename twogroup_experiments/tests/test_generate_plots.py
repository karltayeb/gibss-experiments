from __future__ import annotations

import pytest
import polars as pl


def test_resolve_settings_merges_overrides():
    import generate_plots

    cfg = {
        "supercollections": {
            "my-sc": {
                "default_settings": {
                    "threshold": 2.0,
                    "max_fdp": 0.5,
                }
            }
        },
        "settings": {
            "high_threshold": {"threshold": 3.0},
        },
    }

    result = generate_plots._resolve_settings(cfg, "my-sc", "high_threshold")

    assert result["threshold"] == 3.0
    assert result["max_fdp"] == 0.5


def test_resolve_settings_default_only():
    import generate_plots

    cfg = {
        "supercollections": {
            "my-sc": {
                "default_settings": {"threshold": 2.0, "max_fdp": 0.5}
            }
        },
        "settings": {
            "all_methods": {},
        },
    }

    result = generate_plots._resolve_settings(cfg, "my-sc", "all_methods")

    assert result == {"threshold": 2.0, "max_fdp": 0.5}


def test_load_plot_config_has_two_keys():
    import generate_plots

    cfg = generate_plots._load_plot_config()

    assert "supercollections" in cfg
    assert "settings" in cfg


def test_foreground_methods_returns_all_except_wrong_threshold():
    import generate_plots

    method_metadata = pl.DataFrame({
        "method": ["twogroup_L1", "cox_heavy_L1", "logistic_threshold_L1"],
        "method_family": ["twogroup", "cox_heavy", "logistic_threshold"],
        "L": [1, 1, 1],
        "threshold": [None, None, 2.0],
        "is_thresholded": [False, False, True],
    })

    result = generate_plots._foreground_methods(method_metadata, {"thresholds": [2.0]})
    assert result == {"twogroup_L1", "cox_heavy_L1", "logistic_threshold_L1"}

    result_other = generate_plots._foreground_methods(method_metadata, {"thresholds": [3.0]})
    assert result_other == {"twogroup_L1", "cox_heavy_L1"}

    result_multi = generate_plots._foreground_methods(method_metadata, {"thresholds": [2.0, 3.0]})
    assert result_multi == {"twogroup_L1", "cox_heavy_L1", "logistic_threshold_L1"}


def test_foreground_methods_filters_by_method_families():
    import generate_plots

    method_metadata = pl.DataFrame({
        "method": ["twogroup_L1", "cox_heavy_L1", "logistic_threshold_L1"],
        "method_family": ["twogroup", "cox_heavy", "logistic_threshold"],
        "L": [1, 1, 1],
        "threshold": [None, None, 2.0],
        "is_thresholded": [False, False, True],
    })

    result = generate_plots._foreground_methods(
        method_metadata,
        {"thresholds": [2.0], "method_families": ["twogroup", "logistic_threshold"]},
    )
    assert result == {"twogroup_L1", "logistic_threshold_L1"}
