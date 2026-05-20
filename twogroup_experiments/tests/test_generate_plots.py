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
                    "L": 1,
                    "max_fdp": 0.5,
                    "method_families": ["twogroup"],
                }
            }
        },
        "settings": {
            "cox_only": {"method_families": ["cox_heavy"]},
        },
    }

    result = generate_plots._resolve_settings(cfg, "my-sc", "cox_only")

    assert result["threshold"] == 2.0
    assert result["method_families"] == ["cox_heavy"]
    assert result["max_fdp"] == 0.5


def test_resolve_settings_default_only():
    import generate_plots

    cfg = {
        "supercollections": {
            "my-sc": {
                "default_settings": {"threshold": 2.0, "L": 1}
            }
        },
        "settings": {
            "all_methods": {},
        },
    }

    result = generate_plots._resolve_settings(cfg, "my-sc", "all_methods")

    assert result == {"threshold": 2.0, "L": 1}


def test_load_plot_config_has_two_keys():
    import generate_plots

    cfg = generate_plots._load_plot_config()

    assert "supercollections" in cfg
    assert "settings" in cfg


def test_foreground_methods_filters_by_family_and_L():
    import generate_plots

    method_metadata = pl.DataFrame({
        "method": ["twogroup_L1", "cox_heavy_L1", "twogroup_L2"],
        "method_family": ["twogroup", "cox_heavy", "twogroup"],
        "L": [1, 1, 2],
        "threshold": [None, None, None],
        "is_thresholded": [False, False, False],
    })

    settings = {"method_families": ["twogroup"], "L": 1, "threshold": 2.0}
    result = generate_plots._foreground_methods(method_metadata, settings)

    assert result == {"twogroup_L1"}
