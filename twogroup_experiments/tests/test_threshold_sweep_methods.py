from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import viz_utils
from experiments import loader


def test_new_method_families_color_and_label():
    assert viz_utils.method_color("cox_reversed_censored__threshold=2.00__L=1") == "#E69F00"
    assert viz_utils.method_color("cox_uncensored__L=1") == "#009E73"
    labels = viz_utils.method_family_label_map()
    assert labels["cox_reversed_censored"] == "Cox reversed (censored)"
    assert labels["cox_uncensored"] == "Cox (uncensored)"


def test_cox_uncensored_registered():
    lib = loader.load_library()
    methods = loader.library_methods(lib)
    assert "cox_uncensored__L=1" in methods


def test_003_sweep_method_threshold_split():
    import polars as pl
    cfg = loader.load_config()
    sc = cfg["supercollections"]["003-hallmark-loc-snr"]
    methods = loader.resolve_methods_for_sc(cfg["library"], sc)
    meta = loader.method_metadata(methods)

    line_methods = [
        "cox__threshold=2.00__L=1",
        "cox_reversed_censored__threshold=2.00__L=1",
        "logistic_threshold__threshold=2.00__L=1",
    ]
    horizontal_methods = [
        "cox_uncensored__L=1", "cox_reversed__L=1",
        "twogroup_oracle__L=1", "twogroup__L=1", "twogroup_loc_fam__L=1",
    ]
    for m in line_methods:
        row = meta.filter(pl.col("method") == m)
        assert row.height == 1 and bool(row["is_thresholded"][0]) is True, m
    for m in horizontal_methods:
        row = meta.filter(pl.col("method") == m)
        assert row.height == 1 and bool(row["is_thresholded"][0]) is False, m
