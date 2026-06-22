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
