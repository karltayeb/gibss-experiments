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
