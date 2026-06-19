from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from experiments import loader
from gibss.distributions import Normal, PointMass


def test_format_float():
    assert loader.format_float(2.0) == "2.00"
    assert loader.format_float(0.5) == "0.50"


def test_resolve_distribution_normal_and_pointmass():
    n = loader.resolve_distribution({"Normal": {"loc": 2.0, "scale": 0.1,
                                                 "estimate_loc": False, "estimate_scale": False}})
    assert isinstance(n, Normal) and n.loc == 2.0 and n.scale == 0.1
    p = loader.resolve_distribution({"PointMass": {"value": 0.0}})
    assert isinstance(p, PointMass) and p.value == 0.0


def test_resolve_callable_resolves_core_functions():
    assert loader.resolve_callable("run_cox_method").__name__ == "run_cox_method"
    with pytest.raises(KeyError):
        loader.resolve_callable("does_not_exist")


def _library_for_tests():
    return {
        "defaults": {"base_seed": 20260501, "replicates_per_batch": 50, "n_batches": 1},
        "designs": {"gaussian_p100": {"function": "gaussian_markov_X",
                                      "arguments": {"n": 500, "p": 100, "rho": 0.9}}},
        "enrichments": {"ser_b2": {"function": "uniform_single_effect",
                                   "arguments": {"causal_effect": 2.0}, "intercept": -2.0}},
        "signals": {"loc_2.0": {"f0": {"PointMass": {"value": 0.0}},
                                "f1": {"Normal": {"loc": 2.0, "scale": 0.1,
                                                  "estimate_loc": False, "estimate_scale": False}}}},
        "errors": {"gaussian": None, "t_df_5": {"function": "t_error_sampler", "arguments": {"df": 5}}},
        "methods": {}, "reductions": {}, "analyses": {}, "analysis_groups": {},
    }


def test_resolve_simulation_builds_spec_and_name():
    lib = _library_for_tests()
    spec, name = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian")
    assert name == "gaussian_p100__ser_b2__loc_2.0"
    assert spec.intercept == -2.0
    assert spec.base_seed == 20260501
    assert spec.error_sampler is None
    assert spec.f1.loc == 2.0
    # design/effect samplers are functools.partial of the right callable
    assert spec.design_sampler.func.__name__ == "gaussian_markov_X"
    assert spec.design_sampler.keywords == {"n": 500, "p": 100, "rho": 0.9}


def test_resolve_simulation_nongaussian_error_in_name():
    lib = _library_for_tests()
    spec, name = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "t_df_5")
    assert name == "gaussian_p100__ser_b2__loc_2.0__t_df_5"
    assert spec.error_sampler is not None
    assert spec.error_sampler.keywords == {"df": 5}
