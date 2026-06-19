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


def test_expand_method_cartesian_names_and_kwargs():
    entry = {"function": "run_cox_method", "template": {"time_sign": -1.0},
             "over": {"threshold": [0.0, 2.0], "L": [1, 5]}}
    specs = loader.expand_method("cox_light", entry)
    names = [s.name for s in specs]
    assert names == [
        "cox_light__threshold=0.00__L=1", "cox_light__threshold=0.00__L=5",
        "cox_light__threshold=2.00__L=1", "cox_light__threshold=2.00__L=5",
    ]
    s = specs[2]
    assert s.function.__name__ == "run_cox_method"
    assert s.kwargs == {"time_sign": -1.0, "threshold": 2.0, "L": 1}


def test_expand_method_resolves_distribution_kwargs():
    entry = {"function": "run_twogroup_method",
             "template": {"f1": {"Normal": {"loc": 0.0, "scale": 1.0,
                                            "estimate_loc": True, "estimate_scale": True}}},
             "over": {"L": [1]}}
    specs = loader.expand_method("twogroup", entry)
    assert specs[0].name == "twogroup__L=1"
    from gibss.distributions import Normal
    assert isinstance(specs[0].kwargs["f1"], Normal)


def test_library_methods_expands_all_entries():
    lib = _library_for_tests()
    lib["methods"] = {
        "cox_heavy": {"function": "run_cox_method",
                      "template": {"threshold": None, "time_sign": 1.0}, "over": {"L": [1]}},
        "cox_light": {"function": "run_cox_method", "template": {"time_sign": -1.0},
                      "over": {"threshold": [2.0], "L": [1]}},
    }
    methods = loader.library_methods(lib)
    assert set(methods) == {"cox_heavy__L=1", "cox_light__threshold=2.00__L=1"}


def test_manifest_dict_shape():
    lib = _library_for_tests()
    spec, name = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian")
    method = loader.expand_method("cox_heavy",
        {"function": "run_cox_method", "template": {"threshold": None, "time_sign": 1.0},
         "over": {"L": [1]}})[0]
    manifest = loader.manifest_dict(lib, {name: spec}, {method.name: method})
    assert set(manifest) == {"batches", "method_specs"}
    (batch_hash, batch_node), = manifest["batches"].items()
    assert batch_node["__spec_hash__"] == batch_hash
    assert batch_node["simulation_spec"]["__spec_hash__"]
    assert list(batch_node["replicates"]) == list(range(50))
    (method_hash, method_node), = manifest["method_specs"].items()
    assert method_node["__spec_hash__"] == method_hash
    assert method_node["fields"]["name"] == "cox_heavy__L=1"


def test_expand_collections_within_and_over():
    lib = _library_for_tests()
    lib["signals"]["loc_1.0"] = {"f0": {"PointMass": {"value": 0.0}},
        "f1": {"Normal": {"loc": 1.0, "scale": 0.1, "estimate_loc": False, "estimate_scale": False}}}
    lib["enrichments"]["null_b0"] = {"function": "uniform_single_effect",
                                     "arguments": {"causal_effect": 0.0}, "intercept": -2.0}
    entry = {"template": {"design": "gaussian_p100",
                          "enrichment": ["ser_b2", "null_b0"], "error": "gaussian"},
             "over": {"signal": ["loc_1.0", "loc_2.0"]}}
    colls = loader.expand_collections(lib, "sc", entry)
    assert [c["name"] for c in colls] == ["sc__signal=loc_1.0", "sc__signal=loc_2.0"]
    assert [c["alias"] for c in colls] == ["loc_1.0", "loc_2.0"]
    # within-collection: ser + null pair
    assert {s.name for s in colls[0]["simulations"]} == {
        "gaussian_p100__ser_b2__loc_1.0", "gaussian_p100__null_b0__loc_1.0"}
