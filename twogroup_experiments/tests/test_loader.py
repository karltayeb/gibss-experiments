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
    spec = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian")
    assert spec.name == "gaussian_p100__ser_b2__loc_2.0"
    assert spec.intercept == -2.0
    assert spec.base_seed == 20260501
    assert spec.error_sampler is None
    assert spec.f1.loc == 2.0
    # design/effect samplers are functools.partial of the right callable
    assert spec.design_sampler.func.__name__ == "gaussian_markov_X"
    assert spec.design_sampler.keywords == {"n": 500, "p": 100, "rho": 0.9}


def test_resolve_simulation_nongaussian_error_in_name():
    lib = _library_for_tests()
    spec = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "t_df_5")
    assert spec.name == "gaussian_p100__ser_b2__loc_2.0__t_df_5"
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
    spec = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian")
    method = loader.expand_method("cox_heavy",
        {"function": "run_cox_method", "template": {"threshold": None, "time_sign": 1.0},
         "over": {"L": [1]}})[0]
    manifest = loader.manifest_dict(lib, {spec.name: spec}, {method.name: method})
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


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "experiments"


def test_load_config_and_accessors():
    cfg = loader.load_config(FIXTURE_DIR)
    assert "fixture-sc" in cfg["supercollections"]
    sims = loader.all_simulations(cfg)
    assert {"gaussian_p8__ser_b2__loc_2.0", "gaussian_p8__null_b0__loc_2.0"} <= set(sims)
    methods = loader.all_methods(cfg)
    assert {"cox_heavy__L=1", "twogroup__L=1"} == set(methods)


def test_flatten_analyses_expands_groups_and_dedups():
    cfg = loader.load_config(FIXTURE_DIR)
    flat = loader.flatten_analyses(cfg["library"], ["pip", "pip_calibration"])
    assert flat == ["pip_calibration", "agg_pip_calibration"]


def test_resolve_sc_analyses_pairs():
    cfg = loader.load_config(FIXTURE_DIR)
    pairs = loader.resolve_sc_analyses(cfg, "fixture-sc")
    assert ("pip_calibration", "minimal") in pairs
    assert ("agg_pip_calibration", "minimal") in pairs


def test_reduction_scope_and_paths():
    cfg = loader.load_config(FIXTURE_DIR)
    lib = cfg["library"]
    assert loader.reduction_scope(lib, "pip") == "fit"
    p = loader.reduction_output("BH", "MH", "pip", "fit")
    assert p == "results/by_batch/BH/fits/MH/reductions/pip.parquet"


def test_analysis_inputs_only_required_reductions(tmp_path):
    cfg = loader.load_config(FIXTURE_DIR)
    manifest = loader.manifest_dict(cfg["library"], loader.all_simulations(cfg), loader.all_methods(cfg))
    inputs = loader.analysis_inputs(cfg, manifest, "fixture-sc", "pip_calibration")
    # pip_calibration requires [pip]; every path ends in /reductions/pip.parquet
    assert inputs and all(p.endswith("/reductions/pip.parquet") for p in inputs)


def test_method_metadata_columns():
    cfg = loader.load_config(FIXTURE_DIR)
    methods = loader.all_methods(cfg)
    md = loader.method_metadata(methods)
    assert {"method", "method_family", "L", "threshold", "is_thresholded",
            "is_oracle", "method_display"} <= set(md.columns)
    fams = set(md["method_family"].to_list())
    assert fams == {"cox_heavy", "twogroup"}


def test_load_sc_bundle_tags_collections(tmp_path):
    import polars as pl
    import core, utils
    cfg = loader.load_config(FIXTURE_DIR)
    lib = cfg["library"]
    results = tmp_path / "results"
    # materialize + fit + reduce one collection's units for reduction "pip"
    for coll in loader.supercollection_collections(lib, "fixture-sc", cfg["supercollections"]["fixture-sc"]):
        for spec in coll["simulations"]:
            reps = (0, 1)
            bh = core.dehydrate_hashed(utils.BatchSpec(name=f"{spec.name}__batch0", simulation_spec=spec, replicates=reps))[core.HASH_KEY]
            sims_df = utils.simulate_batch(spec, replicates=reps)
            sample_md = __import__("plot_ready").build_sample_metadata(bh, sims_df)
            for mname, mspec in loader.resolve_methods_for_sc(lib, cfg["supercollections"]["fixture-sc"]).items():
                mh = core.dehydrate_hashed(mspec)[core.HASH_KEY]
                fits = utils.fit_batch_method(spec, method_spec=mspec, replicates=reps).with_columns(pl.lit(bh).alias("batch_hash"))
                red = __import__("plot_ready").build_pip_plot_data(fits, sample_md, sims_df)
                out = results / "by_batch" / bh / "fits" / mh / "reductions" / "pip.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                red.write_parquet(out)
    bundle = loader.load_sc_bundle(cfg, "fixture-sc", ["pip"], results_root=str(results))
    assert "pip_plot_data" in bundle and "method_metadata" in bundle
    assert set(bundle["pip_plot_data"]["collection_name"].unique().to_list()) == {"loc_2.0"}


def test_resolve_simulation_sets_hash_and_fields():
    lib = _library_for_tests()
    spec = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian")
    assert spec.hash == loader.sim_hash(loader.simulation_coordinate(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian"))
    assert spec.design_sampler.func.__name__ == "gaussian_markov_X"
    assert spec.error_sampler is None
    assert spec.name == "gaussian_p100__ser_b2__loc_2.0"


def test_simulation_coordinate_and_hash_stable():
    lib = _library_for_tests()  # existing helper in this file
    c1 = loader.simulation_coordinate(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian")
    assert c1["design"] == lib["designs"]["gaussian_p100"]
    assert c1["enrichment"]["intercept"] == -2.0
    assert c1["error"] is None
    assert c1["base_seed"] == 20260501
    h = loader.sim_hash(c1)
    assert isinstance(h, str) and len(h) == 64
    # stable across calls; differs when a value changes
    assert h == loader.sim_hash(loader.simulation_coordinate(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian"))
    assert h != loader.sim_hash(loader.simulation_coordinate(lib, "gaussian_p100", "ser_b2", "loc_2.0", "t_df_5"))
