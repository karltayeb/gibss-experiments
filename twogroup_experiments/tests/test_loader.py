from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from experiments import loader
from gibss.distributions import Normal, PointMass


def test_sampler_getfile_points_at_submodule():
    import inspect, core
    assert inspect.getfile(core.gaussian_markov_X).endswith("simulations/design/markov.py")
    assert inspect.getfile(core.uniform_single_effect).endswith("simulations/effect/effects.py")


def test_format_float():
    assert loader.format_float(2.0) == "2.00"
    assert loader.format_float(0.5) == "0.50"


def test_resolve_distribution_normal_and_pointmass():
    n = loader.resolve_distribution({"Normal": {"loc": 2.0, "scale": 0.1,
                                                 "estimate_loc": False, "estimate_scale": False}})
    assert isinstance(n, Normal) and n.loc == 2.0 and n.scale == 0.1
    p = loader.resolve_distribution({"PointMass": {"value": 0.0}})
    assert isinstance(p, PointMass) and p.value == 0.0


def test_resolve_distribution_exponential():
    from simulations.distributions import Exponential

    e = loader.resolve_distribution({"Exponential": {"rate": 2.0}})

    assert isinstance(e, Exponential)
    assert e.rate == 2.0


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
    coords = loader.expand_method("cox", entry)
    names = [c["name"] for c in coords]
    assert names == [
        "cox__threshold=0.00__L=1", "cox__threshold=0.00__L=5",
        "cox__threshold=2.00__L=1", "cox__threshold=2.00__L=5",
    ]
    c = coords[2]
    assert c["function"] == "run_cox_method"
    assert c["kwargs"] == {"time_sign": -1.0, "threshold": 2.0, "L": 1}


def test_expand_method_preserves_distribution_node_in_kwargs():
    entry = {"function": "run_twogroup_method",
             "template": {"f1": {"Normal": {"loc": 0.0, "scale": 1.0,
                                            "estimate_loc": True, "estimate_scale": True}}},
             "over": {"L": [1]}}
    coords = loader.expand_method("twogroup", entry)
    assert coords[0]["name"] == "twogroup__L=1"
    # kwargs keeps raw distribution node (NOT yet resolved); resolve_method resolves it
    assert isinstance(coords[0]["kwargs"]["f1"], dict)
    assert "Normal" in coords[0]["kwargs"]["f1"]


def test_library_methods_expands_all_entries():
    lib = _library_for_tests()
    lib["methods"] = {
        "cox_reversed": {"function": "run_cox_method",
                      "template": {"threshold": None, "time_sign": 1.0}, "over": {"L": [1]}},
        "cox": {"function": "run_cox_method", "template": {"time_sign": -1.0},
                      "over": {"threshold": [2.0], "L": [1]}},
    }
    methods = loader.library_methods(lib)
    assert set(methods) == {"cox_reversed__L=1", "cox__threshold=2.00__L=1"}


def test_manifest_dict_shape():
    """Deprecated shape test retained for reference; new shape is tested by test_manifest_dict_coordinate_shape."""
    cfg = loader.load_config(FIXTURE_DIR)
    manifest = loader.manifest_dict(cfg["library"], cfg)
    assert set(manifest) == {"batches", "methods"}
    (batch_hash, batch_node), = list(manifest["batches"].items())[:1]
    assert batch_node["hash"] == batch_hash
    assert "coordinate" in batch_node
    assert "replicates" in batch_node
    (mhash, method_node), = list(manifest["methods"].items())[:1]
    assert method_node["hash"] == mhash
    assert "name" in method_node


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
    assert {"cox_reversed__L=1", "twogroup__L=1"} == set(methods)


def test_flatten_analyses_expands_groups_and_dedups():
    cfg = loader.load_config(FIXTURE_DIR)
    flat = loader.flatten_analyses(cfg["library"], ["pip", "pip_calibration"])
    assert flat == ["pip_calibration", "agg_pip_calibration"]


def test_resolve_sc_analyses_pairs():
    cfg = loader.load_config(FIXTURE_DIR)
    pairs = loader.resolve_sc_analyses(cfg, "fixture-sc")
    assert ("pip_calibration", "minimal") in pairs
    assert ("agg_pip_calibration", "minimal") in pairs


def test_reduction_output_path():
    p = loader.reduction_output("BH", "MH", "pip")
    assert p == "results/by_batch/BH/fits/MH/reductions/pip.parquet"


def test_analysis_inputs_only_required_reductions(tmp_path):
    cfg = loader.load_config(FIXTURE_DIR)
    manifest = loader.manifest_dict(cfg["library"], cfg)
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
    assert fams == {"cox_reversed", "twogroup"}


def test_load_sc_bundle_tags_collections(tmp_path):
    import polars as pl
    import utils
    cfg = loader.load_config(FIXTURE_DIR)
    lib = cfg["library"]
    results = tmp_path / "results"
    # materialize + fit + reduce one collection's units for reduction "pip"
    for coll in loader.supercollection_collections(lib, "fixture-sc", cfg["supercollections"]["fixture-sc"]):
        for spec in coll["simulations"]:
            reps = (0, 1)
            bh = spec.hash  # coordinate-based batch hash (n_batches=1)
            sims_df = utils.simulate_batch(spec, replicates=reps)
            sample_md = __import__("plot_ready").build_sample_metadata(bh, sims_df)
            for mname, mspec in loader.resolve_methods_for_sc(lib, cfg["supercollections"]["fixture-sc"]).items():
                mh = loader.method_hash(mspec)
                fits = utils.fit_batch_method(spec, method_coord=mspec, replicates=reps).with_columns(pl.lit(bh).alias("batch_hash"))
                from reductions import ReductionContext
                from reductions.pip import build as build_pip
                red = build_pip(ReductionContext(fits=fits, sims=sims_df, sample_metadata=sample_md, sim_coordinate={}))
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


# ---------------------------------------------------------------------------
# P1.3 — MethodSpec dropped; methods are coordinate dicts
# ---------------------------------------------------------------------------

def _resolved_tiny_sim():
    """Build a SimulationSpec (with hash) for method-execution tests."""
    from functools import partial
    import core
    from gibss.distributions import Normal, PointMass
    spec = core.SimulationSpec(
        design_sampler=partial(core.gaussian_markov_X, n=30, p=8, rho=0.5),
        effect_sampler=partial(core.uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=2.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        error_sampler=None,
        base_seed=1,
        hash="tinyhash",
        name="tiny",
    )
    return core.simulate(spec, 0)


def test_expand_method_returns_coordinates():
    entry = {"function": "run_cox_method", "template": {"time_sign": -1.0},
             "over": {"threshold": [2.0], "L": [1]}}
    coords = loader.expand_method("cox", entry)
    assert [c["name"] for c in coords] == ["cox__threshold=2.00__L=1"]
    assert coords[0]["function"] == "run_cox_method"
    assert coords[0]["kwargs"] == {"time_sign": -1.0, "threshold": 2.0, "L": 1}


def test_run_method_executes(tmp_path):
    sim = _resolved_tiny_sim()
    coord = {"name": "cox_reversed__L=1", "function": "run_cox_method",
             "kwargs": {"threshold": None, "time_sign": 1.0, "L": 1}}
    row = loader.run_method(coord, sim)
    assert row["method"] == "cox_reversed__L=1" and "single_effects" in row


def test_manifest_dict_coordinate_shape():
    cfg = loader.load_config(FIXTURE_DIR)
    m = loader.manifest_dict(cfg["library"], cfg)
    assert set(m) == {"batches", "methods"}
    bh, b = next(iter(m["batches"].items()))
    assert b["hash"] == bh
    assert set(b["coordinate"]) == {"design", "enrichment", "signal", "error", "base_seed"}
    assert list(b["replicates"]) == list(range(2))   # fixture replicates_per_batch=2
    mh, mc = next(iter(m["methods"].items()))
    assert mc["name"] and "function" in mc


def test_code_files_point_at_modules():
    cfg = loader.load_config(FIXTURE_DIR); lib = cfg["library"]
    coord = loader.simulation_coordinate(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    files = loader.simulation_code_files(coord, lib)
    assert any(f.endswith("simulations/design/markov.py") for f in files)
    assert any(f.endswith("core.py") for f in files)
    rfiles = loader.reduction_code_files("pip", lib)
    assert rfiles == [f for f in rfiles if f.endswith("reductions/pip.py")]


def test_reduction_inputs_fixed_set():
    paths = loader.reduction_inputs({}, "BH", "MH")
    assert paths == [
        "results/by_batch/BH/fits/MH/fits.parquet",
        "results/by_batch/BH/simulations.parquet",
        "results/by_batch/BH/sample_metadata.parquet",
    ]


# ---------------------------------------------------------------------------
# P4.1 — predicates module + method_filter/simulation_filter as predicates
# ---------------------------------------------------------------------------

from experiments import predicates


def test_predicates():
    assert predicates.is_twogroup({"function": "run_twogroup_method"})
    assert not predicates.is_twogroup({"function": "run_cox_method"})
    assert predicates.has_causal({"enrichment": {"arguments": {"causal_effect": 2.0}}})
    assert not predicates.has_causal({"enrichment": {"arguments": {"causal_effect": 0.0}}})


def test_method_filter_predicate_selects_twogroup_only():
    """analysis_inputs for an analysis whose reduction uses method_filter: is_twogroup
    must yield only paths whose method coord has function == run_twogroup_method."""
    cfg = loader.load_config(FIXTURE_DIR)
    manifest = loader.manifest_dict(cfg["library"], cfg)
    # f1 reduction has method_filter: is_twogroup in fixture library; use f1_boxplot
    # fixture library only has pip, so we test via a custom lib with is_twogroup filter
    # Instead verify via collection_method_pairs that filtered pairs are twogroup-only.
    library = cfg["library"]
    library["reductions"]["pip_twogroup_only"] = {
        "function": "build_pip_plot_data",
        "method_filter": "is_twogroup",
    }
    library["analyses"]["pip_twogroup_only_calibration"] = {
        "requires": ["pip_twogroup_only"],
    }
    inputs = loader.analysis_inputs(cfg, manifest, "fixture-sc", "pip_twogroup_only_calibration")
    # all paths should contain the twogroup method hash, not cox
    all_methods = loader.all_methods(cfg)
    twogroup_coords = [c for c in all_methods.values() if c["function"] == "run_twogroup_method"]
    cox_coords = [c for c in all_methods.values() if c["function"] == "run_cox_method"]
    twogroup_hashes = {loader.method_hash(c) for c in twogroup_coords}
    cox_hashes = {loader.method_hash(c) for c in cox_coords}
    # every path must contain a twogroup hash
    assert all(any(mh in p for mh in twogroup_hashes) for p in inputs)
    # no path should contain a cox hash
    assert not any(any(mh in p for mh in cox_hashes) for p in inputs)


def test_over_aliases_assigns_per_collection_label():
    lib = _library_for_tests()
    lib["signals"]["loc_1.0"] = {"f0": {"PointMass": {"value": 0.0}},
        "f1": {"Normal": {"loc": 1.0, "scale": 0.1, "estimate_loc": False, "estimate_scale": False}}}
    block = {"template": {"design": "gaussian_p100", "enrichment": "ser_b2", "error": "gaussian"},
             "over": {"signal": ["loc_1.0", "loc_2.0"]},
             "aliases": ["lo", "hi"]}
    colls = loader.expand_collections(lib, "sc", block)
    assert [c["alias"] for c in colls] == ["lo", "hi"]


def test_over_aliases_length_mismatch_raises():
    import pytest
    lib = _library_for_tests()
    block = {"template": {"design": "gaussian_p100", "enrichment": "ser_b2", "error": "gaussian"},
             "over": {"signal": ["loc_2.0"]}, "aliases": ["a", "b"]}
    with pytest.raises(ValueError):
        loader.expand_collections(lib, "sc", block)


def test_009_cox_well_specified_config_shape():
    cfg = loader.load_config()
    lib = cfg["library"]
    sc_names = [name for name in cfg["supercollections"] if name.startswith("009-")]

    assert sc_names == [
        "009-hallmark-cox-well-specified",
        "009-c2-cox-well-specified",
        "009-c4-cox-well-specified",
        "009-c5-cox-well-specified",
    ]
    for sc_name in sc_names:
        sc = cfg["supercollections"][sc_name]
        collections = loader.supercollection_collections(lib, sc_name, sc)
        methods = loader.resolve_methods_for_sc(lib, sc)
        assert len(collections) == 8
        assert all(len(collection["simulations"]) == 2 for collection in collections)
        assert set(methods) == {"cox_reversed__L=1"}
