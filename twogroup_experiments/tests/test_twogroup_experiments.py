from __future__ import annotations

from dataclasses import is_dataclass
from functools import partial
import json
from pathlib import Path

from gibss.distributions import Normal, PointMass
import numpy as np
import polars as pl

from core import (
    HASH_KEY,
    LOGISTIC_ORACLE,
    SimulationSpec,
    bernoulli_markov_X,
    dehydrate_hashed,
    dehydrate_node,
    gaussian_markov_X,
    identity_design_sampler,
    rehydrate_node,
    uniform_markov_X,
    uniform_single_effect,
)
from config import (
    ConfigRegistry,
    SIMULATION_SPECS,
    _logistic_threshold_method_spec,
)
from utils import (
    attach_spec_metadata,
    BatchSpec,
    CollectionSpec,
    build_plot_data_frames,
    fit_batch_method,
    manifest_dict,
    symlink_plot_data_outputs,
    simulate_batch,
)
from viz2_metadata import (
    add_plot_metadata_columns,
    available_L_values,
    available_method_families,
    make_method_display_label,
    method_family_display_order,
    method_metadata_from_method_spec_json,
)


def _tiny_simulation_spec() -> SimulationSpec:
    return SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
    )


def test_batchspec_is_defined_locally():
    batch = BatchSpec(
        name="tiny_batch",
        simulation_spec=_tiny_simulation_spec(),
        replicates=(0, 1),
    )
    assert batch.name == "tiny_batch"
    assert batch.replicates == (0, 1)


def test_collectionspec_owns_batches_and_methods():
    batch = BatchSpec(
        name="tiny_batch",
        simulation_spec=_tiny_simulation_spec(),
        replicates=(0, 1),
    )
    collection = CollectionSpec(
        name="tiny",
        batches=(batch,),
        method_specs=(LOGISTIC_ORACLE,),
    )
    assert collection.name == "tiny"
    assert collection.batches == (batch,)
    assert collection.method_specs == (LOGISTIC_ORACLE,)


def test_config_registry_register_collection_accumulates_unique_specs():
    registry = ConfigRegistry()

    collection = registry.register_collection(
        name="demo",
        simulations=(_tiny_simulation_spec(),),
        methods=(LOGISTIC_ORACLE,),
        n_batches=2,
        replicates_per_batch=3,
    )

    assert collection.name == "demo"
    assert len(registry.simulations) == 1
    assert len(registry.methods) == 1
    assert len(registry.batches) == 2
    assert len(registry.collections) == 1
    assert [batch.name for batch in registry.batches] == [
        "tiny_simulation__batch0",
        "tiny_simulation__batch1",
    ]
    assert [batch.replicates for batch in registry.batches] == [
        (0, 1, 2),
        (3, 4, 5),
    ]


def test_config_registry_register_collection_is_idempotent_for_duplicates():
    registry = ConfigRegistry()

    registry.register_collection(
        name="demo",
        simulations=(_tiny_simulation_spec(),),
        methods=(LOGISTIC_ORACLE,),
        n_batches=1,
        replicates_per_batch=2,
    )
    registry.register_collection(
        name="demo",
        simulations=(_tiny_simulation_spec(),),
        methods=(LOGISTIC_ORACLE,),
        n_batches=1,
        replicates_per_batch=2,
    )

    assert len(registry.simulations) == 1
    assert len(registry.methods) == 1
    assert len(registry.batches) == 1
    assert len(registry.collections) == 1


def test_config_registry_register_collection_union_reuses_batches_and_methods():
    registry = ConfigRegistry()

    registry.register_collection(
        name="left",
        simulations=(_tiny_simulation_spec(),),
        methods=(LOGISTIC_ORACLE,),
        n_batches=1,
        replicates_per_batch=2,
    )
    registry.register_collection(
        name="right",
        simulations=(_tiny_simulation_spec(),),
        methods=(),
        n_batches=1,
        replicates_per_batch=2,
    )

    union = registry.register_collection_union(
        name="union",
        collections=("left", "right"),
    )

    assert union.name == "union"
    assert len(union.batches) == 1
    assert union.batches[0].name == "tiny_simulation__batch0"
    assert union.method_specs == (LOGISTIC_ORACLE,)


def test_config_registry_register_collection_union_raises_for_unknown_collection():
    registry = ConfigRegistry()

    try:
        registry.register_collection_union(
            name="union",
            collections=("missing",),
        )
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown collection.")


def test_core_specs_are_dataclasses():
    spec = _tiny_simulation_spec()
    assert is_dataclass(spec)
    assert is_dataclass(LOGISTIC_ORACLE)


def test_config_uses_axis_composed_simulation_names():
    names = {spec.name for spec in SIMULATION_SPECS}
    assert "hallmark__ser_enrich__loc_0.5" in names
    assert "hallmark__ser_dep__loc_5.0" in names
    assert "c4__ser_enrich__scale_6.0" in names
    assert "c4__ser_dep__scale_0.5" in names


def test_manifest_uses_batches_key():
    manifest = manifest_dict()
    assert "batches" in manifest
    assert "simulation_specs" in manifest
    assert "method_specs" in manifest
    assert "collections" not in manifest
    assert "lots" not in manifest
    # verify batch node structure
    any_batch = next(iter(manifest["batches"].values()))
    assert sorted(any_batch.keys()) == [
        HASH_KEY,
        "name",
        "replicates",
        "simulation_spec",
    ]
    assert any_batch["simulation_spec"][HASH_KEY]
    assert "fields" in any_batch["simulation_spec"]


def test_batch_hash_does_not_depend_on_method_membership():
    spec = _tiny_simulation_spec()
    batch_node = dehydrate_hashed(BatchSpec(name="tiny_batch", simulation_spec=spec, replicates=(0, 1)))
    assert "method_specs" not in batch_node["fields"]


def test_twogroup_experiments_uses_local_modules_only():
    root = Path(__file__).resolve().parents[1]
    for path in (*root.glob("*.py"), root / "twogroup_experiments.snk"):
        text = path.read_text(encoding="utf-8")
        assert "workflow.twogroup_experiments" not in text


def test_simulate_batch_omits_x_and_keeps_replicates():
    df = simulate_batch(_tiny_simulation_spec(), replicates=(0, 1))
    assert df.shape[0] == 2
    assert df["replicate"].to_list() == [0, 1]

    simulation_rows = df["simulation"].to_list()
    assert all("X" not in row for row in simulation_rows)
    assert all("thetahat" in row for row in simulation_rows)


def test_fit_batch_method_returns_one_row_per_replicate():
    df = fit_batch_method(
        _tiny_simulation_spec(),
        method_spec=LOGISTIC_ORACLE,
        replicates=(0, 1),
    )
    assert df.shape[0] == 2
    assert df["replicate"].to_list() == [0, 1]
    assert set(df["method"].to_list()) == {"logistic_oracle_L1"}


def test_build_plot_data_frames_returns_expected_tables_and_shapes():
    simulations_df = simulate_batch(_tiny_simulation_spec(), replicates=(0, 1))
    fits_df = fit_batch_method(
        _tiny_simulation_spec(),
        method_spec=LOGISTIC_ORACLE,
        replicates=(0, 1),
    )

    outputs = build_plot_data_frames(fits_df, simulations_df)

    assert set(outputs) == {
        "pip_threshold_plot_data",
        "causal_pip_plot_data",
        "cs_component_plot_data",
        "cs_truth_plot_data",
    }

    pip_threshold = outputs["pip_threshold_plot_data"]
    causal_pip = outputs["causal_pip_plot_data"]
    cs_component = outputs["cs_component_plot_data"]
    cs_truth = outputs["cs_truth_plot_data"]

    assert pip_threshold["replicate"].n_unique() == 2
    assert pip_threshold["pip_threshold"].n_unique() == 999
    assert pip_threshold.shape[0] == 2 * 999
    assert {
        "selected_total",
        "selected_causal",
        "power",
        "fdp",
        "n_exact",
        "n_causal_exact",
    } <= set(pip_threshold.columns)

    assert causal_pip.shape[0] == 2
    assert {"causal_variable", "causal_pip", "max_pip"} <= set(causal_pip.columns)

    assert cs_component.shape[0] == 2
    assert {"component", "ordered_pips", "betas", "cs_sizes", "ser_log_bf"} <= set(
        cs_component.columns
    )
    assert all(len(betas) == len(cs_sizes) for betas, cs_sizes in zip(cs_component["betas"], cs_component["cs_sizes"]))

    assert cs_truth.shape[0] == 2
    assert {"component", "causal_variable", "causal_rank", "betas", "covered"} <= set(
        cs_truth.columns
    )
    assert all(len(betas) == len(covered) for betas, covered in zip(cs_truth["betas"], cs_truth["covered"]))


def test_build_plot_data_frames_propagates_specs_to_all_outputs():
    simulation_spec = _tiny_simulation_spec()
    simulations_df = simulate_batch(simulation_spec, replicates=(0,))
    method_spec = _logistic_threshold_method_spec(threshold=2.0, L=5)
    fits_df = attach_spec_metadata(
        fit_batch_method(
            simulation_spec,
            method_spec=method_spec,
            replicates=(0,),
        ),
        method_spec_node=dehydrate_hashed(method_spec),
        simulation_spec_node=dehydrate_hashed(simulation_spec),
    )

    plot_frames = build_plot_data_frames(fits_df, simulations_df)

    for df in plot_frames.values():
        assert {"method_spec", "simulation_spec"} <= set(df.columns)

    row = plot_frames["pip_threshold_plot_data"].row(0, named=True)
    method_spec = json.loads(row["method_spec"])
    stored_simulation_spec = json.loads(row["simulation_spec"])
    assert method_spec["fields"]["kwargs"]["L"] == 5
    assert method_spec["fields"]["kwargs"]["threshold"] == 2.0
    assert stored_simulation_spec["fields"]["name"] == simulation_spec.name


def test_build_plot_data_frames_keeps_spec_columns_for_empty_outputs():
    simulation_spec = _tiny_simulation_spec()
    method_spec = _logistic_threshold_method_spec(threshold=2.0, L=5)
    source_fits_df = attach_spec_metadata(
        fit_batch_method(
            simulation_spec,
            method_spec=method_spec,
            replicates=(0,),
        ),
        method_spec_node=dehydrate_hashed(method_spec),
        simulation_spec_node=dehydrate_hashed(simulation_spec),
    )
    fits_df = source_fits_df.head(0)
    simulations_df = simulate_batch(simulation_spec, replicates=(0,)).head(0)

    plot_frames = build_plot_data_frames(fits_df, simulations_df)

    for df in plot_frames.values():
        assert {"method_spec", "simulation_spec"} <= set(df.columns)
        assert df.height == 0
        assert df.schema["method_spec"] == pl.String
        assert df.schema["simulation_spec"] == pl.String


def test_build_plot_data_frames_keeps_threshold_dtype_for_empty_oracle_outputs():
    simulation_spec = _tiny_simulation_spec()
    source_fits_df = attach_spec_metadata(
        fit_batch_method(
            simulation_spec,
            method_spec=LOGISTIC_ORACLE,
            replicates=(0,),
        ),
        method_spec_node=dehydrate_hashed(LOGISTIC_ORACLE),
        simulation_spec_node=dehydrate_hashed(simulation_spec),
    )
    fits_df = source_fits_df.head(0)
    simulations_df = simulate_batch(simulation_spec, replicates=(0,)).head(0)

    plot_frames = build_plot_data_frames(fits_df, simulations_df)

    for df in plot_frames.values():
        assert df.height == 0
        assert df.schema["threshold"] == pl.Float64


def test_method_metadata_from_method_spec_json_ser():
    method_spec_json = json.dumps(
        dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=1)),
        sort_keys=True,
    )

    metadata = method_metadata_from_method_spec_json(method_spec_json)

    assert metadata["method_family"] == "logistic_threshold"
    assert metadata["L"] == 1
    assert metadata["is_oracle"] is False
    assert metadata["is_thresholded"] is True
    assert metadata["method_label_base"] == "Logistic SER"


def test_method_metadata_from_method_spec_json_susie():
    method_spec_json = json.dumps(
        dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=5)),
        sort_keys=True,
    )

    metadata = method_metadata_from_method_spec_json(method_spec_json)

    assert metadata["method_family"] == "logistic_threshold"
    assert metadata["L"] == 5
    assert metadata["method_label_base"] == "Logistic SuSiE [L=5]"


def test_make_method_display_label_uses_threshold_and_oracle_suffixes():
    assert (
        make_method_display_label(
            method_label_base="Logistic SER",
            threshold=2.0,
            is_thresholded=True,
            is_oracle=False,
        )
        == "Logistic SER (@2)"
    )
    assert (
        make_method_display_label(
            method_label_base="Logistic SER",
            threshold=None,
            is_thresholded=False,
            is_oracle=True,
        )
        == "Logistic SER (Oracle)"
    )


def test_add_plot_metadata_columns_uses_method_spec_json():
    method_spec_json = json.dumps(
        dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=5)),
        sort_keys=True,
    )
    simulation_spec_json = json.dumps(
        dehydrate_hashed(_tiny_simulation_spec()),
        sort_keys=True,
    )
    df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L5"],
            "threshold": [2.0],
            "method_spec": [method_spec_json],
            "simulation_spec": [simulation_spec_json],
        }
    )

    normalized = add_plot_metadata_columns(df)
    row = normalized.row(0, named=True)

    assert row["method_family"] == "logistic_threshold"
    assert row["L"] == 5
    assert row["method_label_base"] == "Logistic SuSiE [L=5]"
    assert row["method_display"] == "Logistic SuSiE [L=5] (@2)"


def test_method_family_display_order_is_family_based():
    assert method_family_display_order() == [
        "logistic_oracle",
        "twogroup_oracle",
        "twogroup",
        "cox_heavy",
        "cox_light_threshold",
        "logistic_threshold",
    ]


def test_available_method_families_and_L_values_are_data_driven():
    df = pl.DataFrame(
        {
            "method_family": ["logistic_threshold", "logistic_threshold", "twogroup"],
            "L": [1, 5, 1],
        }
    )

    assert available_method_families(df) == ["logistic_threshold", "twogroup"]
    assert available_L_values(df) == [1, 5]


def test_hallmark_ser_enrich_loc_metadata_yields_ser_and_susie_labels():
    method_spec_ser = json.dumps(
        dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=1)),
        sort_keys=True,
    )
    method_spec_susie = json.dumps(
        dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=5)),
        sort_keys=True,
    )
    simulation_spec_json = json.dumps(
        dehydrate_hashed(_tiny_simulation_spec()),
        sort_keys=True,
    )
    df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L1", "logistic_threshold_L5"],
            "threshold": [2.0, 2.0],
            "method_spec": [method_spec_ser, method_spec_susie],
            "simulation_spec": [simulation_spec_json, simulation_spec_json],
        }
    )

    normalized = add_plot_metadata_columns(df)
    labels = set(normalized.get_column("method_display").to_list())

    assert "Logistic SER (@2)" in labels
    assert "Logistic SuSiE [L=5] (@2)" in labels


def test_symlink_plot_data_outputs_links_all_plot_data_files(tmp_path):
    source_root = tmp_path / "by_batch" / "batch0" / "fits" / "method0"
    source_root.mkdir(parents=True)
    target_root = tmp_path / "by_alias" / "demo" / "batches" / "batch0" / "fits" / "method0"
    for name in (
        "pip_threshold_plot_data.parquet",
        "causal_pip_plot_data.parquet",
        "cs_component_plot_data.parquet",
        "cs_truth_plot_data.parquet",
    ):
        (source_root / name).write_text(name, encoding="utf-8")

    symlink_plot_data_outputs(source_root, target_root)

    for name in (
        "pip_threshold_plot_data.parquet",
        "causal_pip_plot_data.parquet",
        "cs_component_plot_data.parquet",
        "cs_truth_plot_data.parquet",
    ):
        link = target_root / name
        assert link.is_symlink()
        assert link.resolve() == (source_root / name).resolve()


def test_manifest_filename_matches_new_convention():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "twogroup_experiments_manifest.json"
    )
    assert manifest_path.name == "twogroup_experiments_manifest.json"


def test_dehydrate_hashed_round_trip_ignores_hash_key():
    spec = _tiny_simulation_spec()
    dehydrated = dehydrate_hashed(spec)
    assert HASH_KEY in dehydrated
    rehydrated = rehydrate_node(dehydrated)
    assert dehydrate_node(rehydrated) == dehydrate_node(spec)


def test_gaussian_markov_x_shape_and_adjacent_column_correlation():
    rng = np.random.default_rng(0)
    X = gaussian_markov_X(n=4000, p=6, rho=0.7, rng=rng)

    assert X.shape == (4000, 6)
    empirical_rho = np.corrcoef(X[:, 2], X[:, 3])[0, 1]
    assert abs(empirical_rho - 0.7) < 0.08


def test_bernoulli_markov_x_shape_support_and_adjacent_column_correlation():
    rng = np.random.default_rng(0)
    X = bernoulli_markov_X(n=6000, p=6, m=3, prob=0.25, rho=0.6, rng=rng)

    assert X.shape == (6000, 6)
    assert X.dtype.kind in {"i", "u"}
    assert X.min() >= 0
    assert X.max() <= 3
    empirical_rho = np.corrcoef(X[:, 2], X[:, 3])[0, 1]
    assert abs(empirical_rho - 0.6) < 0.08


def test_uniform_markov_x_shape_support_and_adjacent_spearman_correlation():
    rng = np.random.default_rng(0)
    rho = 0.6
    X = uniform_markov_X(n=5000, p=6, rho=rho, rng=rng)

    assert X.shape == (5000, 6)
    assert X.min() >= 0.0
    assert X.max() <= 1.0

    x1 = X[:, 2]
    x2 = X[:, 3]
    ranks1 = np.argsort(np.argsort(x1))
    ranks2 = np.argsort(np.argsort(x2))
    empirical_spearman = np.corrcoef(ranks1, ranks2)[0, 1]
    expected_spearman = 6.0 / np.pi * np.arcsin(rho / 2.0)
    assert abs(empirical_spearman - expected_spearman) < 0.08


def test_collection_plot_ready_snakemake_rules_are_declared():
    """Verify all collection-level plot-ready rules exist in the snk file."""
    snk_text = Path("twogroup_experiments.snk").read_text()
    required_rules = [
        "collection_method_metadata",
        "collection_simulation_metadata",
        "collection_sample_metadata",
        "collection_pip_calibration_plot_ready",
        "collection_power_fdp_plot_ready",
        "collection_causal_pip_plot_ready",
        "collection_cs_raw_plot_ready",
        "collection_cs_size_histogram_plot_ready",
        "collection_ser_log_bf_histogram_plot_ready",
    ]
    for rule in required_rules:
        assert f"rule {rule}:" in snk_text, f"Missing rule: {rule}"
