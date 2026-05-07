from __future__ import annotations

from dataclasses import is_dataclass
from functools import partial
from pathlib import Path

from gibss.distributions import Normal, PointMass
import numpy as np

from twogroup_experiments.core import (
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
from twogroup_experiments.utils import (
    BatchSpec,
    CollectionSpec,
    build_plot_data_frames,
    fit_batch_method,
    manifest_dict,
    symlink_plot_data_outputs,
    simulate_batch,
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


def test_core_specs_are_dataclasses():
    spec = _tiny_simulation_spec()
    assert is_dataclass(spec)
    assert is_dataclass(LOGISTIC_ORACLE)


def test_manifest_uses_batches_key():
    manifest = manifest_dict()
    assert "batches" in manifest
    assert "collections" in manifest
    assert "lots" not in manifest
    tiny_batch = manifest["collections"]["tiny_test"]["batches"][0]
    assert sorted(tiny_batch.keys()) == [
        HASH_KEY,
        "name",
        "replicates",
        "simulation_spec",
    ]
    assert tiny_batch["name"] == "tiny_test"
    assert tiny_batch["simulation_spec"][HASH_KEY]
    assert tiny_batch["simulation_spec"]["fields"]["name"] == "hallmark_ser_local_a"
    assert all(
        HASH_KEY in method_spec
        for method_spec in manifest["collections"]["tiny_test"]["method_specs"]
    )
    assert manifest["collections"]["tiny_test"]["name"] == "tiny_test"


def test_batch_hash_does_not_depend_on_method_membership():
    spec = _tiny_simulation_spec()
    batch_node = dehydrate_hashed(BatchSpec(name="tiny_batch", simulation_spec=spec, replicates=(0, 1)))
    assert "method_specs" not in batch_node["fields"]


def test_twogroup_experiments_uses_local_modules_only():
    root = Path(__file__).resolve().parents[1]
    for path in (*root.glob("*.py"), root / "twogroup_experiments.snk", root / "update_manifest.py"):
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
    assert set(df["method"].to_list()) == {"logistic_oracle"}


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
