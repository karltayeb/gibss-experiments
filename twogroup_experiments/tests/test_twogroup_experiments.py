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
    SIMULATION_SPECS,
    _logistic_threshold_method_spec,
)
from utils import (
    attach_spec_metadata,
    BatchSpec,
    fit_batch_method,
    manifest_dict,
    simulate_batch,
)
from viz_utils import make_method_display_label, method_metadata_from_method_spec_json


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


def test_core_specs_are_dataclasses():
    spec = _tiny_simulation_spec()
    assert is_dataclass(spec)
    assert is_dataclass(LOGISTIC_ORACLE)


def test_config_uses_axis_composed_simulation_names():
    names = {spec.name for spec in SIMULATION_SPECS}
    assert "design=hallmark__enrichment=ser_enrich__signal=loc_0.25" in names
    assert "design=hallmark__enrichment=ser_enrich__signal=scale_5.00" in names
    assert "design=c4__enrichment=ser_enrich__signal=scale_1.75" in names
    assert (
        "design=gaussian_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=loc_1.50"
        in names
    )
    assert (
        "design=uniform_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=scale_1.75"
        in names
    )


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


def test_fit_batch_method_schema_has_single_effects_list():
    df = fit_batch_method(
        _tiny_simulation_spec(),
        method_spec=_logistic_threshold_method_spec(threshold=2.0, L=1),
        replicates=(0,),
    )
    assert "single_effects" in df.columns
    assert "credible_sets" in df.columns
    assert "ser_posterior" not in df.columns
    assert "credible_set" not in df.columns
    row = df.row(0, named=True)
    assert isinstance(row["single_effects"], list)
    assert isinstance(row["credible_sets"], list)
    assert len(row["single_effects"]) == 1
    assert len(row["credible_sets"]) == 1
    effect = row["single_effects"][0]
    assert "alpha" in effect
    assert "ser_log_bf" in effect
    cs = row["credible_sets"][0]
    assert "cs_size" in cs
    assert "causal_in_cs" in cs


def test_fit_batch_method_L5_has_5_effects():
    df = fit_batch_method(
        _tiny_simulation_spec(),
        method_spec=_logistic_threshold_method_spec(threshold=2.0, L=5),
        replicates=(0,),
    )
    row = df.row(0, named=True)
    assert len(row["single_effects"]) == 5
    assert len(row["credible_sets"]) == 5


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


def test_all_threshold_sweep_methods_in_registry():
    from config import REGISTRY, THRESHOLD_SWEEP_SER_SPECS, THRESHOLD_SWEEP_SUSIE_SPECS
    registered_hashes = {dehydrate_hashed(m)[HASH_KEY] for m in REGISTRY.methods}
    for spec in THRESHOLD_SWEEP_SER_SPECS + THRESHOLD_SWEEP_SUSIE_SPECS:
        h = dehydrate_hashed(spec)[HASH_KEY]
        assert h in registered_hashes, f"Method {spec.name} not in registry"


def test_null_enrich_simulations_have_registered_batches():
    from config import REGISTRY, NULL_ENRICH_SIMULATION_SPECS
    batch_sim_names = {b.simulation_spec.name for b in REGISTRY.batches}
    for spec in NULL_ENRICH_SIMULATION_SPECS:
        assert spec.name in batch_sim_names, f"No batch for {spec.name}"


def test_collection_plot_ready_snakemake_rules_are_declared():
    """Verify all collection-level plot-ready rules exist in the snk file."""
    snk_text = Path("twogroup_experiments.snk").read_text()
    required_rules = [
        "collection_method_metadata",
        "collection_simulation_metadata",
        "collection_sample_metadata",
        "collection_pip_plot_data",
        "collection_cs_plot_data",
    ]
    for rule in required_rules:
        assert f"rule {rule}:" in snk_text, f"Missing rule: {rule}"
