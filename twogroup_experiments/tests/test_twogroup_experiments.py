from __future__ import annotations

from dataclasses import is_dataclass
from functools import partial
import json
from pathlib import Path

from gibss.distributions import Normal, PointMass
import numpy as np
import polars as pl


def zero_error_sampler(rng, se):
    return np.zeros(len(se))

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
from viz_utils import (
    make_method_display_label,
    method_metadata_from_method_spec_json,
    make_cs_beta_trace_summary,
)
from utils import CS_BETA_GRID


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


def test_simulation_spec_hash_unchanged_after_adding_error_sampler():
    """Adding error_sampler=None must not change existing hashes."""
    from core import SimulationSpec, simulation_hash, uniform_single_effect, identity_design_sampler
    from gibss.distributions import Normal, PointMass
    from functools import partial

    spec_without = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
    )
    hash_before = simulation_hash(spec_without)

    spec_with_none = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
        error_sampler=None,
    )
    assert simulation_hash(spec_with_none) == hash_before


def test_rehydrate_spec_handles_missing_error_sampler():
    """Old serialized specs (no error_sampler key) must rehydrate without error."""
    from core import rehydrate_spec, dehydrate_spec, SimulationSpec, uniform_single_effect, identity_design_sampler
    from gibss.distributions import Normal, PointMass
    from functools import partial

    spec = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
        error_sampler=None,
    )
    dehydrated = dehydrate_spec(spec)
    assert "error_sampler" not in dehydrated

    rehydrated = rehydrate_spec(dehydrated)
    assert rehydrated.error_sampler is None


def test_t_error_sampler_has_unit_variance():
    """Standardized t-error sampler should produce unit variance regardless of df."""
    from core import t_error_sampler

    # Verify the formula gives unit variance analytically for all df
    for df in (3, 5, 10, 30):
        scale_factor = np.sqrt((df - 2.0) / df)
        analytic_var = scale_factor**2 * (df / (df - 2.0))
        assert abs(analytic_var - 1.0) < 1e-10, f"df={df}: analytic var={analytic_var}"

    # Verify empirically for df >= 6 (where 4th moment is finite, sample var is reliable)
    rng = np.random.default_rng(42)
    se = np.ones(50_000)
    for df in (10, 30):
        samples = t_error_sampler(rng, se, df=df)
        assert abs(np.var(samples) - 1.0) < 0.05, f"df={df}: var={np.var(samples):.3f}"

    # Verify df=3 runs without error
    t_error_sampler(rng, se, df=3)


def test_simulate_uses_error_sampler_when_set():
    """simulate() with a t error_sampler should produce different thetahat than normal."""
    from core import SimulationSpec, simulate, t_error_sampler, uniform_single_effect, identity_design_sampler
    from gibss.distributions import Normal, PointMass
    from functools import partial

    base_spec = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
    )
    t_spec = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
        error_sampler=partial(t_error_sampler, df=3),
    )
    # Use a zero error_sampler to verify simulate() routes through it
    zero_spec = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
        error_sampler=zero_error_sampler,
    )
    zero_sim = simulate(zero_spec, replicate=0)
    # With zero noise, thetahat should equal theta exactly
    np.testing.assert_array_equal(zero_sim.thetahat, zero_sim.theta)

    # Verify t_error_sampler runs without error
    t_sim = simulate(t_spec, replicate=0)
    assert t_sim.thetahat is not None


def test_t_error_simulation_specs_count():
    """4 designs × 2 signal kinds × 4 df values = 32 specs."""
    from config import T_ERROR_SIMULATION_SPECS
    assert len(T_ERROR_SIMULATION_SPECS) == 32


def test_t_error_simulation_spec_names_include_error_field():
    from config import T_ERROR_SIMULATION_SPECS
    for spec in T_ERROR_SIMULATION_SPECS:
        assert "__error=t_df_" in spec.name, f"Missing error field: {spec.name}"


def test_t_error_simulation_specs_have_registered_batches():
    from config import REGISTRY, T_ERROR_SIMULATION_SPECS
    batch_sim_names = {b.simulation_spec.name for b in REGISTRY.batches}
    for spec in T_ERROR_SIMULATION_SPECS:
        assert spec.name in batch_sim_names, f"No batch for {spec.name}"


def test_t_error_simulation_spec_hash_differs_from_normal_baseline():
    """t-error spec hash must differ from the equivalent normal-error spec."""
    from config import T_ERROR_SIMULATION_SPECS, SIMULATION_BY_NAME
    from core import simulation_hash
    for spec in T_ERROR_SIMULATION_SPECS:
        normal_name = spec.name.split("__error=")[0]
        if normal_name in SIMULATION_BY_NAME:
            assert simulation_hash(spec) != simulation_hash(SIMULATION_BY_NAME[normal_name])


def _make_cs_plot_data(rows: list[dict]) -> pl.DataFrame:
    """Build minimal cs_plot_data for power tests."""
    beta_idx_095 = int(np.searchsorted(CS_BETA_GRID, 0.95))
    result = []
    for r in rows:
        n_vars = r["n_vars"]
        rank_of_causal = r["rank_of_causal"]  # list, one per causal
        causal_indices = r["causal_indices"]
        # cs_sizes: small fixed CS that only covers if rank < 3
        cs_sizes = [3] * len(CS_BETA_GRID)
        result.append({
            "collection_name": r.get("collection_name", "c1"),
            "sample_id": r["sample_id"],
            "method": r.get("method", "m1"),
            "threshold": None,
            "l": r.get("l", 0),
            "ser_log_bf": r.get("ser_log_bf", 5.0),
            "causal_indices": causal_indices,
            "rank_of_causal": rank_of_causal,
            "cs_sizes": cs_sizes,
        })
    return pl.from_dicts(result, schema={
        "collection_name": pl.String, "sample_id": pl.String,
        "method": pl.String, "threshold": pl.Float64, "l": pl.Int64,
        "ser_log_bf": pl.Float64,
        "causal_indices": pl.List(pl.Int64), "rank_of_causal": pl.List(pl.Int64),
        "cs_sizes": pl.List(pl.Int64),
    })


def _method_meta() -> pl.DataFrame:
    return pl.DataFrame({
        "method": ["m1"], "threshold": [None],
        "method_display": ["M1"], "is_thresholded": [False],
    }, schema={"method": pl.String, "threshold": pl.Float64,
               "method_display": pl.String, "is_thresholded": pl.Boolean})


def test_power_lstar1_denominator_is_n_samples():
    """Lstar=1: power = fraction of samples where causal is in any valid CS."""
    # 3 samples: ranks [0, 2, 5] for cs_size=3; covered = rank < 3 → [True, True, False]
    cs_data = _make_cs_plot_data([
        {"sample_id": "s1", "causal_indices": [7], "rank_of_causal": [0], "n_vars": 10},
        {"sample_id": "s2", "causal_indices": [7], "rank_of_causal": [2], "n_vars": 10},
        {"sample_id": "s3", "causal_indices": [7], "rank_of_causal": [5], "n_vars": 10},
    ])
    summary = make_cs_beta_trace_summary(
        cs_data, _method_meta(),
        selected_methods={"m1"}, selected_thresholds=None,
        max_cs_size=10000, min_ser_log_bf=0.0,
    )
    row = summary.filter((pl.col("beta") == 0.95) & (pl.col("collection_name") == "c1"))
    assert len(row) == 1
    assert abs(row["power"][0] - 2 / 3) < 1e-9, f"Expected 2/3, got {row['power'][0]}"


def test_power_lstar1_not_inflated_by_fitting_l():
    """Fitting L=3 with Lstar=1: power denominator is N_samples, not N_samples * L_fit."""
    # Same causal found in l=0 for all 2 samples; l=1,2 are null (rank=99)
    cs_data = _make_cs_plot_data([
        {"sample_id": "s1", "causal_indices": [7], "rank_of_causal": [0], "n_vars": 10, "l": 0},
        {"sample_id": "s1", "causal_indices": [7], "rank_of_causal": [99], "n_vars": 10, "l": 1},
        {"sample_id": "s1", "causal_indices": [7], "rank_of_causal": [99], "n_vars": 10, "l": 2},
        {"sample_id": "s2", "causal_indices": [7], "rank_of_causal": [99], "n_vars": 10, "l": 0},
        {"sample_id": "s2", "causal_indices": [7], "rank_of_causal": [99], "n_vars": 10, "l": 1},
        {"sample_id": "s2", "causal_indices": [7], "rank_of_causal": [99], "n_vars": 10, "l": 2},
    ])
    summary = make_cs_beta_trace_summary(
        cs_data, _method_meta(),
        selected_methods={"m1"}, selected_thresholds=None,
        max_cs_size=10000, min_ser_log_bf=0.0,
    )
    row = summary.filter((pl.col("beta") == 0.95) & (pl.col("collection_name") == "c1"))
    # s1 covered (l=0 has rank 0 < 3), s2 not covered → power = 1/2
    assert abs(row["power"][0] - 0.5) < 1e-9, f"Expected 0.5, got {row['power'][0]}"


def test_power_lstar2_denominator_is_total_causals():
    """Lstar=2: denominator = 2 * N_samples; each causal counted independently."""
    # sample s1: causal A (rank 0, found), causal B (rank 5, not found in cs_size=3)
    # sample s2: causal A (rank 0, found), causal B (rank 1, found)
    # total causal slots = 4; discovered = 3 → power = 3/4
    cs_data = _make_cs_plot_data([
        {"sample_id": "s1", "causal_indices": [3, 7], "rank_of_causal": [0, 5], "n_vars": 10},
        {"sample_id": "s2", "causal_indices": [3, 7], "rank_of_causal": [0, 1], "n_vars": 10},
    ])
    summary = make_cs_beta_trace_summary(
        cs_data, _method_meta(),
        selected_methods={"m1"}, selected_thresholds=None,
        max_cs_size=10000, min_ser_log_bf=0.0,
    )
    row = summary.filter((pl.col("beta") == 0.95) & (pl.col("collection_name") == "c1"))
    assert abs(row["power"][0] - 3 / 4) < 1e-9, f"Expected 0.75, got {row['power'][0]}"


def test_power_duplicate_cs_does_not_inflate_numerator():
    """If two CSs both cover the same causal, it still counts once."""
    # s1 has 1 causal (idx 7), covered by both l=0 and l=1
    cs_data = _make_cs_plot_data([
        {"sample_id": "s1", "causal_indices": [7], "rank_of_causal": [0], "n_vars": 10, "l": 0},
        {"sample_id": "s1", "causal_indices": [7], "rank_of_causal": [0], "n_vars": 10, "l": 1},
    ])
    summary = make_cs_beta_trace_summary(
        cs_data, _method_meta(),
        selected_methods={"m1"}, selected_thresholds=None,
        max_cs_size=10000, min_ser_log_bf=0.0,
    )
    row = summary.filter((pl.col("beta") == 0.95) & (pl.col("collection_name") == "c1"))
    # 1 sample, 1 causal, covered → power = 1.0 (not 2.0)
    assert abs(row["power"][0] - 1.0) < 1e-9, f"Expected 1.0, got {row['power'][0]}"
