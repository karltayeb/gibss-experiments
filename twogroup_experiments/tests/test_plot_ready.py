from __future__ import annotations

import polars as pl

import plot_ready


def test_build_method_metadata_emits_method_threshold_rows():
    fits_df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L1", "logistic_threshold_L1", "twogroup_L1"],
            "threshold": [1.0, 2.0, None],
            "method_spec": [
                '{"fields":{"name":"logistic_threshold_L1","kwargs":{"L":1}}}',
                '{"fields":{"name":"logistic_threshold_L1","kwargs":{"L":1}}}',
                '{"fields":{"name":"twogroup_L1","kwargs":{"L":1}}}',
            ],
        }
    )

    metadata = plot_ready.build_method_metadata(fits_df)

    assert metadata.select("method", "threshold").rows() == [
        ("logistic_threshold_L1", 1.0),
        ("logistic_threshold_L1", 2.0),
        ("twogroup_L1", None),
    ]
    assert "method_display" in metadata.columns


def test_build_simulation_metadata_uses_collection_batch_info():
    collection = {
        "batches": [
            {
                "hash": "batch-a",
                "name": "batch-a",
                "simulation_spec": {"fields": {"name": "sim-a"}},
            }
        ]
    }

    simulation_metadata = plot_ready.build_simulation_metadata(collection)

    assert simulation_metadata["batch_hash"].to_list() == ["batch-a"]
    assert simulation_metadata["simulation_name"].to_list() == ["sim-a"]


def test_build_sample_metadata_uses_batch_hash_and_replicate():
    collection_batches = [{"hash": "batch-a", "name": "batch-a"}]
    simulations = {"batch-a": pl.DataFrame({"replicate": [0, 1]})}

    sample_metadata = plot_ready.build_sample_metadata(collection_batches, simulations)

    assert sample_metadata["sample_id"].to_list() == ["batch-a::0", "batch-a::1"]
    assert sample_metadata["batch_hash"].to_list() == ["batch-a", "batch-a"]


def test_build_pip_calibration_returns_collection_level_bins():
    per_sample = pl.DataFrame(
        {
            "sample_id": ["a::0", "a::0", "a::1", "a::1"],
            "method": ["logistic_threshold_L1"] * 4,
            "threshold": [1.0] * 4,
            "pip_bin_index": [0, 1, 0, 1],
            "n_exact": [10, 5, 8, 7],
            "n_causal_exact": [1, 2, 1, 3],
        }
    )

    result = plot_ready.aggregate_pip_calibration(per_sample)

    assert result.columns == [
        "method",
        "threshold",
        "pip_bin_index",
        "pip_left",
        "pip_right",
        "pip_mid",
        "n_total",
        "n_causal",
        "empirical_rate",
    ]
    assert result.height == 2


def test_build_power_fdp_returns_collection_level_curve():
    per_sample = pl.DataFrame(
        {
            "sample_id": ["a::0", "a::1"],
            "method": ["logistic_threshold_L1", "logistic_threshold_L1"],
            "threshold": [1.0, 1.0],
            "pip_threshold": [0.5, 0.5],
            "power": [0.2, 0.4],
            "fdp": [0.1, 0.3],
        }
    )

    result = plot_ready.aggregate_power_fdp(per_sample)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["method"] == "logistic_threshold_L1"
    assert row["threshold"] == 1.0
    assert row["pip_threshold"] == 0.5
    assert abs(row["power"] - 0.3) < 1e-9
    assert abs(row["fdp"] - 0.2) < 1e-9


def test_build_causal_pip_returns_collection_means():
    per_sample = pl.DataFrame(
        {
            "sample_id": ["a::0", "a::1"],
            "method": ["logistic_threshold_L1", "logistic_threshold_L1"],
            "threshold": [1.0, 1.0],
            "mean_causal_pip": [0.4, 0.6],
        }
    )

    result = plot_ready.aggregate_causal_pip(per_sample)

    assert result.rows(named=True) == [
        {
            "method": "logistic_threshold_L1",
            "threshold": 1.0,
            "mean_causal_pip": 0.5,
        }
    ]


def test_build_cs_summary_returns_three_metrics():
    per_sample = pl.DataFrame(
        {
            "sample_id": ["a::0", "a::0", "a::0"],
            "method": ["logistic_threshold_L1"] * 3,
            "threshold": [1.0] * 3,
            "metric": ["Power", "CS Size", "Coverage"],
            "value": [0.5, 4.0, 0.8],
        }
    )

    result = plot_ready.aggregate_cs_summary(per_sample)

    assert sorted(result["metric"].to_list()) == ["CS Size", "Coverage", "Power"]


def test_build_cs_size_histogram_returns_raw_observations():
    observations = pl.DataFrame(
        {
            "method": ["logistic_threshold_L1", "logistic_threshold_L1"],
            "threshold": [1.0, 1.0],
            "cs_size": [3, 5],
        }
    )

    result = plot_ready.finalize_cs_size_histogram(observations)

    assert result.rows(named=True) == observations.rows(named=True)


def test_build_ser_log_bf_histogram_returns_raw_observations():
    observations = pl.DataFrame(
        {
            "method": ["logistic_threshold_L1"],
            "threshold": [1.0],
            "ser_log_bf": [2.5],
        }
    )

    result = plot_ready.finalize_ser_log_bf_histogram(observations)

    assert result.rows(named=True) == observations.rows(named=True)
