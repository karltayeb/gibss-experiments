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
