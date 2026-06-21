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
                "__spec_hash__": "batch-a",
                "name": "batch-a",
                "simulation_spec": {"fields": {"name": "sim-a"}},
            }
        ]
    }

    simulation_metadata = plot_ready.build_simulation_metadata(collection)

    assert simulation_metadata["batch_hash"].to_list() == ["batch-a"]
    assert simulation_metadata["simulation_name"].to_list() == ["sim-a"]


def test_build_collection_yaml_node_roundtrip():
    # Synthetic nodes with __spec_hash__ (the shape these functions require)
    batch_node = {"__spec_hash__": "batch-1", "name": "sim-a", "fields": {}}
    method_node = {"__spec_hash__": "method-1", "name": "twogroup_L1", "fields": {}}

    result = plot_ready.build_collection_yaml_node(
        name="test_collection",
        batch_nodes=[batch_node],
        method_nodes=[method_node],
    )

    assert result["name"] == "test_collection"
    assert len(result["batches"]) == 1
    assert len(result["method_specs"]) == 1
    assert "__spec_hash__" not in result


def test_union_collection_yaml_nodes_deduplicates():
    # Two distinct batch nodes, one method node shared across two sub-collections
    batch_nodes = [
        {"__spec_hash__": "batch-1", "name": "sim-a", "fields": {}},
        {"__spec_hash__": "batch-2", "name": "sim-b", "fields": {}},
    ]
    method_node = {"__spec_hash__": "method-1", "name": "twogroup_L1", "fields": {}}

    node_a = plot_ready.build_collection_yaml_node(
        name="a", batch_nodes=batch_nodes[:1], method_nodes=[method_node]
    )
    node_b = plot_ready.build_collection_yaml_node(
        name="b", batch_nodes=batch_nodes, method_nodes=[method_node]
    )

    result = plot_ready.union_collection_yaml_nodes("union", [node_a, node_b])

    assert result["name"] == "union"
    assert len(result["batches"]) == 2   # deduped
    assert len(result["method_specs"]) == 1  # deduped
    assert "__spec_hash__" not in result


