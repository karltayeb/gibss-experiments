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


def test_dashboard_notebook_module_loads():
    import runpy
    from pathlib import Path

    globals_dict = runpy.run_path(
        str(Path("notebooks") / "dashboard.py"),
        run_name="dashboard_test",
    )
    assert "app" in globals_dict


def test_build_collection_yaml_node_roundtrip():
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parent.parent / "results" / "manifest.json").read_text()
    )
    batch_hash = next(iter(manifest["batches"]))
    method_hash = next(iter(manifest["method_specs"]))
    batch_node = manifest["batches"][batch_hash]
    method_node = manifest["method_specs"][method_hash]

    result = plot_ready.build_collection_yaml_node(
        name="test_collection",
        batch_nodes=[batch_node],
        method_nodes=[method_node],
    )

    assert result["name"] == "test_collection"
    assert len(result["batches"]) == 1
    assert len(result["method_specs"]) == 1
    assert "__spec_hash__" in result


def test_union_collection_yaml_nodes_deduplicates():
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parent.parent / "results" / "manifest.json").read_text()
    )
    batch_hashes = list(manifest["batches"].keys())[:2]
    method_hash = next(iter(manifest["method_specs"]))
    batch_nodes = [manifest["batches"][h] for h in batch_hashes]
    method_node = manifest["method_specs"][method_hash]

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
    assert "__spec_hash__" in result


def _make_pip_fits_df():
    """Minimal fits_df with new schema: single_effects as list of structs."""
    return pl.DataFrame({
        "method": ["cox_L1", "cox_L1"],
        "threshold": [None, None],
        "batch_hash": ["batchA", "batchA"],
        "replicate": [0, 1],
        "single_effects": [
            [{"alpha": [0.9, 0.05, 0.05], "mu": [1.0, 0.0, 0.0], "var": [0.1, 0.1, 0.1],
              "prior_variance": 1.0, "marginal_log_likelihood": -1.0,
              "null_log_likelihood": -2.0, "ser_log_bf": 1.0, "kl": 0.1}],
            [{"alpha": [0.1, 0.8, 0.1], "mu": [0.0, 1.0, 0.0], "var": [0.1, 0.1, 0.1],
              "prior_variance": 1.0, "marginal_log_likelihood": -1.5,
              "null_log_likelihood": -2.5, "ser_log_bf": 1.0, "kl": 0.2}],
        ],
        "credible_sets": [
            [{"cs": [0], "cs_size": 1, "causal_indices": [0], "causal_in_cs": True,
              "top_feature": 0, "top_feature_is_causal": True}],
            [{"cs": [1], "cs_size": 1, "causal_indices": [0], "causal_in_cs": False,
              "top_feature": 1, "top_feature_is_causal": False}],
        ],
        "fit_summary": [
            {"n_selected": 10, "n_iter": 5, "converged": True},
            {"n_selected": 8, "n_iter": 4, "converged": True},
        ],
    })


def _make_sample_metadata():
    return pl.DataFrame({
        "sample_id": ["batchA::0", "batchA::1"],
        "batch_hash": ["batchA", "batchA"],
        "replicate": [0, 1],
    })


def _make_simulations_by_batch():
    return {
        "batchA": pl.DataFrame({
            "replicate": [0, 1],
            "simulation": [
                {"causal_indices": [0], "causal_effects": [1.0]},
                {"causal_indices": [0], "causal_effects": [1.0]},
            ],
        })
    }


def test_build_pip_plot_data_schema():
    fits_df = _make_pip_fits_df()
    sample_metadata = _make_sample_metadata()
    simulations_by_batch = _make_simulations_by_batch()

    result = plot_ready.build_pip_plot_data(fits_df, sample_metadata, simulations_by_batch)

    assert result.height == 2  # one row per (sample, method, threshold)
    assert set(result.columns) == {
        "sample_id", "method", "threshold",
        "causal_indices", "causal_pips",
        "pip_bin_counts", "pip_bin_causal_counts",
        "power_at_threshold", "fdp_at_threshold",
    }
    assert result["pip_bin_counts"].dtype == pl.List(pl.Int64)
    assert result["pip_bin_counts"][0].len() == 20
    assert result["power_at_threshold"][0].len() == 10


def test_build_pip_plot_data_causal_pips_correct():
    """Verifies marginal_pip = 1 - prod_l(1 - alpha_lj) at causal indices."""
    fits_df = _make_pip_fits_df()
    sample_metadata = _make_sample_metadata()
    simulations_by_batch = _make_simulations_by_batch()

    result = plot_ready.build_pip_plot_data(fits_df, sample_metadata, simulations_by_batch)

    # replicate 0: alpha=[0.9, 0.05, 0.05], L=1, marginal_pip=alpha, causal=0 → pip=0.9
    row0 = result.filter(pl.col("sample_id") == "batchA::0").row(0, named=True)
    assert abs(row0["causal_pips"][0] - 0.9) < 1e-9

    # replicate 1: alpha=[0.1, 0.8, 0.1], L=1, causal=0 → pip=0.1
    row1 = result.filter(pl.col("sample_id") == "batchA::1").row(0, named=True)
    assert abs(row1["causal_pips"][0] - 0.1) < 1e-9


def test_build_cs_plot_data_schema():
    from utils import CS_BETA_GRID

    fits_df = _make_pip_fits_df()
    sample_metadata = _make_sample_metadata()

    result = plot_ready.build_cs_plot_data(fits_df, sample_metadata)

    # 2 replicates × L=1 effect = 2 rows
    assert result.height == 2
    assert set(result.columns) == {
        "sample_id", "method", "threshold", "l",
        "ser_log_bf", "causal_indices", "causal_alpha", "rank_of_causal",
        "mass_above_causal", "cs_sizes",
    }
    assert result["l"].dtype == pl.Int64
    assert result["cs_sizes"].dtype == pl.List(pl.Int64)


def test_build_cs_plot_data_cs_sizes_length():
    from utils import CS_BETA_GRID

    fits_df = _make_pip_fits_df()
    sample_metadata = _make_sample_metadata()

    result = plot_ready.build_cs_plot_data(fits_df, sample_metadata)

    for row in result.iter_rows(named=True):
        assert len(row["cs_sizes"]) == len(CS_BETA_GRID)
        assert len(row["rank_of_causal"]) == len(row["causal_indices"])
        assert len(row["mass_above_causal"]) == len(row["causal_indices"])
