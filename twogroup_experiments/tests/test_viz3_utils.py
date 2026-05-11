from __future__ import annotations

import json
from pathlib import Path
import runpy

import polars as pl
import yaml

import viz3_utils


def mock_method_spec(name: str, L: int = 1) -> str:
    return json.dumps(
        {
            "fields": {
                "name": name,
                "kwargs": {"L": L},
            }
        }
    )


def test_method_metadata_from_method_spec_json_parses_threshold_method():
    metadata = viz3_utils.method_metadata_from_method_spec_json(
        mock_method_spec("logistic_threshold_L1", L=1)
    )

    assert metadata["method_family"] == "logistic_threshold"
    assert metadata["L"] == 1
    assert metadata["is_thresholded"] is True
    assert metadata["is_oracle"] is False
    assert metadata["method_label_base"] == "Logistic SER"


def test_add_plot_metadata_columns_adds_display_columns():
    df = pl.DataFrame(
        {
            "method": ["twogroup_L5"],
            "method_spec": [mock_method_spec("twogroup_L5", L=5)],
            "threshold": [None],
        }
    )

    enriched = viz3_utils.add_plot_metadata_columns(df)

    assert {
        "method_family",
        "L",
        "is_thresholded",
        "is_oracle",
        "method_label_base",
        "method_display",
    }.issubset(enriched.columns)
    assert enriched["method_display"].to_list() == ["Twogroup SuSiE [L=5]"]


def test_empty_plot_tables_have_expected_columns():
    pip_df = viz3_utils.empty_pip_threshold_plot_data()

    assert "pip_threshold" in pip_df.columns


def test_load_collection_bundle_reads_present_tables(tmp_path: Path):
    alias_root = tmp_path / "by_alias"
    collection_root = alias_root / "demo"
    fit_dir = collection_root / "batches" / "batch-a" / "fits" / "fit-a"
    fit_dir.mkdir(parents=True)
    (collection_root / "collection_spec.yaml").write_text(
        yaml.safe_dump({"name": "demo", "batches": [{"name": "batch-a"}]})
    )

    pip_df = pl.DataFrame(
        {
            "replicate": [0],
            "method": ["twogroup_L1"],
            "threshold": [None],
            "method_spec": [mock_method_spec("twogroup_L1", L=1)],
            "simulation_spec": ["sim"],
            "pip_threshold": [0.2],
            "selected_total": [5],
            "selected_causal": [2],
            "power": [0.4],
            "fdp": [0.1],
            "n_exact": [5],
            "n_causal_exact": [2],
            "batch_hash": ["a"],
            "batch_name": ["batch-a"],
            "simulation_name": ["sim-a"],
        }
    )
    pip_df.write_parquet(fit_dir / "pip_threshold_plot_data.parquet")

    collections = viz3_utils.load_collection_specs(alias_root)
    bundle = viz3_utils.load_collection_bundle(alias_root / "demo")

    assert collections["demo"]["name"] == "demo"
    assert bundle.collection_spec["name"] == "demo"
    assert bundle.pip_threshold_plot_data.height == 1
    assert "method_family" in bundle.pip_threshold_plot_data.columns


def test_method_selection_helpers_follow_available_data():
    df = pl.DataFrame(
        {
            "method": ["twogroup_L1", "twogroup_L5"],
            "method_family": ["twogroup", "twogroup"],
            "L": [1, 5],
        }
    )

    assert viz3_utils.available_method_families(df) == ["twogroup"]
    assert viz3_utils.available_L_values(df) == [1, 5]
    assert viz3_utils.selected_method_names(
        df, selected_method_family="twogroup", selected_L=5
    ) == {"twogroup_L5"}


def test_available_thresholds_returns_sorted_non_null_values():
    df = pl.DataFrame({"threshold": [None, 1.0, 3.0, None, 2.0]}, schema={"threshold": pl.Float64})

    assert viz3_utils.available_thresholds(df) == [1.0, 2.0, 3.0]


def test_viz3_notebook_module_loads():
    globals_dict = runpy.run_path(str(Path("notebooks") / "viz3.py"), run_name="viz3_test")
    assert "app" in globals_dict


def test_viz3_notebook_waits_for_collection_selection():
    source = (Path("notebooks") / "viz3.py").read_text()

    assert "allow_select_none=True" in source
    assert "value=None" in source
    assert "mo.stop(" in source
    assert "collection_dropdown.value is None" in source


def test_pip_calibration_pipeline_produces_summary():
    method_spec = mock_method_spec("logistic_threshold_L1", L=1)
    data = pl.DataFrame(
        {
            "replicate": [0, 0, 1, 1],
            "method": ["logistic_threshold_L1"] * 4,
            "threshold": [1.0] * 4,
            "method_spec": [method_spec] * 4,
            "simulation_spec": ["sim"] * 4,
            "pip_threshold": [0.1, 0.2, 0.5, 0.6],
            "selected_total": [10, 10, 10, 10],
            "selected_causal": [1, 2, 5, 6],
            "power": [0.1, 0.2, 0.3, 0.4],
            "fdp": [0.0, 0.0, 0.0, 0.0],
            "n_exact": [10, 10, 10, 10],
            "n_causal_exact": [1, 2, 5, 6],
            "batch_hash": ["b1"] * 4,
            "batch_name": ["batch1"] * 4,
            "simulation_name": ["sim1"] * 4,
        }
    )

    enriched = viz3_utils.add_plot_metadata_columns(data)
    filtered = viz3_utils.filter_thresholded_methods(enriched, selected_threshold=1.0)
    labeled = viz3_utils.add_method_display_labels(filtered, selected_threshold=1.0)
    summary = viz3_utils.summarize_pip_calibration(labeled)

    assert "empirical_rate" in summary.columns
    assert summary.height > 0


def test_pip_calibration_bootstrap_adds_confidence_intervals():
    summary = pl.DataFrame(
        {
            "method_display": ["Logistic SER", "Logistic SER"],
            "series_label": ["Logistic SER", "Logistic SER"],
            "pip_bin_index": [0, 0],
            "pip_left": [0.0, 0.0],
            "pip_right": [0.05, 0.05],
            "pip_mid": [0.025, 0.025],
            "n_total": [10, 12],
            "n_causal": [1, 2],
        }
    )

    boot = viz3_utils.summarize_calibration_with_bootstrap(
        summary,
        group_cols=["method_display", "series_label"],
    )

    assert "ci_lower" in boot.columns
    assert "ci_upper" in boot.columns
    assert boot.height == 1


def test_power_fdp_summary_generation():
    data = pl.DataFrame(
        {
            "simulation_name": ["sim1"] * 3,
            "method": ["twogroup_L1"] * 3,
            "threshold": [None] * 3,
            "method_spec": [mock_method_spec("twogroup_L1", L=1)] * 3,
            "pip_threshold": [0.1, 0.2, 0.3],
            "power": [0.5, 0.6, 0.7],
            "fdp": [0.05, 0.10, 0.20],
            "is_thresholded": [False] * 3,
            "method_label_base": ["Twogroup SER"] * 3,
            "method_display": ["Twogroup SER"] * 3,
        }
    )

    prepared = viz3_utils.prepare_power_fdp_plot_data_frame(
        data,
        selected_threshold=1.0,
        selected_methods={"twogroup_L1"},
        show_background_threshold_traces=False,
    )
    summary = viz3_utils.make_power_fdp_summary(prepared)

    assert "trace_label" in prepared.columns
    assert "fdp" in summary.columns
    assert "power" in summary.columns


def test_causal_pip_summary_generation():
    data = pl.DataFrame(
        {
            "simulation_name": ["sim1", "sim1"],
            "method": ["logistic_threshold_L1", "logistic_threshold_L1"],
            "threshold": [1.0, 2.0],
            "causal_pip": [0.8, 0.9],
            "method_display": ["Logistic SER (@1)", "Logistic SER (@2)"],
        }
    )

    summary = viz3_utils.make_causal_pip_summary(data)

    assert "mean_causal_pip" in summary.columns
    assert summary.height == 2


def test_conditional_cs_summary_handles_empty_inputs():
    aggregate, by_sim = viz3_utils.make_conditional_cs_summary(
        viz3_utils.empty_cs_component_plot_data(),
        viz3_utils.empty_cs_truth_plot_data(),
        nominal_coverage=0.95,
        max_cs_size=10,
        min_ser_log_bf=2.0,
    )

    assert aggregate.is_empty()
    assert by_sim.is_empty()


def test_prepare_cs_histogram_data_handles_empty_input():
    size_df, bf_df = viz3_utils.prepare_cs_histogram_data(
        viz3_utils.empty_cs_component_plot_data(),
        nominal_coverage=0.95,
        selected_threshold=1.0,
    )

    assert size_df.is_empty()
    assert bf_df.is_empty()


def test_replicate_summary_bootstrap_handles_empty_input():
    result = viz3_utils.summarize_replicate_metric_with_bootstrap(
        pl.DataFrame(
            schema={
                "simulation_name": pl.String,
                "method": pl.String,
                "metric": pl.String,
                "value": pl.Float64,
            }
        ),
        group_cols=["simulation_name", "method", "metric"],
    )

    assert result.is_empty()
