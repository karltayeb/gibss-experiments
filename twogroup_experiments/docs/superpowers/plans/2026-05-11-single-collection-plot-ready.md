# Single-Collection Plot-Ready Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build collection-local `plot_ready` parquet outputs and a new Marimo notebook that renders collection-level plots without loading batch-level raw plot-data tables.

**Architecture:** Keep current simulation and fit rules unchanged, then add collection-level Snakemake rules that read `simulations.parquet` and `fits.parquet` directly and materialize one parquet per plot family under `results/by_alias/{collection_alias}/plot_ready/`. Use a new `plot_ready.py` helper module to compute metadata and plot-ready tables, and a new `notebooks/viz4_plot_ready.py` notebook that loads only collection-local metadata and plot-ready outputs. V1 ships point estimates only, but builder internals must preserve explicit per-`sample_id` summary steps so percentile bootstrap CIs can be added later without redesigning the pipeline.

**Tech Stack:** Python, Snakemake, Polars, NumPy, Marimo, Matplotlib, YAML, Pytest, `uv`

---

## File Structure

- `plot_ready.py`
  New helper module for collection-level metadata builders and plot-ready table builders. Keep one top-level function per output parquet, plus focused internal helpers for sample joins, method metadata extraction, and per-sample summaries.

- `twogroup_experiments.snk`
  Add one Snakemake rule per metadata/plot output under `results/by_alias/{collection_alias}/plot_ready/`. Keep old plot-data rules in place for migration; do not delete them in this plan.

- `notebooks/viz4_plot_ready.py`
  New Marimo notebook that reads only `collection_spec.yaml` and `plot_ready/*.parquet` from one collection alias. Reuse or reimplement only the minimal rendering helpers needed.

- `tests/test_plot_ready.py`
  New unit tests for metadata and plot-ready builders.

- `tests/test_twogroup_experiments.py`
  Extend workflow tests to cover new collection-level plot-ready outputs and new Snakemake rules.

## Output Contracts

Collection-local metadata files under `results/by_alias/{collection_alias}/plot_ready/`:

- `method_metadata.parquet`
  - columns: `method`, `threshold`, `method_spec`, `method_family`, `L`, `is_thresholded`, `is_oracle`, `oracle_label`, `method_label_base`, `method_display`
  - one row per effective method choice
  - non-thresholded methods use `threshold = null`

- `simulation_metadata.parquet`
  - columns: `batch_hash`, `batch_name`, `simulation_spec`, `simulation_name`

- `sample_metadata.parquet`
  - columns: `sample_id`, `batch_hash`, `batch_name`, `replicate`
  - `sample_id = "{batch_hash}::{replicate}"`

Collection-local plot-ready files under `results/by_alias/{collection_alias}/plot_ready/`:

- `pip_calibration.parquet`
  - columns: `method`, `threshold`, `pip_bin_index`, `pip_left`, `pip_right`, `pip_mid`, `n_total`, `n_causal`, `empirical_rate`

- `power_fdp.parquet`
  - columns: `method`, `threshold`, `pip_threshold`, `power`, `fdp`

- `causal_pip.parquet`
  - columns: `method`, `threshold`, `mean_causal_pip`

- `cs_summary.parquet`
  - columns: `method`, `threshold`, `metric`, `value`
  - metrics: `Power`, `CS Size`, `Coverage`

- `cs_size_histogram.parquet`
  - columns: `method`, `threshold`, `cs_size`
  - raw observations, not pre-binned counts

- `ser_log_bf_histogram.parquet`
  - columns: `method`, `threshold`, `ser_log_bf`
  - raw observations, not pre-binned counts

## Notebook Semantics

Notebook controls:

- `method_family` multiselect
- `L` dropdown
- threshold dropdown

Selection behavior:

- thresholded methods at selected threshold are foreground
- thresholded methods at other thresholds appear only as background traces in power/FDP
- all other plots use only selected-threshold rows for thresholded methods
- non-thresholded methods always remain visible

Threshold choices:

- derive from distinct non-null thresholds in `method_metadata.parquet`
- use all collection thresholds so control remains stable as other filters change

Notebook data access rule:

- load only from `results/by_alias/{collection_alias}/collection_spec.yaml`
- load only from `results/by_alias/{collection_alias}/plot_ready/*.parquet`
- do not read `results/by_batch/...`
- do not read old `batches/*/fits/*/*plot_data.parquet`

## Task 1: Add `plot_ready.py` Metadata Builders

**Files:**
- Create: `plot_ready.py`
- Test: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing tests for metadata contracts**

```python
from pathlib import Path

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
```

- [ ] **Step 2: Run metadata tests to verify failure**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k metadata -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'plot_ready'` or missing builder functions.

- [ ] **Step 3: Implement metadata builders in `plot_ready.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from viz3_utils import method_metadata_from_method_spec_json, make_method_display_label


def build_method_metadata(fits_df: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in (
        fits_df.select("method", "threshold", "method_spec")
        .unique()
        .sort("method", "threshold")
        .iter_rows(named=True)
    ):
        metadata = method_metadata_from_method_spec_json(row["method_spec"])
        rows.append(
            {
                "method": row["method"],
                "threshold": row["threshold"],
                "method_spec": row["method_spec"],
                **metadata,
                "method_display": make_method_display_label(
                    method_label_base=str(metadata["method_label_base"]),
                    threshold=row["threshold"],
                    is_thresholded=bool(metadata["is_thresholded"]),
                    is_oracle=bool(metadata["is_oracle"]),
                    oracle_label=str(metadata["oracle_label"]),
                ),
            }
        )
    return pl.from_dicts(rows)


def build_simulation_metadata(collection: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for batch in collection["batches"]:
        rows.append(
            {
                "batch_hash": batch["hash"],
                "batch_name": batch["name"],
                "simulation_spec": json.dumps(batch["simulation_spec"], sort_keys=True),
                "simulation_name": batch["simulation_spec"]["fields"]["name"],
            }
        )
    return pl.from_dicts(rows)


def build_sample_metadata(
    collection_batches: list[dict[str, Any]],
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    rows = []
    for batch in collection_batches:
        batch_hash = batch["hash"]
        batch_name = batch["name"]
        for replicate in simulations_by_batch[batch_hash]["replicate"].to_list():
            rows.append(
                {
                    "sample_id": f"{batch_hash}::{int(replicate)}",
                    "batch_hash": batch_hash,
                    "batch_name": batch_name,
                    "replicate": int(replicate),
                }
            )
    return pl.from_dicts(rows)
```

- [ ] **Step 4: Run metadata tests to verify pass**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k metadata -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plot_ready.py tests/test_plot_ready.py
git commit -m "feat: add plot-ready metadata builders"
```

## Task 2: Build PIP Calibration And Power/FDP Plot-Ready Tables

**Files:**
- Modify: `plot_ready.py`
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing tests for PIP calibration and power/FDP outputs**

```python
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

    assert result.rows(named=True) == [
        {
            "method": "logistic_threshold_L1",
            "threshold": 1.0,
            "pip_threshold": 0.5,
            "power": 0.3,
            "fdp": 0.2,
        }
    ]
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k "pip_calibration or power_fdp" -q
```

Expected: FAIL with missing aggregate builder functions.

- [ ] **Step 3: Implement per-sample summaries and collection aggregations**

```python
def summarize_pip_calibration_per_sample(fits_df: pl.DataFrame, sample_metadata: pl.DataFrame) -> pl.DataFrame:
    # Build one row per sample_id x method x threshold x pip_bin_index.
    ...


def aggregate_pip_calibration(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold", "pip_bin_index")
        .agg(
            pl.col("n_exact").sum().alias("n_total"),
            pl.col("n_causal_exact").sum().alias("n_causal"),
        )
        .with_columns(
            (pl.col("pip_bin_index") * 0.05).alias("pip_left"),
            ((pl.col("pip_bin_index") + 1) * 0.05).alias("pip_right"),
            ((pl.col("pip_bin_index") + 0.5) * 0.05).alias("pip_mid"),
            (pl.col("n_causal") / pl.col("n_total")).alias("empirical_rate"),
        )
        .select(
            "method",
            "threshold",
            "pip_bin_index",
            "pip_left",
            "pip_right",
            "pip_mid",
            "n_total",
            "n_causal",
            "empirical_rate",
        )
        .sort("method", "threshold", "pip_bin_index")
    )


def summarize_power_fdp_per_sample(fits_df: pl.DataFrame, sample_metadata: pl.DataFrame) -> pl.DataFrame:
    # Build one row per sample_id x method x threshold x pip_threshold.
    ...


def aggregate_power_fdp(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold", "pip_threshold")
        .agg(
            pl.col("power").mean().alias("power"),
            pl.col("fdp").mean().alias("fdp"),
        )
        .sort("method", "threshold", "pip_threshold")
    )
```

- [ ] **Step 4: Run targeted tests to verify pass**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k "pip_calibration or power_fdp" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plot_ready.py tests/test_plot_ready.py
git commit -m "feat: add pip calibration and power fdp plot-ready builders"
```

## Task 3: Build Causal PIP And CS Summary Plot-Ready Tables

**Files:**
- Modify: `plot_ready.py`
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing tests for causal PIP and CS summary outputs**

```python
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
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k "causal_pip or cs_summary" -q
```

Expected: FAIL with missing builder functions.

- [ ] **Step 3: Implement causal PIP and CS summary builders**

```python
def summarize_causal_pip_per_sample(fits_df: pl.DataFrame, sample_metadata: pl.DataFrame) -> pl.DataFrame:
    # Build one row per sample_id x method x threshold with sample-level mean causal PIP.
    ...


def aggregate_causal_pip(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold")
        .agg(pl.col("mean_causal_pip").mean().alias("mean_causal_pip"))
        .sort("method", "threshold")
    )


def summarize_cs_metrics_per_sample(fits_df: pl.DataFrame, sample_metadata: pl.DataFrame) -> pl.DataFrame:
    # Build one row per sample_id x method x threshold x metric.
    ...


def aggregate_cs_summary(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold", "metric")
        .agg(pl.col("value").mean().alias("value"))
        .sort("method", "threshold", "metric")
    )
```

- [ ] **Step 4: Run targeted tests to verify pass**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k "causal_pip or cs_summary" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plot_ready.py tests/test_plot_ready.py
git commit -m "feat: add causal pip and cs summary plot-ready builders"
```

## Task 4: Build Histogram Observation Tables

**Files:**
- Modify: `plot_ready.py`
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing tests for raw histogram observation outputs**

```python
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
```

- [ ] **Step 2: Run histogram tests to verify failure**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k histogram -q
```

Expected: FAIL with missing histogram finalizer functions.

- [ ] **Step 3: Implement histogram observation builders**

```python
def summarize_cs_size_histogram_observations(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    # Build raw method x threshold x cs_size observations.
    ...


def finalize_cs_size_histogram(observations: pl.DataFrame) -> pl.DataFrame:
    return observations.select("method", "threshold", "cs_size").sort(
        "method", "threshold", "cs_size"
    )


def summarize_ser_log_bf_histogram_observations(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    # Build raw method x threshold x ser_log_bf observations.
    ...


def finalize_ser_log_bf_histogram(observations: pl.DataFrame) -> pl.DataFrame:
    return observations.select("method", "threshold", "ser_log_bf").sort(
        "method", "threshold", "ser_log_bf"
    )
```

- [ ] **Step 4: Run histogram tests to verify pass**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k histogram -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plot_ready.py tests/test_plot_ready.py
git commit -m "feat: add histogram plot-ready builders"
```

## Task 5: Add Collection-Level Snakemake Rules

**Files:**
- Modify: `twogroup_experiments.snk`
- Modify: `tests/test_twogroup_experiments.py`
- Test: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing workflow tests for new plot-ready outputs**

```python
def test_collection_plot_ready_outputs_are_declared():
    expected = {
        "method_metadata.parquet",
        "simulation_metadata.parquet",
        "sample_metadata.parquet",
        "pip_calibration.parquet",
        "power_fdp.parquet",
        "causal_pip.parquet",
        "cs_summary.parquet",
        "cs_size_histogram.parquet",
        "ser_log_bf_histogram.parquet",
    }

    from pathlib import Path

    plot_ready_files = {path.name for path in Path("results/by_alias/demo/plot_ready").glob("*.parquet")}
    assert expected.issubset(plot_ready_files)
```

- [ ] **Step 2: Run workflow test to verify failure**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_twogroup_experiments.py -k plot_ready -q
```

Expected: FAIL because new outputs and rules do not exist.

- [ ] **Step 3: Add one rule per collection-local output in `twogroup_experiments.snk`**

```python
rule collection_method_metadata:
    input:
        fits=lambda wildcards: [
            f"{RESULTS_ROOT}/by_batch/{batch[HASH_KEY]}/fits/{method_spec[HASH_KEY]}/fits.parquet"
            for batch in config["collections"][wildcards.collection_alias]["batches"]
            for method_spec in config["collections"][wildcards.collection_alias]["method_specs"]
        ]
    output:
        f"{RESULTS_ROOT}/by_alias/{{collection_alias}}/plot_ready/method_metadata.parquet"
    run:
        collection = config["collections"][wildcards.collection_alias]
        fits_df = load_collection_fits_with_specs(collection)
        write_parquet(build_method_metadata(fits_df), output[0])


rule collection_pip_calibration_plot_ready:
    input:
        simulations=...,
        fits=...,
    output:
        f"{RESULTS_ROOT}/by_alias/{{collection_alias}}/plot_ready/pip_calibration.parquet"
    run:
        collection = config["collections"][wildcards.collection_alias]
        fits_df = load_collection_fits_with_specs(collection)
        simulations_by_batch = load_collection_simulations(collection)
        sample_metadata = build_sample_metadata(collection["batches"], simulations_by_batch)
        per_sample = summarize_pip_calibration_per_sample(fits_df, sample_metadata)
        write_parquet(aggregate_pip_calibration(per_sample), output[0])
```

Required rule set:

```python
collection_method_metadata
collection_simulation_metadata
collection_sample_metadata
collection_pip_calibration_plot_ready
collection_power_fdp_plot_ready
collection_causal_pip_plot_ready
collection_cs_summary_plot_ready
collection_cs_size_histogram_plot_ready
collection_ser_log_bf_histogram_plot_ready
```

Update `materialize_twogroup_experiment_collection_alias` input or the final target rule so all new `plot_ready/*.parquet` files are built for every collection alias.

- [ ] **Step 4: Run workflow tests to verify pass**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_twogroup_experiments.py -k plot_ready -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments.snk tests/test_twogroup_experiments.py plot_ready.py tests/test_plot_ready.py
git commit -m "feat: add collection-level plot-ready snakemake rules"
```

## Task 6: Create New Plot-Ready Notebook

**Files:**
- Create: `notebooks/viz4_plot_ready.py`
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing notebook smoke test**

```python
def test_viz4_plot_ready_notebook_module_loads():
    import runpy
    from pathlib import Path

    globals_dict = runpy.run_path(
        str(Path("notebooks") / "viz4_plot_ready.py"),
        run_name="viz4_plot_ready_test",
    )
    assert "app" in globals_dict
```

- [ ] **Step 2: Run notebook smoke test to verify failure**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k viz4_plot_ready_notebook_module_loads -q
```

Expected: FAIL because notebook file does not exist.

- [ ] **Step 3: Implement new notebook that reads only collection-local plot-ready files**

```python
import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl

    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    import plot_ready
    import viz3_utils


@app.cell
def collection_selector_cell():
    collection_alias_root = Path(__file__).parent.parent / "results" / "by_alias"
    collections = plot_ready.available_plot_ready_collections(collection_alias_root)
    collection_dropdown = mo.ui.dropdown(
        options=collections,
        value=None,
        allow_select_none=True,
        label="collection",
    )
    return collection_alias_root, collection_dropdown


@app.cell
def collection_bundle_cell(collection_alias_root, collection_dropdown):
    mo.stop(
        collection_dropdown.value is None,
        mo.md("Select a collection to load plot-ready data."),
    )
    return (
        plot_ready.load_plot_ready_collection(
            collection_alias_root / collection_dropdown.value
        ),
    )
```

Notebook must:
- load only `results/by_alias/{alias}/plot_ready/*.parquet`
- derive threshold choices from `method_metadata.parquet`
- preserve current threshold highlighting behavior in power/FDP
- not read old batch-level plot-data parquet files

- [ ] **Step 4: Run notebook smoke test and Marimo check**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -k viz4_plot_ready_notebook_module_loads -q
uvx marimo check notebooks/viz4_plot_ready.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/viz4_plot_ready.py tests/test_plot_ready.py plot_ready.py
git commit -m "feat: add plot-ready marimo notebook"
```

## Task 7: Full Verification

**Files:**
- Modify: any touched files from prior tasks
- Test: `tests/test_plot_ready.py`
- Test: `tests/test_twogroup_experiments.py`

- [ ] **Step 1: Run full plot-ready unit test suite**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_plot_ready.py -q
```

Expected: PASS.

- [ ] **Step 2: Run relevant workflow tests**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_twogroup_experiments.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Marimo notebook validation**

Run:

```bash
uvx marimo check notebooks/viz4_plot_ready.py
```

Expected: exit code 0.

- [ ] **Step 4: Re-read plan against implementation and fix any gaps**

Checklist:
- all 3 metadata parquets created under `plot_ready/`
- all 6 plot-ready parquets created under `plot_ready/`
- notebook loads only collection-local files
- no CI columns added in v1
- per-sample summary helpers exist for later bootstrap extension

- [ ] **Step 5: Commit final integration**

```bash
git add plot_ready.py twogroup_experiments.snk notebooks/viz4_plot_ready.py tests/test_plot_ready.py tests/test_twogroup_experiments.py
git commit -m "feat: add single-collection plot-ready pipeline"
```

## Self-Review

- Spec coverage: covers collection-local metadata, all agreed plot-ready outputs, per-plot Snakemake rule split, new notebook, and future bootstrap-ready internal structure.
- Placeholder scan: implementation snippets marked `...` appear only inside plan-local code examples describing helper internals; before execution, replace each with concrete code in the task implementation itself.
- Type consistency: all final output names and schemas match agreed contracts:
  - `method_metadata.parquet`
  - `simulation_metadata.parquet`
  - `sample_metadata.parquet`
  - `pip_calibration.parquet`
  - `power_fdp.parquet`
  - `causal_pip.parquet`
  - `cs_summary.parquet`
  - `cs_size_histogram.parquet`
  - `ser_log_bf_histogram.parquet`
