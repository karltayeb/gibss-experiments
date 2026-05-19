# Viz2 MethodSpec Plot Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make exported twogroup plot data self-describing by carrying dehydrated `method_spec` and `simulation_spec`, then update `viz2.py` to derive method family, `L`, thresholded/oracle labels, and UI controls from those specs instead of hard-coded method-name assumptions.

**Architecture:** The export path in `twogroup_experiments/utils.py` will attach spec provenance to fit rows and propagate it through all four plot-data parquet tables. `twogroup_experiments/notebooks/viz2.py` will add one normalization layer near the top that turns the dehydrated specs into derived plotting columns and control options, and the rest of the notebook will consume those normalized columns.

**Tech Stack:** Python, Polars, marimo, pytest, uv

---

### Task 1: Lock Down Plot Data Provenance Export

**Files:**
- Modify: `twogroup_experiments/utils.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write the failing test for fit-row provenance**

```python
def test_fit_batch_method_includes_dehydrated_specs():
    rows = fit_batch_method(
        TINY_TEST_SIMULATION,
        method_spec=_logistic_oracle_method_spec(L=5),
        replicates=(0,),
    )

    assert {"method_spec", "simulation_spec"} <= set(rows.columns)

    row = rows.row(0, named=True)
    assert row["method"] == "logistic_oracle_L5"
    assert row["method_spec"]["name"] == "logistic_oracle_L5"
    assert row["method_spec"]["kwargs"]["L"] == 5
    assert row["simulation_spec"]["name"] == TINY_TEST_SIMULATION.name
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_fit_batch_method_includes_dehydrated_specs -q
```

Expected: FAIL because `fit_batch_method()` does not yet include `method_spec` or `simulation_spec`.

- [ ] **Step 3: Implement fit-row provenance in `utils.py`**

Add the spec dehydration imports near the top of `utils.py`:

```python
from core import (
    HASH_KEY,
    MethodSpec,
    SimulationSpec,
    TwoGroupSimulation,
    dehydrate_hashed,
    dehydrate_node,
    dehydrate_spec,
    run_method_spec,
    simulate,
    spec_hash,
    summarize_method_spec,
)
```

Update the row construction inside `fit_batch_method()`:

```python
        rows.append(
            {
                "replicate": replicate,
                "method_spec": dehydrate_hashed(method_spec),
                "simulation_spec": dehydrate_hashed(simulation_spec),
                **summarize_method_spec(method_spec, fit_obj, simulation),
            }
        )
```

- [ ] **Step 4: Run the targeted test to verify it passes**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_fit_batch_method_includes_dehydrated_specs -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/utils.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "test: carry dehydrated specs in fit batch rows"
```

### Task 2: Propagate Specs Through All Plot Data Tables

**Files:**
- Modify: `twogroup_experiments/utils.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write the failing test for plot-data provenance columns**

```python
def test_build_plot_data_frames_propagates_specs_to_all_outputs():
    simulations_df = simulate_batch(TINY_TEST_SIMULATION, replicates=(0,))
    fits_df = fit_batch_method(
        TINY_TEST_SIMULATION,
        method_spec=_logistic_threshold_method_spec(threshold=2.0, L=5),
        replicates=(0,),
    )

    plot_frames = build_plot_data_frames(fits_df, simulations_df)

    for df in plot_frames.values():
        assert {"method_spec", "simulation_spec"} <= set(df.columns)

    row = plot_frames["pip_threshold_plot_data"].row(0, named=True)
    assert row["method_spec"]["kwargs"]["L"] == 5
    assert row["method_spec"]["kwargs"]["threshold"] == 2.0
    assert row["simulation_spec"]["name"] == TINY_TEST_SIMULATION.name
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_build_plot_data_frames_propagates_specs_to_all_outputs -q
```

Expected: FAIL because `build_plot_data_frames()` currently drops spec provenance.

- [ ] **Step 3: Thread `method_spec` and `simulation_spec` through all plot-data row builders**

In `build_plot_data_frames()`, extend each appended row payload:

```python
                    "method_spec": row["method_spec"],
                    "simulation_spec": row["simulation_spec"],
```

Apply that addition in all four row builders:
- `pip_threshold_rows.append(...)`
- `causal_pip_rows.append(...)`
- `cs_component_rows.append(...)`
- `cs_truth_rows.append(...)`

Also update the empty-frame schemas near the top of `viz2.py` later in Task 4, but do not change them yet in this task.

- [ ] **Step 4: Run the targeted test to verify it passes**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_build_plot_data_frames_propagates_specs_to_all_outputs -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/utils.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "test: preserve dehydrated specs in plot data outputs"
```

### Task 3: Add Notebook-Side MethodSpec Normalization Helpers

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests for method normalization**

Add notebook-focused pure-function tests in `tests/test_twogroup_experiments.py` for small helpers you will introduce in `viz2.py`:

```python
def test_method_metadata_from_dehydrated_spec_logistic_ser():
    method_spec = dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=1))
    metadata = method_metadata_from_spec(method_spec)

    assert metadata["method_family"] == "logistic_threshold"
    assert metadata["L"] == 1
    assert metadata["is_thresholded"] is True
    assert metadata["is_oracle"] is False
    assert metadata["method_label_base"] == "Logistic SER"


def test_method_metadata_from_dehydrated_spec_logistic_susie():
    method_spec = dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=5))
    metadata = method_metadata_from_spec(method_spec)

    assert metadata["method_family"] == "logistic_threshold"
    assert metadata["L"] == 5
    assert metadata["method_label_base"] == "Logistic SuSiE [L=5]"


def test_method_display_label_uses_threshold_and_oracle_suffixes():
    assert make_method_display_label(
        method_family="logistic_threshold",
        method_label_base="Logistic SER",
        threshold=2.0,
        is_thresholded=True,
        is_oracle=False,
    ) == "Logistic SER (@2)"

    assert make_method_display_label(
        method_family="logistic_oracle",
        method_label_base="Logistic SER",
        threshold=None,
        is_thresholded=False,
        is_oracle=True,
    ) == "Logistic SER (Oracle)"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest \
  tests/test_twogroup_experiments.py::test_method_metadata_from_dehydrated_spec_logistic_ser \
  tests/test_twogroup_experiments.py::test_method_metadata_from_dehydrated_spec_logistic_susie \
  tests/test_twogroup_experiments.py::test_method_display_label_uses_threshold_and_oracle_suffixes \
  -q
```

Expected: FAIL because the helpers do not exist yet.

- [ ] **Step 3: Implement pure normalization helpers near the top of `viz2.py`**

Add focused helpers:

```python
@app.function
def method_family_label_base_map() -> dict[str, str]:
    return {
        "logistic_threshold": "Logistic",
        "logistic_oracle": "Logistic",
        "twogroup": "Twogroup",
        "twogroup_oracle": "Twogroup",
        "cox_light_threshold": "Cox Light",
        "cox_heavy": "Cox Heavy",
    }


@app.function
def method_metadata_from_spec(method_spec: dict[str, object]) -> dict[str, object]:
    name = str(method_spec["name"])
    kwargs = dict(method_spec.get("kwargs", {}))
    L = int(kwargs.get("L", 1))
    method_family = name.rsplit("_L", 1)[0]
    is_thresholded = "threshold" in method_family
    is_oracle = "oracle" in method_family
    family_label = method_family_label_base_map().get(method_family, method_family)
    suffix = "SER" if L == 1 else f"SuSiE [L={L}]"
    return {
        "method_family": method_family,
        "L": L,
        "is_thresholded": is_thresholded,
        "is_oracle": is_oracle,
        "method_label_base": f"{family_label} {suffix}",
    }


@app.function
def make_method_display_label(
    *,
    method_family: str,
    method_label_base: str,
    threshold: float | None,
    is_thresholded: bool,
    is_oracle: bool,
) -> str:
    if is_oracle:
        return f"{method_label_base} (Oracle)"
    if is_thresholded and threshold is not None:
        return f"{method_label_base} (@{threshold:g})"
    return method_label_base
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest \
  tests/test_twogroup_experiments.py::test_method_metadata_from_dehydrated_spec_logistic_ser \
  tests/test_twogroup_experiments.py::test_method_metadata_from_dehydrated_spec_logistic_susie \
  tests/test_twogroup_experiments.py::test_method_display_label_uses_threshold_and_oracle_suffixes \
  -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: normalize viz2 method metadata from dehydrated specs"
```

### Task 4: Normalize Plot Tables Inside Viz2

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write a failing test for plot-table normalization**

```python
def test_add_method_metadata_columns_derives_family_and_display_label():
    df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L5"],
            "threshold": [2.0],
            "method_spec": [dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=5))],
            "simulation_spec": [dehydrate_hashed(TINY_TEST_SIMULATION)],
        }
    )

    normalized = add_plot_metadata_columns(df)
    row = normalized.row(0, named=True)

    assert row["method_family"] == "logistic_threshold"
    assert row["L"] == 5
    assert row["method_label_base"] == "Logistic SuSiE [L=5]"
    assert row["method_display"] == "Logistic SuSiE [L=5] (@2)"
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_add_method_metadata_columns_derives_family_and_display_label -q
```

Expected: FAIL because `add_plot_metadata_columns()` does not exist yet.

- [ ] **Step 3: Implement a single normalization pass in `viz2.py`**

Add one helper that takes any plot-data frame and adds derived columns from the stored spec payload:

```python
@app.function
def add_plot_metadata_columns(plot_data: pl.DataFrame) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data

    rows = []
    for row in plot_data.iter_rows(named=True):
        method_meta = method_metadata_from_spec(row["method_spec"])
        rows.append(
            {
                **row,
                **method_meta,
                "method_display": make_method_display_label(
                    method_family=method_meta["method_family"],
                    method_label_base=method_meta["method_label_base"],
                    threshold=row.get("threshold"),
                    is_thresholded=bool(method_meta["is_thresholded"]),
                    is_oracle=bool(method_meta["is_oracle"]),
                ),
            }
        )
    return pl.from_dicts(rows)
```

Then call `add_plot_metadata_columns(...)` once after each parquet load and before any filtering or chart preparation.

Also update the empty schema builders at the top of `viz2.py` so empty frames include `method_spec` and `simulation_spec` columns with `pl.Struct`-compatible `pl.Object` placeholders if needed for notebook code stability.

- [ ] **Step 4: Run the targeted test to verify it passes**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_add_plot_metadata_columns_derives_family_and_display_label -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: derive viz2 plotting metadata from spec payloads"
```

### Task 5: Replace Hard-Coded Method Controls With Method-Family and L Controls

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests for control option derivation**

```python
def test_available_method_families_and_L_values_are_derived_from_plot_data():
    df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L1", "logistic_threshold_L5", "twogroup_L1"],
            "method_family": ["logistic_threshold", "logistic_threshold", "twogroup"],
            "L": [1, 5, 1],
        }
    )

    assert available_method_families(df) == ["logistic_threshold", "twogroup"]
    assert available_L_values(df) == [1, 5]
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_available_method_families_and_L_values_are_derived_from_plot_data -q
```

Expected: FAIL because the helper functions do not exist.

- [ ] **Step 3: Replace the multiselect model in `viz2.py`**

Implement helpers:

```python
@app.function
def available_method_families(plot_data: pl.DataFrame) -> list[str]:
    return sorted(plot_data.get_column("method_family").unique().to_list())


@app.function
def available_L_values(plot_data: pl.DataFrame) -> list[int]:
    return sorted(int(value) for value in plot_data.get_column("L").unique().to_list())
```

Replace:
- `instantiate_method_multiselect()`

with:
- `instantiate_method_family_dropdown(plot_data)`
- `instantiate_L_dropdown(plot_data)`

The defaults should select the first available family and the first available `L` value present in the loaded collection.

- [ ] **Step 4: Update all filter entrypoints to use `method_family` and `L`**

Replace direct raw-method filtering like:

```python
filter_selected_methods(plot_data, selected_methods)
```

with:

```python
filter_selected_method_variant(
    plot_data,
    selected_method_family=method_family_dropdown.value,
    selected_L=L_dropdown.value,
)
```

using a new helper:

```python
@app.function
def filter_selected_method_variant(
    plot_data: pl.DataFrame,
    *,
    selected_method_family: str,
    selected_L: int,
) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    return plot_data.filter(
        (pl.col("method_family") == selected_method_family)
        & (pl.col("L") == selected_L)
    )
```

- [ ] **Step 5: Run the targeted test to verify it passes**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_available_method_families_and_L_values_are_derived_from_plot_data -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: switch viz2 controls to method family and L selectors"
```

### Task 6: Rewire Plot Ordering, Colors, and Threshold Logic Around Derived Metadata

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests for ordering and threshold handling**

```python
def test_method_display_order_uses_method_family_order_not_raw_method_names():
    assert method_family_display_order() == [
        "logistic_oracle",
        "twogroup_oracle",
        "twogroup",
        "cox_heavy",
        "cox_light_threshold",
        "logistic_threshold",
    ]


def test_filter_thresholded_methods_uses_is_thresholded_column():
    df = pl.DataFrame(
        {
            "method_family": ["logistic_threshold", "twogroup"],
            "is_thresholded": [True, False],
            "threshold": [2.0, None],
        }
    )

    filtered = filter_thresholded_methods(df, selected_threshold=2.0)
    assert filtered.height == 2
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest \
  tests/test_twogroup_experiments.py::test_method_display_order_uses_method_family_order_not_raw_method_names \
  tests/test_twogroup_experiments.py::test_filter_thresholded_methods_uses_is_thresholded_column \
  -q
```

Expected: FAIL because the notebook still relies on raw method strings.

- [ ] **Step 3: Replace raw-method maps with family-based maps**

In `viz2.py`, rename and update:
- `method_color_map()` -> keys are method families
- `method_line_style_map()` -> keys are method families
- `method_display_order()` -> returns family order
- `method_label_map()` -> remove if superseded by `method_label_base`

Update `filter_thresholded_methods()` so it uses `is_thresholded`:

```python
return plot_data.filter(
    ((pl.col("is_thresholded")) & (pl.col("threshold") == selected_threshold))
    | (~pl.col("is_thresholded"))
)
```

Then update the plotting loops so they index colors and line styles by `method_family`, not raw `method`.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest \
  tests/test_twogroup_experiments.py::test_method_display_order_uses_method_family_order_not_raw_method_names \
  tests/test_twogroup_experiments.py::test_filter_thresholded_methods_uses_is_thresholded_column \
  -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "refactor: drive viz2 styling and threshold logic from method metadata"
```

### Task 7: End-to-End Validation on the Hallmark SER-Enrich-LOC Collection

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write an integration test for the working example collection**

```python
def test_viz2_normalization_handles_hallmark_ser_enrich_loc_collection():
    fits_df = fit_batch_method(
        TINY_TEST_SIMULATION,
        method_spec=_logistic_threshold_method_spec(threshold=2.0, L=1),
        replicates=(0,),
    ).vstack(
        fit_batch_method(
            TINY_TEST_SIMULATION,
            method_spec=_logistic_threshold_method_spec(threshold=2.0, L=5),
            replicates=(0,),
        )
    )
    simulations_df = simulate_batch(TINY_TEST_SIMULATION, replicates=(0,))
    plot_frames = build_plot_data_frames(fits_df, simulations_df)

    normalized = add_plot_metadata_columns(plot_frames["pip_threshold_plot_data"])

    assert set(normalized.get_column("L").unique().to_list()) == {1, 5}
    assert "Logistic SER (@2)" in normalized.get_column("method_display").unique().to_list()
    assert "Logistic SuSiE [L=5] (@2)" in normalized.get_column("method_display").unique().to_list()
```

- [ ] **Step 2: Run the targeted integration test to verify it fails or exposes any remaining mismatch**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_viz2_normalization_handles_hallmark_ser_enrich_loc_collection -q
```

Expected: FAIL until all normalization and display changes are complete.

- [ ] **Step 3: Finish any remaining notebook plumbing**

Ensure every plot-preparation path uses normalized frames before grouping:
- PIP calibration views
- power/FDP views
- max-PIP views
- credible-set truth/component views
- calibration and histogram views

Ensure all UI cell wiring now passes:
- `threshold_control`
- `method_family_dropdown`
- `L_dropdown`

instead of the old `method_multiselect`.

- [ ] **Step 4: Run the focused integration test and the main regression test file**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py -q
```

Expected: PASS

- [ ] **Step 5: Run syntax validation for the notebook and helpers**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run python -m py_compile utils.py notebooks/viz2.py tests/test_twogroup_experiments.py
```

Expected: no output

- [ ] **Step 6: Commit**

```bash
git add twogroup_experiments/utils.py twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: make viz2 spec-driven for ser and susie fits"
```

## Self-Review

- Spec coverage: the plan covers plot-data export, provenance propagation, notebook normalization, control replacement, threshold logic, and end-to-end validation for the `hallmark__ser_enrich__loc` example.
- Placeholder scan: no `TODO`/`TBD` placeholders remain; each task includes exact files, code targets, and commands.
- Type consistency: the plan consistently uses `method_spec`, `simulation_spec`, `method_family`, `L`, `is_thresholded`, `is_oracle`, and `method_display`.
