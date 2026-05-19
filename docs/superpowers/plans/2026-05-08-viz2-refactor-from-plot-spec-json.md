# Viz2 Refactor From Plot Spec JSON Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `twogroup_experiments/notebooks/viz2.py` so it derives method and simulation metadata from `method_spec` / `simulation_spec` JSON columns in plot data, replacing the old hard-coded method-name logic.

**Architecture:** Keep the plot-data files as the reporting boundary: `viz2.py` should parse the stored spec JSON once, attach normalized metadata columns to each loaded frame, and then drive all controls, labels, ordering, and filtering from those derived columns. The notebook should no longer depend on exact raw method names like `logistic_threshold` or assume a single fit family.

**Tech Stack:** Python, marimo, Polars, JSON, matplotlib, uv

---

### Task 1: Add JSON-Spec Normalization Helpers At The Top Of Viz2

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests for method-spec parsing**

Add pure-function tests in `twogroup_experiments/tests/test_twogroup_experiments.py`:

```python
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
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'method_metadata_from_method_spec_json' -q
```

Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Implement pure JSON parsing helpers in `viz2.py`**

Add focused helpers near the top of `twogroup_experiments/notebooks/viz2.py`:

```python
@app.function
def method_family_label_map() -> dict[str, str]:
    return {
        "logistic_threshold": "Logistic",
        "logistic_oracle": "Logistic",
        "twogroup": "Twogroup",
        "twogroup_oracle": "Twogroup",
        "cox_light_threshold": "Cox Light",
        "cox_heavy": "Cox Heavy",
    }


@app.function
def method_metadata_from_method_spec_json(method_spec_json: str) -> dict[str, object]:
    method_spec = json.loads(method_spec_json)
    name = str(method_spec["fields"]["name"])
    kwargs = dict(method_spec["fields"].get("kwargs", {}))
    L = int(kwargs.get("L", 1))
    method_family = name.rsplit("_L", 1)[0]
    is_thresholded = "threshold" in method_family
    is_oracle = "oracle" in method_family
    family_label = method_family_label_map().get(method_family, method_family)
    suffix = "SER" if L == 1 else f"SuSiE [L={L}]"
    return {
        "method_family": method_family,
        "L": L,
        "is_thresholded": is_thresholded,
        "is_oracle": is_oracle,
        "method_label_base": f"{family_label} {suffix}",
    }
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'method_metadata_from_method_spec_json' -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: parse viz2 method metadata from spec json"
```

### Task 2: Normalize Loaded Plot Frames Once

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write a failing test for frame normalization**

```python
def test_add_plot_metadata_columns_uses_method_spec_json():
    method_spec_json = json.dumps(
        dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=5)),
        sort_keys=True,
    )
    simulation_spec_json = json.dumps(
        dehydrate_hashed(_tiny_simulation_spec()),
        sort_keys=True,
    )
    df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L5"],
            "threshold": [2.0],
            "method_spec": [method_spec_json],
            "simulation_spec": [simulation_spec_json],
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
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'add_plot_metadata_columns_uses_method_spec_json' -q
```

Expected: FAIL because the normalization helper does not exist.

- [ ] **Step 3: Implement one normalization pass in `viz2.py`**

Add a helper that parses `method_spec` / `simulation_spec` JSON and attaches derived columns:

```python
@app.function
def make_method_display_label(
    *,
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


@app.function
def add_plot_metadata_columns(plot_data: pl.DataFrame) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data

    rows = []
    for row in plot_data.iter_rows(named=True):
        method_meta = method_metadata_from_method_spec_json(row["method_spec"])
        rows.append(
            {
                **row,
                **method_meta,
                "method_display": make_method_display_label(
                    method_label_base=str(method_meta["method_label_base"]),
                    threshold=row.get("threshold"),
                    is_thresholded=bool(method_meta["is_thresholded"]),
                    is_oracle=bool(method_meta["is_oracle"]),
                ),
            }
        )
    return pl.from_dicts(rows)
```

Then make the notebook apply `add_plot_metadata_columns(...)` right after each plot-data parquet load.

- [ ] **Step 4: Run the targeted test to verify it passes**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'add_plot_metadata_columns_uses_method_spec_json' -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: normalize viz2 frames from spec json"
```

### Task 3: Replace Hard-Coded Method Maps With Metadata-Driven Maps

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests for display ordering and threshold logic**

```python
def test_method_family_display_order_is_family_based():
    assert method_family_display_order() == [
        "logistic_oracle",
        "twogroup_oracle",
        "twogroup",
        "cox_heavy",
        "cox_light_threshold",
        "logistic_threshold",
    ]


def test_filter_thresholded_methods_uses_is_thresholded():
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
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'method_family_display_order_is_family_based or filter_thresholded_methods_uses_is_thresholded' -q
```

Expected: FAIL because `viz2.py` still uses raw method-name assumptions.

- [ ] **Step 3: Refactor color/order/filter helpers**

Replace the current raw-method helpers with family-based helpers:

```python
@app.function
def method_color_map() -> dict[str, str]:
    return {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }


@app.function
def method_family_display_order() -> list[str]:
    return [
        "logistic_oracle",
        "twogroup_oracle",
        "twogroup",
        "cox_heavy",
        "cox_light_threshold",
        "logistic_threshold",
    ]
```

Update `filter_thresholded_methods()` to use `is_thresholded` rather than exact raw method strings:

```python
return plot_data.filter(
    ((pl.col("is_thresholded")) & (pl.col("threshold") == selected_threshold))
    | (~pl.col("is_thresholded"))
)
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'method_family_display_order_is_family_based or filter_thresholded_methods_uses_is_thresholded' -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "refactor: drive viz2 ordering and threshold logic from metadata"
```

### Task 4: Replace The Method Multiselect With Method-Family And L Controls

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests for control option derivation**

```python
def test_available_method_families_and_L_values_are_data_driven():
    df = pl.DataFrame(
        {
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
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'available_method_families_and_L_values_are_data_driven' -q
```

Expected: FAIL because those helpers do not exist.

- [ ] **Step 3: Add data-driven controls and filtering**

Implement helpers:

```python
@app.function
def available_method_families(plot_data: pl.DataFrame) -> list[str]:
    return sorted(plot_data.get_column("method_family").unique().to_list())


@app.function
def available_L_values(plot_data: pl.DataFrame) -> list[int]:
    return sorted(int(value) for value in plot_data.get_column("L").unique().to_list())


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

Replace `instantiate_method_multiselect()` and all downstream consumers with:
- `instantiate_method_family_dropdown(plot_data)`
- `instantiate_L_dropdown(plot_data)`

- [ ] **Step 4: Run the targeted test to verify it passes**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'available_method_families_and_L_values_are_data_driven' -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: replace viz2 method multiselect with family and L controls"
```

### Task 5: Rewire Plotting Call Sites To Use Normalized Columns

**Files:**
- Modify: `twogroup_experiments/notebooks/viz2.py`
- Modify: `twogroup_experiments/tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write a focused integration test for the hallmark working example**

```python
def test_hallmark_ser_enrich_loc_metadata_yields_ser_and_susie_labels():
    method_spec_ser = json.dumps(
        dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=1)),
        sort_keys=True,
    )
    method_spec_susie = json.dumps(
        dehydrate_hashed(_logistic_threshold_method_spec(threshold=2.0, L=5)),
        sort_keys=True,
    )
    simulation_spec_json = json.dumps(
        dehydrate_hashed(_tiny_simulation_spec()),
        sort_keys=True,
    )
    df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L1", "logistic_threshold_L5"],
            "threshold": [2.0, 2.0],
            "method_spec": [method_spec_ser, method_spec_susie],
            "simulation_spec": [simulation_spec_json, simulation_spec_json],
        }
    )

    normalized = add_plot_metadata_columns(df)
    labels = set(normalized.get_column("method_display").to_list())

    assert "Logistic SER (@2)" in labels
    assert "Logistic SuSiE [L=5] (@2)" in labels
```

- [ ] **Step 2: Run the targeted test to verify it fails or exposes remaining notebook gaps**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run --with pytest python -m pytest \
  tests/test_twogroup_experiments.py -k 'hallmark_ser_enrich_loc_metadata_yields_ser_and_susie_labels' -q
```

Expected: FAIL until the plotting path is fully rewired.

- [ ] **Step 3: Update all notebook preparation call sites**

Ensure every plot-prep path uses normalized frames and the new controls:
- PIP calibration views
- power/FDP views
- max-PIP views
- credible-set truth/component views
- calibration/histogram views

Replace remaining direct raw-method checks like:
- `pl.col("method") == "logistic_threshold"`
- `"logistic_oracle" in _selected_methods`
- `method_label_map().get(method_name, method_name)`

with metadata-driven equivalents using:
- `method_family`
- `L`
- `method_label_base`
- `method_display`

- [ ] **Step 4: Run the focused integration test and the notebook-adjacent test file**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run --with pytest python -m pytest tests/test_twogroup_experiments.py -q
```

Expected: PASS

- [ ] **Step 5: Run syntax validation for the modified files**

Run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run python -m py_compile utils.py tests/test_twogroup_experiments.py
```

Expected: no output

- [ ] **Step 6: Commit**

```bash
git add twogroup_experiments/notebooks/viz2.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: make viz2 spec-json driven for ser and susie fits"
```

## Self-Review

- Spec coverage: the plan covers JSON parsing, metadata normalization, family/L controls, removal of old hard-coded method logic, and integration against the `hallmark__ser_enrich__loc` use case.
- Placeholder scan: no `TODO`/`TBD` placeholders remain; each task has concrete files, commands, and code targets.
- Type consistency: the plan consistently uses `method_spec` / `simulation_spec` JSON columns, `method_family`, `L`, `is_thresholded`, `is_oracle`, `method_label_base`, and `method_display`.
