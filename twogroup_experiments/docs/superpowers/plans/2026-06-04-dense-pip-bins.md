# Dense PIP Bins — Derived Power/FDP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sparse precomputed power/FDP arrays and 20-bin calibration grid with 200 fine bins (width 0.005), deriving power/FDP at plot time via reverse cumulative sums for smoother, correctly-aggregated curves.

**Architecture:** `build_pip_plot_data` stores only two 200-element bin arrays per row. `expand_power_fdp_from_compact` sums bins across replicates per `(collection_name, method, threshold)`, then computes power/FDP via reverse cumulative sums (200 threshold points). `expand_pip_calibration_from_compact` aggregates the 200 fine bins into 20 coarse bins at plot time. `make_power_fdp_summary` is removed (now a no-op). Aggregate plots sum bins across collections before computing, replacing the biased `mean(power)` pattern.

**Tech Stack:** Python, polars, numpy, pytest

---

## File Map

| File | Change |
|------|--------|
| `plot_ready.py` | Remove `_PIP_THRESHOLD_GRID`, `power_at_threshold`, `fdp_at_threshold`; bins 20→200 |
| `viz_utils.py` | Rewrite `expand_pip_calibration_from_compact`, `expand_power_fdp_from_compact`; remove `make_power_fdp_summary`; fix marker index in `_plot_power_fdp_on_ax` |
| `generate_plots.py` | Drop `make_power_fdp_summary` calls; update `_make_agg_power_fdp`, `_make_agg_pip_calibration` |
| `notebooks/dashboard.py` | Drop `make_power_fdp_summary` call |
| `tests/test_plot_ready.py` | Update schema assertions for 200 bins, no power/fdp arrays |

---

### Task 1: Update `build_pip_plot_data` in `plot_ready.py`

**Files:**
- Modify: `plot_ready.py:156-216`
- Test: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing test**

```python
def test_build_pip_plot_data_schema_200_bins():
    fits_df = _make_pip_fits_df()
    sample_metadata = _make_sample_metadata()
    simulations_by_batch = _make_simulations_by_batch()
    result = plot_ready.build_pip_plot_data(fits_df, sample_metadata, simulations_by_batch)
    assert result.height == 2
    assert set(result.columns) == {
        "sample_id", "method", "threshold",
        "causal_indices", "causal_pips",
        "pip_bin_counts", "pip_bin_causal_counts",
    }
    assert result["pip_bin_counts"].dtype == pl.List(pl.Int64)
    assert result["pip_bin_counts"][0].len() == 200
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/test_plot_ready.py::test_build_pip_plot_data_schema_200_bins -v
```

Expected: `AssertionError` (columns include `power_at_threshold`/`fdp_at_threshold`, length is 20).

- [ ] **Step 3: Implement**

In `plot_ready.py`, replace lines 156–216:

```python
_N_PIP_BINS = 200
_PIP_BIN_WIDTH = 1.0 / _N_PIP_BINS  # 0.005


def build_pip_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold). Bin arrays used to derive plots."""
    empty_schema = {
        "sample_id": pl.String, "method": pl.String, "threshold": pl.Float64,
        "causal_indices": pl.List(pl.Int64), "causal_pips": pl.List(pl.Float64),
        "pip_bin_counts": pl.List(pl.Int64), "pip_bin_causal_counts": pl.List(pl.Int64),
    }
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        alphas = np.stack([np.asarray(e["alpha"], dtype=float) for e in row["single_effects"]])
        marginal_pip = 1.0 - np.prod(1.0 - alphas, axis=0)

        sim_df = simulations_by_batch[row["batch_hash"]]
        sim_row = sim_df.filter(pl.col("replicate") == row["replicate"]).row(0, named=True)
        causal_indices = sorted(set(int(i) for i in sim_row["simulation"]["causal_indices"]))

        causal_pips = [float(marginal_pip[ci]) for ci in causal_indices]

        bin_idx = np.clip((marginal_pip * _N_PIP_BINS).astype(int), 0, _N_PIP_BINS - 1)
        is_causal = np.zeros(len(marginal_pip), dtype=bool)
        is_causal[causal_indices] = True
        pip_bin_counts = [int((bin_idx == b).sum()) for b in range(_N_PIP_BINS)]
        pip_bin_causal_counts = [int(((bin_idx == b) & is_causal).sum()) for b in range(_N_PIP_BINS)]

        rows.append({
            "sample_id": row["sample_id"],
            "method": row["method"],
            "threshold": row["threshold"],
            "causal_indices": causal_indices,
            "causal_pips": causal_pips,
            "pip_bin_counts": pip_bin_counts,
            "pip_bin_causal_counts": pip_bin_causal_counts,
        })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)
```

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/test_plot_ready.py::test_build_pip_plot_data_schema_200_bins -v
```

- [ ] **Step 5: Update existing schema test**

In `tests/test_plot_ready.py`, update `test_build_pip_plot_data_schema` (around line 175):

```python
def test_build_pip_plot_data_schema():
    fits_df = _make_pip_fits_df()
    sample_metadata = _make_sample_metadata()
    simulations_by_batch = _make_simulations_by_batch()
    result = plot_ready.build_pip_plot_data(fits_df, sample_metadata, simulations_by_batch)
    assert result.height == 2
    assert set(result.columns) == {
        "sample_id", "method", "threshold",
        "causal_indices", "causal_pips",
        "pip_bin_counts", "pip_bin_causal_counts",
    }
    assert result["pip_bin_counts"].dtype == pl.List(pl.Int64)
    assert result["pip_bin_counts"][0].len() == 200
```

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/test_plot_ready.py -v
```

Expected: all pass (causal_pips tests unaffected).

- [ ] **Step 7: Commit**

```bash
git add plot_ready.py tests/test_plot_ready.py
git commit -m "refactor: dense pip bins (200), remove precomputed power/fdp arrays"
```

---

### Task 2: Update `expand_pip_calibration_from_compact` in `viz_utils.py`

**Files:**
- Modify: `viz_utils.py:103-167`

Aggregate 200 fine bins (width 0.005) into 20 coarse bins (width 0.05) at plot time. Coarse bin `j` = sum of fine bins `j*10 .. (j+1)*10 - 1`.

- [ ] **Step 1: Write failing test**

```python
def test_expand_pip_calibration_200_bins():
    import polars as pl
    import numpy as np
    from viz_utils import expand_pip_calibration_from_compact

    # 200 fine bins: put 1 feature in each fine bin
    pip_bin_counts = [1] * 200
    # causal: 1 in each bin (so empirical_rate = 1.0 everywhere)
    pip_bin_causal_counts = [1] * 200

    pip_data = pl.DataFrame({
        "collection_name": ["col_a"],
        "method": ["twogroup_L1"],
        "threshold": [None],
        "pip_bin_counts": [pip_bin_counts],
        "pip_bin_causal_counts": [pip_bin_causal_counts],
    }, schema={
        "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "pip_bin_counts": pl.List(pl.Int64), "pip_bin_causal_counts": pl.List(pl.Int64),
    })
    method_meta = pl.DataFrame({
        "method": ["twogroup_L1"], "threshold": [None],
        "method_display": ["TwoGroup"], "method_display_base": ["TwoGroup"],
        "method_label_base": ["TwoGroup"], "is_thresholded": [False], "is_oracle": [False],
    }, schema={
        "method": pl.String, "threshold": pl.Float64,
        "method_display": pl.String, "method_display_base": pl.String,
        "method_label_base": pl.String, "is_thresholded": pl.Boolean, "is_oracle": pl.Boolean,
    })
    result = expand_pip_calibration_from_compact(pip_data, method_meta, selected_thresholds=None)
    assert result["pip_bin_index"].to_list() == list(range(20))
    # each coarse bin aggregates 10 fine bins → n_total=10, n_causal=10
    assert result["n_total"].to_list() == [10] * 20
    assert result["n_causal"].to_list() == [10] * 20
    assert all(abs(r - 1.0) < 1e-9 for r in result["empirical_rate"].to_list())
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/ -k "test_expand_pip_calibration_200_bins" -v
```

Expected: `AssertionError` (currently reads 20 fine bins, gets 20 coarse bins by accident but wrong widths).

- [ ] **Step 3: Implement**

Replace `expand_pip_calibration_from_compact` in `viz_utils.py` (lines 103–167):

```python
_N_FINE_BINS = 200
_N_COARSE_BINS = 20
_FINE_PER_COARSE = _N_FINE_BINS // _N_COARSE_BINS  # 10

def expand_pip_calibration_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_thresholds: list[float] | None,
) -> pl.DataFrame:
    """Expand pip_plot_data to 20 coarse-bin rows for render_pip_calibration.

    Aggregates 200 fine bins (width 0.005) into 20 coarse bins (width 0.05).
    """
    if pip_plot_data.is_empty():
        return pl.DataFrame(schema={
            "collection_name": pl.String, "simulation_name": pl.String,
            "method": pl.String, "method_display": pl.String,
            "method_family": pl.String, "series_label": pl.String,
            "pip_bin_index": pl.Int64, "pip_left": pl.Float64, "pip_right": pl.Float64,
            "pip_mid": pl.Float64, "n_total": pl.Int64, "n_causal": pl.Int64,
            "empirical_rate": pl.Float64,
        })
    meta = method_metadata.select(
        "method", "threshold", "method_display", "method_display_base",
        "method_label_base", "is_thresholded", "is_oracle",
    ).with_columns(
        pl.col("method_display").alias("series_label"),
        pl.col("method_display_base").alias("method_family"),
    )
    rows = []
    for row in pip_plot_data.iter_rows(named=True):
        counts = row["pip_bin_counts"]       # length 200
        causal_counts = row["pip_bin_causal_counts"]  # length 200
        for j in range(_N_COARSE_BINS):
            start = j * _FINE_PER_COARSE
            stop = start + _FINE_PER_COARSE
            rows.append({
                "collection_name": row.get("collection_name", ""),
                "method": row["method"],
                "threshold": row["threshold"],
                "pip_bin_index": j,
                "pip_left": j * 0.05,
                "pip_right": (j + 1) * 0.05,
                "pip_mid": (j + 0.5) * 0.05,
                "n_total": sum(counts[start:stop]),
                "n_causal": sum(causal_counts[start:stop]),
            })
    expanded = pl.from_dicts(rows, schema={
        "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "pip_bin_index": pl.Int64, "pip_left": pl.Float64, "pip_right": pl.Float64,
        "pip_mid": pl.Float64, "n_total": pl.Int64, "n_causal": pl.Int64,
    })
    return (
        expanded
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .filter(
            ~pl.col("is_thresholded")
            | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
        )
        .group_by(
            "collection_name", "method", "method_display", "method_family",
            "series_label", "pip_bin_index", "pip_left", "pip_right", "pip_mid",
        )
        .agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
        .with_columns(
            pl.when(pl.col("n_total") > 0)
            .then(pl.col("n_causal") / pl.col("n_total"))
            .otherwise(None)
            .alias("empirical_rate"),
            pl.col("collection_name").alias("simulation_name"),
        )
        .sort("collection_name", "method_display", "pip_mid")
    )
```

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/ -k "test_expand_pip_calibration_200_bins" -v
```

- [ ] **Step 5: Commit**

```bash
git add viz_utils.py tests/
git commit -m "refactor: expand_pip_calibration aggregates 200→20 bins at plot time"
```

---

### Task 3: Rewrite `expand_power_fdp_from_compact` in `viz_utils.py`

**Files:**
- Modify: `viz_utils.py:170-238`

Replace the zip over precomputed `power_at_threshold`/`fdp_at_threshold` with reverse cumulative sums over the 200-bin arrays. Group by `(collection_name, method, threshold)` first, sum bins, then derive 200 threshold points. Add `aggregate_across_collections=False` parameter for aggregate plots.

- [ ] **Step 1: Write failing test**

```python
def test_expand_power_fdp_200_thresholds():
    import polars as pl
    import numpy as np
    from viz_utils import expand_power_fdp_from_compact

    # 200 bins: feature 0 has pip in [0.995, 1.0) → bin 199 (causal)
    #           feature 1 has pip in [0.000, 0.005) → bin 0 (not causal)
    counts = [0] * 200
    causal = [0] * 200
    counts[199] = 1   # high-pip feature
    counts[0] = 1     # low-pip feature
    causal[199] = 1   # causal = high-pip feature

    pip_data = pl.DataFrame({
        "collection_name": ["col_a"],
        "method": ["twogroup_L1"],
        "threshold": [None],
        "pip_bin_counts": [counts],
        "pip_bin_causal_counts": [causal],
    }, schema={
        "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "pip_bin_counts": pl.List(pl.Int64), "pip_bin_causal_counts": pl.List(pl.Int64),
    })
    method_meta = pl.DataFrame({
        "method": ["twogroup_L1"], "threshold": [None],
        "method_display": ["TwoGroup"], "method_label_base": ["TwoGroup"],
        "is_thresholded": [False],
    }, schema={
        "method": pl.String, "threshold": pl.Float64,
        "method_display": pl.String, "method_label_base": pl.String,
        "is_thresholded": pl.Boolean,
    })
    result = expand_power_fdp_from_compact(
        pip_data, method_meta,
        selected_methods={"twogroup_L1"},
        selected_thresholds=None,
    )
    # 200 threshold points
    assert result.height == 200
    # threshold 0.0 (k=0): all selected → power=1.0, fdp=0.5 (1 of 2 features is non-causal)
    row0 = result.filter(pl.col("pip_threshold") < 0.001).row(0, named=True)
    assert abs(row0["power"] - 1.0) < 1e-9
    assert abs(row0["fdp"] - 0.5) < 1e-9
    # threshold 0.995 (k=199): only causal selected → power=1.0, fdp=0.0
    row199 = result.filter(pl.col("pip_threshold") > 0.994).row(0, named=True)
    assert abs(row199["power"] - 1.0) < 1e-9
    assert abs(row199["fdp"] - 0.0) < 1e-9
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/ -k "test_expand_power_fdp_200_thresholds" -v
```

- [ ] **Step 3: Implement**

Replace `expand_power_fdp_from_compact` in `viz_utils.py` (lines 170–238) and delete `make_power_fdp_summary` (lines 366–393):

```python
_N_PIP_BINS = 200
_PIP_BIN_WIDTH = 1.0 / _N_PIP_BINS  # 0.005
_PIP_THRESHOLD_GRID = np.arange(_N_PIP_BINS) * _PIP_BIN_WIDTH  # [0.000, 0.005, ..., 0.995]


def _bins_to_power_fdp(
    counts: np.ndarray,
    causal_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (power, fdp) for each threshold k*0.005 via reverse cumulative sums."""
    rev_cum_counts = np.cumsum(counts[::-1])[::-1]
    rev_cum_causal = np.cumsum(causal_counts[::-1])[::-1]
    total_causal = int(causal_counts.sum())
    power = rev_cum_causal / max(total_causal, 1)
    fdp = (rev_cum_counts - rev_cum_causal) / np.maximum(rev_cum_counts, 1)
    return power.astype(float), fdp.astype(float)


def expand_power_fdp_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None,
    aggregate_across_collections: bool = False,
) -> pl.DataFrame:
    """Derive per-threshold power/FDP rows from 200-bin arrays.

    Bins are summed per (collection_name, method, threshold) across replicates,
    then power/FDP are computed via reverse cumulative sums (200 threshold points).
    When aggregate_across_collections=True, bins are summed over all collections
    before computing — correct for aggregate plots.
    """
    empty = pl.DataFrame(schema={
        "simulation_name": pl.String, "method": pl.String, "method_display": pl.String,
        "trace_label": pl.String, "legend_label": pl.String,
        "is_selected_threshold": pl.Boolean,
        "pip_threshold": pl.Float64, "power": pl.Float64, "fdp": pl.Float64,
    })
    if pip_plot_data.is_empty():
        return empty

    meta = method_metadata.select(
        "method", "threshold", "method_display", "method_label_base", "is_thresholded",
    )

    filtered = (
        pip_plot_data
        .filter(pl.col("method").is_in(list(selected_methods)))
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded")
                | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
            ).alias("is_selected_threshold")
        )
        .filter(pl.col("is_selected_threshold"))
    )
    if filtered.is_empty():
        return empty

    group_keys = ["method", "threshold", "method_display", "method_label_base",
                  "is_thresholded", "is_selected_threshold"]
    if not aggregate_across_collections:
        group_keys = ["collection_name"] + group_keys

    # Sum bin arrays per group
    summed = (
        filtered
        .group_by(group_keys)
        .agg(
            pl.col("pip_bin_counts").list.sum(),
            pl.col("pip_bin_causal_counts").list.sum(),
        )
    )

    rows = []
    for row in summed.iter_rows(named=True):
        counts = np.asarray(row["pip_bin_counts"], dtype=float)
        causal = np.asarray(row["pip_bin_causal_counts"], dtype=float)
        power_arr, fdp_arr = _bins_to_power_fdp(counts, causal)
        col_name = "" if aggregate_across_collections else row.get("collection_name", "")
        trace_label = (
            f"{row['method_label_base']} (@{row['threshold']})"
            if row["is_thresholded"] else row["method_display"]
        )
        for k in range(_N_PIP_BINS):
            rows.append({
                "simulation_name": col_name,
                "method": row["method"],
                "threshold": row["threshold"],
                "method_display": row["method_display"],
                "is_thresholded": row["is_thresholded"],
                "is_selected_threshold": row["is_selected_threshold"],
                "trace_label": trace_label,
                "legend_label": row["method_display"] if row["is_selected_threshold"] else None,
                "pip_threshold": float(_PIP_THRESHOLD_GRID[k]),
                "power": float(power_arr[k]),
                "fdp": float(fdp_arr[k]),
            })

    return pl.from_dicts(rows, schema={
        "simulation_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "method_display": pl.String, "is_thresholded": pl.Boolean,
        "is_selected_threshold": pl.Boolean,
        "trace_label": pl.String, "legend_label": pl.String,
        "pip_threshold": pl.Float64, "power": pl.Float64, "fdp": pl.Float64,
    }).sort("simulation_name", "method_display", "pip_threshold")
```

Also delete `make_power_fdp_summary` (lines 366–393 — the entire function). It's a no-op now.

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/ -k "test_expand_power_fdp_200_thresholds" -v
```

- [ ] **Step 5: Commit**

```bash
git add viz_utils.py tests/
git commit -m "refactor: expand_power_fdp derives 200-pt curve from bins, remove make_power_fdp_summary"
```

---

### Task 4: Fix marker index in `_plot_power_fdp_on_ax`

**Files:**
- Modify: `viz_utils.py:430` (the `idx = int(round(thresh * 1000)) - 1` line)

The new grid has `pip_threshold[k] = k * 0.005`, so `thresh=0.5` → index 100, `thresh=0.9` → index 180, `thresh=0.99` → index 198.

- [ ] **Step 1: Replace the index formula**

Change line ~430:
```python
# old: idx = int(round(thresh * 1000)) - 1  # threshold_grid[i] = (i+1)/1000
idx = int(thresh * _N_PIP_BINS)  # pip_threshold[k] = k * 0.005, so k = thresh / 0.005
```

- [ ] **Step 2: Verify markers still within bounds**

`thresh` values are `[0.5, 0.9, 0.99]` → indices `[100, 180, 198]`. All in `[0, 199]`. ✓

- [ ] **Step 3: Commit**

```bash
git add viz_utils.py
git commit -m "fix: power/fdp marker index for 200-pt threshold grid"
```

---

### Task 5: Update `generate_plots.py`

**Files:**
- Modify: `generate_plots.py:519-576`

Remove calls to `make_power_fdp_summary`. Use `aggregate_across_collections=True` in `_make_agg_power_fdp`.

- [ ] **Step 1: Update `_make_agg_pip_calibration` (line 519)**

The current code calls `expand_pip_calibration_from_compact` then re-aggregates. The new `expand_pip_calibration_from_compact` already returns per-collection calibration — aggregate version just groups across collections. No change needed in `_make_agg_pip_calibration` since it already does:
```python
agg = (
    summary
    .group_by("method", "method_display", "series_label", "method_family",
              "pip_bin_index", "pip_left", "pip_right", "pip_mid")
    .agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
    .with_columns(
        pl.when(pl.col("n_total") > 0)
        .then(pl.col("n_causal") / pl.col("n_total"))
        .otherwise(None)
        .alias("empirical_rate")
    )
)
```
This is correct — sum `n_total`/`n_causal` across collections, then recompute `empirical_rate`. No change needed here.

- [ ] **Step 2: Update `_make_power_fdp` (line 147) — remove `make_power_fdp_summary`**

```python
def _make_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
    )
    if power_fdp.is_empty():
        return viz_utils.make_placeholder_chart("No power/FDP data")
    return viz_utils.render_power_fdp_chart(
        power_fdp,
        facet=True,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
        collection_names=combined_data["collection_names"],
    )
```

- [ ] **Step 3: Update `_make_agg_power_fdp` (line 547) — use `aggregate_across_collections=True`**

```python
def _make_agg_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        aggregate_across_collections=True,
    )
    if power_fdp.is_empty():
        return viz_utils.make_placeholder_chart("No power/FDP data")
    fig = viz_utils.render_power_fdp_chart(
        power_fdp,
        facet=False,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
    )
    _set_agg_facecolor(fig)
    return fig
```

- [ ] **Step 4: Commit**

```bash
git add generate_plots.py
git commit -m "refactor: remove make_power_fdp_summary calls, use aggregate_across_collections"
```

---

### Task 6: Update `notebooks/dashboard.py`

**Files:**
- Modify: `notebooks/dashboard.py:254-256`

- [ ] **Step 1: Remove `make_power_fdp_summary` call**

Change lines 247–265 in `power_fdp_cell`:

```python
    _power_fdp = viz_utils.expand_power_fdp_from_compact(
        _pip_plot, _method_meta,
        selected_methods=foreground_methods,
        selected_thresholds=selected_thresholds,
    )
    if _power_fdp.is_empty():
        power_fdp_chart = viz_utils.make_placeholder_chart("No power/FDP data")
    else:
        power_fdp_chart = viz_utils.render_power_fdp_chart(
            _power_fdp,
            facet=True,
            max_fdp=max_fdp_slider.value,
            fixed_y_scale=True,
            legend_outside=True,
            square_axes=True,
            collection_names=combined_data["collection_names"],
        )
```

(Remove the `_summary = viz_utils.make_power_fdp_summary(_power_fdp)` line and pass `_power_fdp` directly to `render_power_fdp_chart`.)

- [ ] **Step 2: Commit**

```bash
git add notebooks/dashboard.py
git commit -m "refactor: remove make_power_fdp_summary from dashboard"
```

---

### Task 7: Invalidate and regenerate `pip_plot_data`

All existing `pip_plot_data.parquet` files have the old schema (20 bins, contains `power_at_threshold`/`fdp_at_threshold`) and must be deleted before running Snakemake.

- [ ] **Step 1: Delete stale files**

```bash
find results/ -name "pip_plot_data.parquet" -delete
```

- [ ] **Step 2: Verify deletion**

```bash
find results/ -name "pip_plot_data.parquet" | wc -l
```

Expected: `0`

- [ ] **Step 3: Dry-run to check regeneration scope**

```bash
uv run snakemake all_collections -n 2>&1 | grep "collection_pip_plot_data\|total" | tail -5
```

- [ ] **Step 4: Commit note**

```bash
git commit --allow-empty -m "chore: invalidated pip_plot_data.parquet (schema change: 200 bins)"
```

---

## Self-Review

**Spec coverage:**
- ✓ 200 bins, width 0.005 — Task 1
- ✓ Remove precomputed power/fdp — Task 1
- ✓ Calibration aggregates 200→20 at plot time — Task 2
- ✓ Power/FDP from reverse cumulative sums — Task 3
- ✓ Sum bins first, then compute (correct aggregation) — Task 3 (`aggregate_across_collections`)
- ✓ Remove legacy `make_power_fdp_summary` — Task 3 + 5 + 6
- ✓ Fix marker index — Task 4
- ✓ Invalidate parquet — Task 7

**Placeholder scan:** None found.

**Type consistency:**
- `_N_PIP_BINS = 200` defined in both `plot_ready.py` (Task 1) and `viz_utils.py` (Task 3) independently — they must match. Consider importing from a shared constant, or just keep both in sync (both are 200).
- `_N_COARSE_BINS = 20`, `_FINE_PER_COARSE = 10` — only in `viz_utils.py` Task 2.
- `aggregate_across_collections` parameter added in Task 3, used in Task 5.
- `make_power_fdp_summary` deleted in Task 3, all call sites removed in Tasks 5–6.
