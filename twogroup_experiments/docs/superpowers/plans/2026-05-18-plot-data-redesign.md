# Plot Data Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace seven stale plot_ready tables with two compact tables (`cs_plot_data`, `pip_plot_data`) that handle arbitrary L SuSiE effects, pre-aggregate into arrays, and require only simple operations at plot time.

**Architecture:** `build_pip_plot_data` and `build_cs_plot_data` replace all twelve old summarize/aggregate/finalize functions in `plot_ready.py`. `viz_utils.py` gains new helpers that expand compact arrays for rendering. Dashboard cells are updated to use the two new table keys.

**Tech Stack:** Python, Polars, NumPy, Marimo dashboard.

---

## File Map

| File | Change |
|---|---|
| `core.py` | Remove `marginal_pip`, `causal_pip`, `max_pip` from `_make_fit_summary_struct` |
| `plot_ready.py` | Replace 12 old functions with `build_pip_plot_data` + `build_cs_plot_data`; update `load_collection_fits_with_specs` column select; update `load_plot_ready_collection` names |
| `viz_utils.py` | Rewrite `make_causal_rank_summary`; add `_expand_cs_to_beta_rows`; rewrite `make_cs_beta_trace_summary`; add `expand_pip_calibration_from_compact`, `expand_power_fdp_from_compact`, `expand_causal_pip_from_compact`; remove three `mass_above_causal` functions |
| `notebooks/dashboard.py` | Update `combined_data` dict (7 keys → 2); update 7 cell functions; remove 2 mass_above_causal cells |
| `utils.py` | Remove `build_plot_data_frames`, `_plot_frame`, legacy schema constants, `PLOT_DATA_FILENAMES`, `symlink_plot_data_outputs` |
| `tests/test_twogroup_experiments.py` | Remove 2 marginal_pip tests + 4 build_plot_data_frames tests |
| `tests/test_plot_ready.py` | Remove 7 stale tests; add `test_build_pip_plot_data_schema`, `test_build_pip_plot_data_causal_pips_correct`, `test_build_cs_plot_data_schema`, `test_build_cs_plot_data_cs_sizes_length` |

---

### Task 1: Strip fit_summary in core.py

**Files:**
- Modify: `core.py:221-237`
- Modify: `tests/test_twogroup_experiments.py`

- [ ] **Step 1: Run the two marginal_pip tests to confirm they currently pass**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_twogroup_experiments.py -k "marginal_pip" -v
```
Expected: 2 passed.

- [ ] **Step 2: Remove marginal_pip, causal_pip, max_pip from _make_fit_summary_struct in core.py**

Replace the full function body (lines 221–237):

```python
def _make_fit_summary_struct(
    state: Any, simulation: TwoGroupSimulation, n_selected: int | None
) -> dict[str, Any]:
    return {
        "n_selected": None if n_selected is None else int(n_selected),
        "n_iter": int(state.n_iter),
        "converged": bool(state.converged),
    }
```

- [ ] **Step 3: Remove the two marginal_pip tests from test_twogroup_experiments.py**

Delete `test_fit_batch_method_marginal_pip_correct_for_L1` and `test_fit_batch_method_marginal_pip_correct_for_L5` (these tested fit_summary.marginal_pip which no longer exists; the computation will be tested in Task 2).

- [ ] **Step 4: Run schema tests to confirm they still pass**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_twogroup_experiments.py -k "schema or single_effects or L5 or L1" -v
```
Expected: 3 passed (test_fit_batch_method_schema_has_single_effects_list, test_fit_batch_method_L5_has_5_effects, test_fit_batch_method_marginal_pip was removed).

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/core.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "feat: strip fit_summary to {n_selected, n_iter, converged}"
```

---

### Task 2: Add build_pip_plot_data to plot_ready.py

**Files:**
- Modify: `plot_ready.py`
- Modify: `tests/test_plot_ready.py`

Context: `_PIP_THRESHOLD_GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]` (defined in plot_ready.py line 226). `CS_BETA_GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)` (50 values, defined in utils.py).

- [ ] **Step 1: Write two failing tests in tests/test_plot_ready.py**

Add after the last existing test:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_plot_ready.py -k "build_pip_plot_data" -v
```
Expected: FAIL with `AttributeError: module 'plot_ready' has no attribute 'build_pip_plot_data'`.

- [ ] **Step 3: Implement build_pip_plot_data in plot_ready.py**

Add after `summarize_causal_pip_per_sample` (before `aggregate_causal_pip`):

```python
def build_pip_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold). Arrays pre-aggregated for plot time."""
    empty_schema = {
        "sample_id": pl.String, "method": pl.String, "threshold": pl.Float64,
        "causal_indices": pl.List(pl.Int64), "causal_pips": pl.List(pl.Float64),
        "pip_bin_counts": pl.List(pl.Int64), "pip_bin_causal_counts": pl.List(pl.Int64),
        "power_at_threshold": pl.List(pl.Float64), "fdp_at_threshold": pl.List(pl.Float64),
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
        causal_indices = [int(i) for i in sim_row["simulation"]["causal_indices"]]

        causal_pips = [float(marginal_pip[ci]) for ci in causal_indices]

        bin_idx = np.clip((marginal_pip * 20).astype(int), 0, 19)
        is_causal = np.zeros(len(marginal_pip), dtype=bool)
        is_causal[causal_indices] = True
        pip_bin_counts = [int((bin_idx == b).sum()) for b in range(20)]
        pip_bin_causal_counts = [int(((bin_idx == b) & is_causal).sum()) for b in range(20)]

        n_causal = max(len(causal_indices), 1)
        power_at_threshold = []
        fdp_at_threshold = []
        for t in _PIP_THRESHOLD_GRID:
            selected = marginal_pip >= t
            sel_causal = int(selected[causal_indices].sum())
            sel_total = int(selected.sum())
            power_at_threshold.append(float(sel_causal / n_causal))
            fdp_at_threshold.append(float((sel_total - sel_causal) / max(sel_total, 1)))

        rows.append({
            "sample_id": row["sample_id"],
            "method": row["method"],
            "threshold": row["threshold"],
            "causal_indices": causal_indices,
            "causal_pips": causal_pips,
            "pip_bin_counts": pip_bin_counts,
            "pip_bin_causal_counts": pip_bin_causal_counts,
            "power_at_threshold": power_at_threshold,
            "fdp_at_threshold": fdp_at_threshold,
        })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_plot_ready.py -k "build_pip_plot_data" -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/plot_ready.py twogroup_experiments/tests/test_plot_ready.py
git commit -m "feat: add build_pip_plot_data with compact array format"
```

---

### Task 3: Add build_cs_plot_data to plot_ready.py

**Files:**
- Modify: `plot_ready.py`
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Write two failing tests**

Add to `tests/test_plot_ready.py` (reuse `_make_pip_fits_df`, `_make_sample_metadata` from Task 2):

```python
def test_build_cs_plot_data_schema():
    from utils import CS_BETA_GRID

    fits_df = _make_pip_fits_df()
    sample_metadata = _make_sample_metadata()

    result = plot_ready.build_cs_plot_data(fits_df, sample_metadata)

    # 2 replicates × L=1 effect = 2 rows
    assert result.height == 2
    assert set(result.columns) == {
        "sample_id", "method", "threshold", "l",
        "ser_log_bf", "causal_indices", "rank_of_causal",
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_plot_ready.py -k "build_cs_plot_data" -v
```
Expected: FAIL with `AttributeError: module 'plot_ready' has no attribute 'build_cs_plot_data'`.

- [ ] **Step 3: Implement build_cs_plot_data in plot_ready.py**

Add after `build_pip_plot_data`:

```python
def build_cs_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold, l). Arrays for CS sweep at each beta."""
    from utils import CS_BETA_GRID

    empty_schema = {
        "sample_id": pl.String, "method": pl.String, "threshold": pl.Float64,
        "l": pl.Int64, "ser_log_bf": pl.Float64,
        "causal_indices": pl.List(pl.Int64), "rank_of_causal": pl.List(pl.Int64),
        "mass_above_causal": pl.List(pl.Float64), "cs_sizes": pl.List(pl.Int64),
    }
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        for l, effect in enumerate(row["single_effects"]):
            alpha = np.asarray(effect["alpha"], dtype=float)
            cs_struct = row["credible_sets"][l]
            causal_indices = [int(ci) for ci in cs_struct["causal_indices"]]

            order = np.argsort(-alpha)
            cumulative = np.cumsum(alpha[order])
            rank_of = {int(feat): rk for rk, feat in enumerate(order.tolist())}

            rank_of_causal = [rank_of[ci] for ci in causal_indices]
            mass_above_causal = [
                float(cumulative[rk - 1]) if rk > 0 else 0.0
                for rk in rank_of_causal
            ]
            cs_sizes = [
                int(np.searchsorted(cumulative, float(beta), side="left") + 1)
                for beta in CS_BETA_GRID
            ]
            rows.append({
                "sample_id": row["sample_id"],
                "method": row["method"],
                "threshold": row["threshold"],
                "l": l,
                "ser_log_bf": float(effect["ser_log_bf"]),
                "causal_indices": causal_indices,
                "rank_of_causal": rank_of_causal,
                "mass_above_causal": mass_above_causal,
                "cs_sizes": cs_sizes,
            })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_plot_ready.py -k "build_cs_plot_data or build_pip_plot_data" -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add twogroup_experiments/plot_ready.py twogroup_experiments/tests/test_plot_ready.py
git commit -m "feat: add build_cs_plot_data with compact array format"
```

---

### Task 4: Update load helpers and remove old plot_ready functions

**Files:**
- Modify: `plot_ready.py`
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Update load_collection_fits_with_specs column select (plot_ready.py line 32-35)**

```python
            frames.append(fits_df.select(
                "method", "threshold", "method_spec", "batch_hash", "replicate",
                "single_effects", "fit_summary", "credible_sets",
            ))
```

- [ ] **Step 2: Update load_plot_ready_collection names list (plot_ready.py ~line 497-508)**

```python
    names = [
        "method_metadata",
        "simulation_metadata",
        "sample_metadata",
        "pip_plot_data",
        "cs_plot_data",
    ]
```

- [ ] **Step 3: Remove old summarize/aggregate/finalize functions from plot_ready.py**

Delete the following functions entirely:
- `summarize_pip_calibration_per_sample` (lines 142–192)
- `aggregate_pip_calibration` (lines 195–223)
- `summarize_power_fdp_per_sample` (lines 229–280)
- `aggregate_power_fdp` (lines 283–291)
- `summarize_causal_pip_per_sample` (lines 294–309)
- `aggregate_causal_pip` (lines 312–317)
- `summarize_cs_raw_per_sample` (lines 320–337)
- `summarize_cs_beta_trace` (lines 340–394)
- `summarize_cs_size_histogram_observations` (lines 397–411)
- `finalize_cs_size_histogram` (lines 414–417)
- `summarize_ser_log_bf_histogram_observations` (lines 420–434)
- `finalize_ser_log_bf_histogram` (lines 437–440)

Also remove the module-level `_PIP_THRESHOLD_GRID` constant (line 226) — it is now embedded in `build_pip_plot_data`.

- [ ] **Step 4: Remove stale tests from test_plot_ready.py**

Delete these test functions:
- `test_build_pip_calibration_returns_collection_level_bins`
- `test_build_power_fdp_returns_collection_level_curve`
- `test_build_causal_pip_returns_collection_means`
- `test_summarize_cs_raw_per_sample_columns`
- `test_build_cs_size_histogram_returns_raw_observations`
- `test_build_ser_log_bf_histogram_returns_raw_observations`

- [ ] **Step 5: Run all plot_ready tests**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_plot_ready.py -v
```
Expected: tests that do not depend on removed functions pass; new build_pip/cs tests pass.

- [ ] **Step 6: Commit**

```bash
git add twogroup_experiments/plot_ready.py twogroup_experiments/tests/test_plot_ready.py
git commit -m "refactor: replace 12 old plot_ready functions with build_pip_plot_data/build_cs_plot_data"
```

---

### Task 5: Rewrite CS viz helpers in viz_utils.py

**Files:**
- Modify: `viz_utils.py`

Context: `make_causal_rank_summary` (line 745) currently takes `cs_beta_trace` (flat per-beta rows) and computes minimum cs_size where covered=True. With `cs_plot_data`, causal rank = `min(rank_of_causal) + 1` per effect, then `min` across effects per sample. `make_cs_beta_trace_summary` (line 1449) needs expansion to per-beta rows.

- [ ] **Step 1: Rewrite make_causal_rank_summary to accept cs_plot_data**

Replace the existing `make_causal_rank_summary` function (lines 745–785) with:

```python
def make_causal_rank_summary(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
) -> pl.DataFrame:
    """Mean causal rank per (collection_name, method, threshold).

    Causal rank = minimum cs_size required to include any causal = min(rank_of_causal) + 1.
    Takes the minimum across all L effects per sample, then averages across samples.
    """
    empty_schema = {
        "simulation_name": pl.String,
        "method": pl.String,
        "threshold": pl.Float64,
        "method_display": pl.String,
        "method_display_base": pl.String,
        "mean_causal_rank": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select(
        "method", "threshold", "method_display", "method_display_base", "is_thresholded"
    )
    per_effect = (
        cs_plot_data
        .filter(pl.col("method").is_in(list(selected_methods)))
        .filter(pl.col("rank_of_causal").list.len() > 0)
        .with_columns(
            (pl.col("rank_of_causal").list.min() + 1).alias("causal_rank")
        )
    )
    if per_effect.is_empty():
        return pl.DataFrame(schema=empty_schema)
    per_sample = (
        per_effect
        .group_by("collection_name", "sample_id", "method", "threshold")
        .agg(pl.col("causal_rank").min())
    )
    return (
        per_sample
        .group_by("collection_name", "method", "threshold")
        .agg(pl.col("causal_rank").mean().alias("mean_causal_rank"))
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(pl.col("collection_name").alias("simulation_name"))
        .sort("simulation_name", "method_display", "threshold")
    )
```

- [ ] **Step 2: Add _expand_cs_to_beta_rows helper before make_cs_beta_trace_summary**

Insert before `make_cs_beta_trace_summary` (line 1449):

```python
def _expand_cs_to_beta_rows(cs_plot_data: pl.DataFrame) -> pl.DataFrame:
    """Expand compact cs_plot_data to one row per (sample_id, method, threshold, l, beta).

    covered = any causal in CS_l at this beta.
    power = fraction of causals in CS_l at this beta.
    """
    from utils import CS_BETA_GRID

    rows: list[dict] = []
    for row in cs_plot_data.iter_rows(named=True):
        ranks = row["rank_of_causal"]
        n_causal = max(len(ranks), 1)
        for beta, cs_size in zip(CS_BETA_GRID.tolist(), row["cs_sizes"]):
            n_covered = sum(1 for r in ranks if r < cs_size)
            rows.append({
                "collection_name": row["collection_name"],
                "sample_id": row["sample_id"],
                "method": row["method"],
                "threshold": row["threshold"],
                "l": row["l"],
                "beta": float(beta),
                "cs_size": cs_size,
                "covered": n_covered > 0,
                "power": float(n_covered / n_causal),
                "ser_log_bf": row["ser_log_bf"],
            })
    if not rows:
        return pl.DataFrame(schema={
            "collection_name": pl.String, "sample_id": pl.String,
            "method": pl.String, "threshold": pl.Float64, "l": pl.Int64,
            "beta": pl.Float64, "cs_size": pl.Int64, "covered": pl.Boolean,
            "power": pl.Float64, "ser_log_bf": pl.Float64,
        })
    return pl.from_dicts(rows, schema={
        "collection_name": pl.String, "sample_id": pl.String,
        "method": pl.String, "threshold": pl.Float64, "l": pl.Int64,
        "beta": pl.Float64, "cs_size": pl.Int64, "covered": pl.Boolean,
        "power": pl.Float64, "ser_log_bf": pl.Float64,
    })
```

- [ ] **Step 3: Rewrite make_cs_beta_trace_summary to accept cs_plot_data**

Replace the existing `make_cs_beta_trace_summary` (lines 1449–1495):

```python
def make_cs_beta_trace_summary(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_threshold: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    """Summarize cs_plot_data across all betas for each (collection_name, method, threshold, beta)."""
    empty_schema = {
        "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean,
        "is_selected_threshold": pl.Boolean, "beta": pl.Float64,
        "power": pl.Float64, "coverage": pl.Float64, "cs_size": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    expanded = _expand_cs_to_beta_rows(
        cs_plot_data.filter(pl.col("method").is_in(list(selected_methods)))
    )
    if expanded.is_empty():
        return pl.DataFrame(schema=empty_schema)
    filtered = (
        expanded
        .with_columns(
            ((pl.col("cs_size") <= max_cs_size) & (pl.col("ser_log_bf") >= min_ser_log_bf)).alias("valid_cs")
        )
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded") | (pl.col("threshold") == selected_threshold)
            ).alias("is_selected_threshold")
        )
    )
    return (
        filtered
        .group_by("collection_name", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(
            (pl.col("covered") & pl.col("valid_cs")).cast(pl.Float64).mean().alias("power"),
            pl.when(pl.col("valid_cs")).then(pl.col("covered").cast(pl.Float64)).mean().alias("coverage"),
            pl.when(pl.col("valid_cs")).then(pl.col("cs_size").cast(pl.Float64)).mean().alias("cs_size"),
        )
        .sort("collection_name", "method_display", "threshold", "beta")
    )
```

- [ ] **Step 4: Commit**

```bash
git add twogroup_experiments/viz_utils.py
git commit -m "refactor: rewrite CS viz helpers to use cs_plot_data compact format"
```

---

### Task 6: Add PIP expansion helpers to viz_utils.py

**Files:**
- Modify: `viz_utils.py`

Context: The dashboard pip cells currently call `summarize_pip_calibration`, `prepare_power_fdp_plot_data_frame`/`make_power_fdp_summary`, `make_causal_pip_summary` with flat per-sample DataFrames. We add new helpers that expand `pip_plot_data` compact format for the render functions. The render functions themselves are unchanged.

- [ ] **Step 1: Add expand_pip_calibration_from_compact before summarize_pip_calibration (line 311)**

```python
def expand_pip_calibration_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Expand pip_plot_data to per-bin rows for render_pip_calibration.

    Returns DataFrame with columns: collection_name, method, method_display,
    method_family, series_label, pip_bin_index, pip_left, pip_right, pip_mid,
    n_total, n_causal, empirical_rate.
    """
    if pip_plot_data.is_empty():
        return pl.DataFrame(schema={
            "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
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
        counts = row["pip_bin_counts"]
        causal_counts = row["pip_bin_causal_counts"]
        for b in range(20):
            rows.append({
                "collection_name": row.get("collection_name", ""),
                "method": row["method"],
                "threshold": row["threshold"],
                "pip_bin_index": b,
                "pip_left": b * 0.05,
                "pip_right": (b + 1) * 0.05,
                "pip_mid": (b + 0.5) * 0.05,
                "n_total": counts[b],
                "n_causal": causal_counts[b],
            })
    expanded = pl.from_dicts(rows, schema={
        "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "pip_bin_index": pl.Int64, "pip_left": pl.Float64, "pip_right": pl.Float64,
        "pip_mid": pl.Float64, "n_total": pl.Int64, "n_causal": pl.Int64,
    })
    return (
        expanded
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .group_by(
            "collection_name", "method", "method_display", "method_family",
            "series_label", "pip_bin_index", "pip_left", "pip_right", "pip_mid",
        )
        .agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
        .with_columns(
            pl.when(pl.col("n_total") > 0)
            .then(pl.col("n_causal") / pl.col("n_total"))
            .otherwise(None)
            .alias("empirical_rate")
        )
        .sort("collection_name", "method_display", "pip_mid")
    )
```

- [ ] **Step 2: Add expand_power_fdp_from_compact**

Add after `expand_pip_calibration_from_compact`:

```python
def expand_power_fdp_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_threshold: float,
    show_background_threshold_traces: bool,
) -> pl.DataFrame:
    """Expand pip_plot_data to per-threshold rows for render_power_fdp_chart."""
    from utils import PIP_THRESHOLD_GRID as _GRID

    if pip_plot_data.is_empty():
        return pl.DataFrame(schema={
            "simulation_name": pl.String, "method": pl.String, "method_display": pl.String,
            "trace_label": pl.String, "legend_label": pl.String,
            "is_selected_threshold": pl.Boolean,
            "pip_threshold": pl.Float64, "power": pl.Float64, "fdp": pl.Float64,
        })
    meta = method_metadata.select(
        "method", "threshold", "method_display", "method_label_base", "is_thresholded",
    )
    rows = []
    for row in pip_plot_data.iter_rows(named=True):
        for t_idx, (t, power, fdp) in enumerate(zip(
            _GRID, row["power_at_threshold"], row["fdp_at_threshold"]
        )):
            rows.append({
                "collection_name": row.get("collection_name", ""),
                "method": row["method"],
                "threshold": row["threshold"],
                "pip_threshold": float(t),
                "power": power,
                "fdp": fdp,
            })
    expanded = pl.from_dicts(rows, schema={
        "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "pip_threshold": pl.Float64, "power": pl.Float64, "fdp": pl.Float64,
    })
    joined = (
        expanded
        .filter(pl.col("method").is_in(list(selected_methods)))
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded") | (pl.col("threshold") == selected_threshold)
            ).alias("is_selected_threshold")
        )
    )
    if not show_background_threshold_traces:
        joined = joined.filter(pl.col("is_selected_threshold"))
    return (
        joined
        .group_by(
            "collection_name", "method", "method_display", "method_label_base",
            "is_thresholded", "is_selected_threshold", "threshold", "pip_threshold",
        )
        .agg(pl.col("power").mean(), pl.col("fdp").mean())
        .with_columns(
            pl.col("collection_name").alias("simulation_name"),
            pl.when(pl.col("is_thresholded"))
            .then(pl.format("{} (@{})", pl.col("method_label_base"), pl.col("threshold")))
            .otherwise(pl.col("method_display"))
            .alias("trace_label"),
            pl.when(pl.col("is_selected_threshold"))
            .then(pl.col("method_display"))
            .otherwise(None)
            .alias("legend_label"),
        )
        .sort("simulation_name", "method_display", "pip_threshold")
    )
```

- [ ] **Step 3: Add expand_causal_pip_from_compact**

Add after `expand_power_fdp_from_compact`:

```python
def expand_causal_pip_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Expand pip_plot_data to per-causal rows for render_causal_pip_chart."""
    if pip_plot_data.is_empty():
        return pl.DataFrame(schema={
            "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
            "causal_pip": pl.Float64,
        })
    meta = method_metadata.select("method", "threshold", "method_display")
    rows = []
    for row in pip_plot_data.iter_rows(named=True):
        for pip in row["causal_pips"]:
            rows.append({
                "collection_name": row.get("collection_name", ""),
                "method": row["method"],
                "threshold": row["threshold"],
                "causal_pip": float(pip),
            })
    expanded = pl.from_dicts(rows, schema={
        "collection_name": pl.String, "method": pl.String,
        "threshold": pl.Float64, "causal_pip": pl.Float64,
    })
    return (
        expanded
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(pl.col("collection_name").alias("simulation_name"))
        .sort("simulation_name", "method_display")
    )
```

- [ ] **Step 4: Commit**

```bash
git add twogroup_experiments/viz_utils.py
git commit -m "feat: add PIP expansion helpers for compact pip_plot_data format"
```

---

### Task 7: Update dashboard to use new table keys

**Files:**
- Modify: `notebooks/dashboard.py`

Context: `combined_data` dict at line 360 maps old table names. Each cell function pulls from `combined_data[key]`. Update the dict to use `pip_plot_data` and `cs_plot_data`, then update each cell function.

- [ ] **Step 1: Update combined_data dict (lines 360–370)**

```python
    combined_data = {
        "method_metadata": combined_method_metadata,
        "collection_names": [_aliases.get(n, n) for n in _selected],
        "pip_plot_data": _tag("pip_plot_data"),
        "cs_plot_data": _tag("cs_plot_data"),
    }
```

- [ ] **Step 2: Update pip_calibration_cell**

The cell currently reads `combined_data["pip_calibration"]`. Change to use `expand_pip_calibration_from_compact`:

```python
@app.cell
def pip_calibration_cell(combined_data, foreground_methods, facet_pip_calibration):
    _pip_plot = combined_data["pip_plot_data"]
    _method_meta = combined_data["method_metadata"]
    _pip_cal_summary = viz_utils.expand_pip_calibration_from_compact(_pip_plot, _method_meta)
    if _pip_cal_summary.is_empty():
        pip_calibration_chart = viz_utils.make_placeholder_chart("No PIP calibration data")
    else:
        pip_calibration_chart = viz_utils.render_pip_calibration(
            _pip_cal_summary, facet_by_simulation=facet_pip_calibration.value
        )
    return (pip_calibration_chart,)
```

- [ ] **Step 3: Update power_fdp_cell**

Read `combined_data["pip_plot_data"]` and call `expand_power_fdp_from_compact`. The cell passes `selected_threshold`, `show_background_threshold_traces`, `selected_methods` parameters — match existing cell signature:

```python
@app.cell
def power_fdp_cell(combined_data, foreground_methods, threshold_dropdown, show_all_thresholds):
    _pip_plot = combined_data["pip_plot_data"]
    _method_meta = combined_data["method_metadata"]
    _selected_threshold = float(threshold_dropdown.value) if threshold_dropdown.value else 2.0
    _power_fdp = viz_utils.expand_power_fdp_from_compact(
        _pip_plot, _method_meta,
        selected_methods=set(foreground_methods),
        selected_threshold=_selected_threshold,
        show_background_threshold_traces=show_all_thresholds.value,
    )
    if _power_fdp.is_empty():
        power_fdp_chart = viz_utils.make_placeholder_chart("No power/FDP data")
    else:
        _summary = viz_utils.make_power_fdp_summary(_power_fdp)
        power_fdp_chart = viz_utils.render_power_fdp_chart(_summary)
    return (power_fdp_chart,)
```

- [ ] **Step 4: Update causal_pip_cell**

```python
@app.cell
def causal_pip_cell(combined_data, foreground_methods):
    _pip_plot = combined_data["pip_plot_data"]
    _method_meta = combined_data["method_metadata"]
    _causal_pip = viz_utils.expand_causal_pip_from_compact(_pip_plot, _method_meta)
    if _causal_pip.is_empty():
        causal_pip_chart = viz_utils.make_placeholder_chart("No causal PIP data")
    else:
        _summary = viz_utils.make_causal_pip_summary(
            _causal_pip.filter(pl.col("method").is_in(foreground_methods))
        )
        causal_pip_chart = viz_utils.render_causal_pip_chart(_summary)
    return (causal_pip_chart,)
```

- [ ] **Step 5: Update cs_summary_cell (causal rank chart)**

Find the cell that calls `make_causal_rank_summary` (around line 663). Change `combined_data.get("cs_beta_trace", ...)` to `combined_data.get("cs_plot_data", pl.DataFrame())`:

```python
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    _rank_summary = viz_utils.make_causal_rank_summary(
        _cs_data, combined_data["method_metadata"],
        selected_methods=set(foreground_methods),
    )
```

- [ ] **Step 6: Update cs_beta_trace_cell**

Find the cell that calls `make_cs_beta_trace_summary` (around line 796). Change to use `cs_plot_data`:

```python
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    _summary = viz_utils.make_cs_beta_trace_summary(
        _cs_data, combined_data["method_metadata"],
        selected_methods=set(foreground_methods),
        selected_threshold=selected_threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_ser_log_bf,
    )
```

- [ ] **Step 7: Update cs_histograms_cell**

Find the cell that calls `render_cs_histograms` (around line 751). It currently reads `cs_size_histogram` and `ser_log_bf_histogram`. Extract these from `cs_plot_data` using `cs_sizes` at beta=0.95 (index 45 in CS_BETA_GRID since 0.95 = 0.50 + 45*0.01):

```python
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    if not _cs_data.is_empty():
        _BETA_095_IDX = 45  # CS_BETA_GRID[45] == 0.95
        _cs_size_hist = _cs_data.with_columns(
            pl.col("cs_sizes").list.get(_BETA_095_IDX).alias("cs_size")
        ).select("method", "threshold", "cs_size")
        _ser_log_bf_hist = _cs_data.select("method", "threshold", "ser_log_bf")
    else:
        _cs_size_hist = pl.DataFrame(schema={"method": pl.String, "threshold": pl.Float64, "cs_size": pl.Int64})
        _ser_log_bf_hist = pl.DataFrame(schema={"method": pl.String, "threshold": pl.Float64, "ser_log_bf": pl.Float64})
    cs_histograms_chart = viz_utils.render_cs_histograms(_cs_size_hist, _ser_log_bf_hist)
```

- [ ] **Step 8: Update cs_power_fdp_cell**

Find the cell that reads `combined_data["cs_raw"]` (around line 844). Derive CS-based power/FDP from `cs_plot_data`:

```python
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    if not _cs_data.is_empty():
        _BETA_095_IDX = 45
        _cs_raw2 = _cs_data.with_columns(
            pl.col("cs_sizes").list.get(_BETA_095_IDX).alias("cs_size"),
            (pl.col("rank_of_causal").list.min() < pl.col("cs_sizes").list.get(_BETA_095_IDX)).alias("causal_in_cs"),
        ).select("sample_id", "method", "threshold", "l", "cs_size", "causal_in_cs", "ser_log_bf")
    else:
        _cs_raw2 = pl.DataFrame(schema={
            "sample_id": pl.String, "method": pl.String, "threshold": pl.Float64,
            "l": pl.Int64, "cs_size": pl.Int64, "causal_in_cs": pl.Boolean, "ser_log_bf": pl.Float64,
        })
```

(Preserve the rest of the existing cs_power_fdp_cell logic that filters and renders.)

- [ ] **Step 9: Run dashboard notebook load test**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_plot_ready.py::test_dashboard_notebook_module_loads -v
```
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add twogroup_experiments/notebooks/dashboard.py
git commit -m "refactor: update dashboard to use pip_plot_data/cs_plot_data keys"
```

---

### Task 8: Remove mass_above_causal from viz_utils and dashboard

**Files:**
- Modify: `viz_utils.py`
- Modify: `notebooks/dashboard.py`

- [ ] **Step 1: Remove three functions from viz_utils.py**

Delete:
- `make_mass_above_causal_summary` (line 881 — ~40 lines)
- `_plot_mass_above_causal_on_ax` (line 924 — ~35 lines)
- `render_mass_above_causal_chart` (line 957 — ~50 lines)

- [ ] **Step 2: Remove two cells from dashboard.py**

Delete:
- `mass_above_causal_heading_cell` (the `@app.cell` decorator + function around line 698)
- `mass_above_causal_cell` (the `@app.cell` decorator + function around line 706)

- [ ] **Step 3: Run dashboard load test**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_plot_ready.py::test_dashboard_notebook_module_loads -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add twogroup_experiments/viz_utils.py twogroup_experiments/notebooks/dashboard.py
git commit -m "refactor: remove mass_above_causal from viz_utils and dashboard"
```

---

### Task 9: Remove legacy utils + stale tests

**Files:**
- Modify: `utils.py`
- Modify: `tests/test_twogroup_experiments.py`

- [ ] **Step 1: Remove from utils.py**

Delete these items from `utils.py`:
- `build_plot_data_frames` function (lines 195–310)
- `_plot_frame` function (lines 313–316)
- `PIP_THRESHOLD_PLOT_DATA`, `CAUSAL_PIP_PLOT_DATA`, `CS_COMPONENT_PLOT_DATA`, `CS_TRUTH_PLOT_DATA`, `PLOT_DATA_FILENAMES` constants (lines 41–50)
- `PIP_THRESHOLD_GRID` constant (line 51) — now embedded in `build_pip_plot_data`
- `PIP_THRESHOLD_PLOT_BASE_SCHEMA`, `CAUSAL_PIP_PLOT_BASE_SCHEMA`, `CS_COMPONENT_PLOT_BASE_SCHEMA`, `CS_TRUTH_PLOT_BASE_SCHEMA` dicts (lines 53–100)
- `symlink_plot_data_outputs` function (lines 362–370)

Keep: `CS_BETA_GRID` (line 52) — used by `build_cs_plot_data` and viz.

- [ ] **Step 2: Remove 4 build_plot_data_frames tests from test_twogroup_experiments.py**

Delete:
- `test_build_plot_data_frames_produces_four_tables`
- `test_build_plot_data_frames_pip_threshold_power_is_one_for_causal_pip_above_threshold`
- `test_build_plot_data_frames_fdp_is_zero_when_only_causal_selected`
- `test_build_plot_data_frames_propagates_specs_to_all_outputs`

- [ ] **Step 3: Run all tests**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/ -v --ignore=twogroup_experiments/tests/test_twogroup_experiments.py -x
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_twogroup_experiments.py -k "not config_registry" -v
```
Expected: All tests that do not require file I/O pass.

- [ ] **Step 4: Commit**

```bash
git add twogroup_experiments/utils.py twogroup_experiments/tests/test_twogroup_experiments.py
git commit -m "refactor: remove legacy build_plot_data_frames and dead utils constants"
```

---

### Task 10: Final validation

**Files:** read-only checks

- [ ] **Step 1: Confirm CS_BETA_GRID still exported from utils.py**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -c "from twogroup_experiments.utils import CS_BETA_GRID; print(len(CS_BETA_GRID))"
```
Expected: `50`

- [ ] **Step 2: Run full test suite**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/ -v -k "not config_registry and not manifest"
```
Expected: All tests pass except pre-existing failures unrelated to this change.

- [ ] **Step 3: Verify dashboard loads clean**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -m pytest twogroup_experiments/tests/test_plot_ready.py::test_dashboard_notebook_module_loads -v
```
Expected: PASS.

- [ ] **Step 4: Spot-check imports**

```bash
cd /Users/ktayeb/research/gibss-experiments && .venv/bin/python -c "
from twogroup_experiments.plot_ready import build_pip_plot_data, build_cs_plot_data, load_plot_ready_collection
from twogroup_experiments.viz_utils import make_causal_rank_summary, make_cs_beta_trace_summary, expand_pip_calibration_from_compact
print('imports OK')
"
```
Expected: `imports OK`

- [ ] **Step 5: Commit summary**

```bash
git add .
git commit -m "chore: final cleanup — plot data redesign complete"
```

---

## Self-Review

**Spec coverage:**
- ✅ `cs_plot_data` schema: sample_id, method, threshold, l, ser_log_bf, causal_indices, rank_of_causal, mass_above_causal, cs_sizes — Task 3
- ✅ `pip_plot_data` schema: sample_id, method, threshold, causal_indices, causal_pips, pip_bin_counts, pip_bin_causal_counts, power_at_threshold, fdp_at_threshold — Task 2
- ✅ `core.py` fit_summary stripped — Task 1
- ✅ `load_collection_fits_with_specs` column select updated — Task 4
- ✅ `load_plot_ready_collection` names updated — Task 4
- ✅ 12 old functions removed from plot_ready.py — Task 4
- ✅ `make_causal_rank_summary` rewritten — Task 5
- ✅ `_expand_cs_to_beta_rows` + `make_cs_beta_trace_summary` rewritten — Task 5
- ✅ PIP expansion helpers added — Task 6
- ✅ Dashboard `combined_data` updated and all 7 cell functions updated — Task 7
- ✅ mass_above_causal removed from viz_utils + dashboard — Task 8
- ✅ `build_plot_data_frames` + dead constants removed from utils.py — Task 9
- ✅ Stale tests removed, new tests added — Tasks 2, 3, 4, 9

**Placeholder scan:** None found.

**Type consistency:** `CS_BETA_GRID` is used in `build_cs_plot_data` (Task 3) and `_expand_cs_to_beta_rows` (Task 5) — both import from `utils`. `_PIP_THRESHOLD_GRID` is embedded as a local list `[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]` in `build_pip_plot_data` (10 values) and `expand_power_fdp_from_compact` uses `utils.PIP_THRESHOLD_GRID` — note these are **different grids**: the old `PIP_THRESHOLD_GRID` in utils.py had 999 values (`np.arange(0.001, 1.0, 0.001)`), but `build_pip_plot_data` uses only 10 values. **Fix:** In `build_pip_plot_data`, keep the 10-value grid as a local constant `_PIP_THRESHOLD_GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]`. In `expand_power_fdp_from_compact`, import this same constant from plot_ready or define it locally as the same 10-value list. Do NOT import the 999-value `PIP_THRESHOLD_GRID` from utils.py in viz_utils.

**Fix for expand_power_fdp_from_compact:** Replace `from utils import PIP_THRESHOLD_GRID as _GRID` with:
```python
_GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]
```
(The array length must match `len(row["power_at_threshold"])` which is 10.)
