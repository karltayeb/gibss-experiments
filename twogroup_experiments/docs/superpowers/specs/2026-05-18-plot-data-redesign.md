# Plot Data Redesign — Design Spec

## Problem

`plot_ready.py` produces seven separate tables, all referencing stale schema columns (`ser_posterior`, `credible_set`) that no longer exist after the fits.parquet refactor. Functions only handle L=1 (single effect). Three tables (`pip_calibration`, `power_fdp`, `causal_pip`) each independently iterate over all rows and recompute the same marginal PIP. Four CS tables (`cs_beta_trace`, `cs_raw`, `cs_size_histogram`, `ser_log_bf_histogram`) each iterate over fits and emit wide flat rows, repeating the same `ser_log_bf` and causal metadata for every beta value or CS size observation.

## Goal

Two compact parquet tables — `cs_plot_data` and `pip_plot_data` — that handle arbitrary L, store pre-aggregated arrays rather than exploded rows, and require only simple array operations at plot time.

---

## `cs_plot_data.parquet`

One row per **(sample_id, method, threshold, l)**.

### Schema

| Column | Type | Description |
|---|---|---|
| `sample_id` | String | |
| `method` | String | |
| `threshold` | Float64 | |
| `l` | Int64 | Effect index (0-based) |
| `ser_log_bf` | Float64 | `marginal_log_likelihood - null_log_likelihood` for effect l |
| `causal_indices` | List[Int64] | Ground-truth causal feature indices, length m |
| `rank_of_causal` | List[Int64] | Rank of each causal in alpha_l ordering (0 = highest alpha), length m |
| `mass_above_causal` | List[Float64] | Cumulative alpha_l mass above each causal (= min beta to include), length m |
| `cs_sizes` | List[Int64] | CS size at each beta in `CS_BETA_GRID`, length 50 |

### Derivations at plot time

- `covered[i, b] = rank_of_causal[i] < cs_sizes[b]` — no recomputation
- `power[b] = mean(rank_of_causal[i] < cs_sizes[b])` over causals
- `causal_rank = min(rank_of_causal)` — rank of best causal across causals
- `cs_size at fixed beta` = `cs_sizes[b_idx]`
- `log BF distribution` = `ser_log_bf`

### Replaces

`cs_beta_trace`, `cs_raw`, `cs_size_histogram`, `ser_log_bf_histogram`

### Build logic

```python
for row in fits_with_sid.iter_rows(named=True):
    for l, effect in enumerate(row["single_effects"]):
        alpha = np.asarray(effect["alpha"], dtype=float)
        cs_struct = row["credible_sets"][l]
        causal_indices = cs_struct["causal_indices"]

        order = np.argsort(-alpha)
        cumulative = np.cumsum(alpha[order])
        rank_of = {int(feat): rk for rk, feat in enumerate(order.tolist())}

        rank_of_causal = [rank_of[ci] for ci in causal_indices]
        mass_above_causal = [
            float(cumulative[rk - 1]) if rk > 0 else 0.0
            for rk in rank_of_causal
        ]
        cs_sizes = [
            int(np.searchsorted(cumulative, beta, side="left") + 1)
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
```

---

## `pip_plot_data.parquet`

One row per **(sample_id, method, threshold)**.

### Schema

| Column | Type | Description |
|---|---|---|
| `sample_id` | String | |
| `method` | String | |
| `threshold` | Float64 | |
| `causal_indices` | List[Int64] | Ground-truth causal feature indices, length m |
| `causal_pips` | List[Float64] | Marginal PIP at each causal index, length m |
| `pip_bin_counts` | List[Int64] | Number of features per PIP bin (20 bins, width 0.05), length 20 |
| `pip_bin_causal_counts` | List[Int64] | Number of causal features per PIP bin, length 20 |
| `power_at_threshold` | List[Float64] | Power at each threshold in `PIP_THRESHOLD_GRID`, length 10 |
| `fdp_at_threshold` | List[Float64] | FDP at each threshold in `PIP_THRESHOLD_GRID`, length 10 |

### Derivations at plot time

- PIP calibration: `sum(pip_bin_causal_counts[b]) / sum(pip_bin_counts[b])` grouped by (method, threshold, bin)
- Power/FDP curve: `mean(power_at_threshold[i])` and `mean(fdp_at_threshold[i])` grouped by (method, threshold, threshold_idx)
- Causal PIP: `mean(max(causal_pips))` grouped by (method, threshold)

### Replaces

`pip_calibration`, `power_fdp`, `causal_pip`

### Build logic

```python
# marginal_pip: 1 - prod_l(1 - alpha_lj) across all L effects
alphas = np.stack([
    np.asarray(e["alpha"], dtype=float)
    for e in row["single_effects"]
])
marginal_pip = 1.0 - np.prod(1.0 - alphas, axis=0)

causal_indices = list(simrow["simulation"]["causal_indices"])
causal_pips = [float(marginal_pip[ci]) for ci in causal_indices]

bin_indices = np.clip((marginal_pip * 20).astype(int), 0, 19)
is_causal = np.zeros(len(marginal_pip), dtype=bool)
is_causal[causal_indices] = True
pip_bin_counts = [int((bin_indices == b).sum()) for b in range(20)]
pip_bin_causal_counts = [int(((bin_indices == b) & is_causal).sum()) for b in range(20)]

n_causal = max(len(causal_indices), 1)
power_at_threshold = []
fdp_at_threshold = []
for t in PIP_THRESHOLD_GRID:
    selected = marginal_pip >= t
    sel_causal = int(selected[causal_indices].sum())
    sel_total = int(selected.sum())
    power_at_threshold.append(float(sel_causal / n_causal))
    fdp_at_threshold.append(float((sel_total - sel_causal) / max(sel_total, 1)))
```

---

## `plot_ready.py` Changes

### Functions removed

| Function | Replaced by |
|---|---|
| `summarize_pip_calibration_per_sample` | `build_pip_plot_data` |
| `aggregate_pip_calibration` | plot-time aggregation |
| `summarize_power_fdp_per_sample` | `build_pip_plot_data` |
| `aggregate_power_fdp` | plot-time aggregation |
| `summarize_causal_pip_per_sample` | `build_pip_plot_data` |
| `aggregate_causal_pip` | plot-time aggregation |
| `summarize_cs_beta_trace` | `build_cs_plot_data` |
| `summarize_cs_raw_per_sample` | `build_cs_plot_data` |
| `summarize_cs_size_histogram_observations` | `build_cs_plot_data` |
| `finalize_cs_size_histogram` | plot-time aggregation |
| `summarize_ser_log_bf_histogram_observations` | `build_cs_plot_data` |
| `finalize_ser_log_bf_histogram` | plot-time aggregation |

### Functions added

```python
def build_cs_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame: ...

def build_pip_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame: ...
```

### `load_collection_fits_with_specs` column select

Change: `ser_posterior` → `single_effects`, `credible_set` → `credible_sets`.

### `load_plot_ready_collection` names

Replace `pip_calibration`, `power_fdp`, `causal_pip`, `cs_raw`, `cs_beta_trace`, `cs_size_histogram`, `ser_log_bf_histogram` with `cs_plot_data`, `pip_plot_data`.

---

## `core.py` Changes

`_make_fit_summary_struct`: remove `marginal_pip`, `causal_pip`, `max_pip`. Return only `{n_selected, n_iter, converged}`.

---

## `viz_utils.py` / `dashboard.py` Changes

Functions that read old table names need updating to read from `cs_plot_data` and `pip_plot_data`. Existing aggregation logic that previously ran in `plot_ready.py` moves to viz-layer helpers or plot-time Polars expressions.

Remove: `make_mass_above_causal_summary`, `_plot_mass_above_causal_on_ax`, `render_mass_above_causal_chart` from `viz_utils.py`.
Remove: `mass_above_causal_heading_cell`, `mass_above_causal_cell` from `dashboard.py`.

---

## `utils.py` Changes

Remove `build_plot_data_frames` (legacy function, references old schema).

---

## Tests

- Remove 4 `test_build_plot_data_frames_*` tests from `test_twogroup_experiments.py`
- Update `test_plot_ready.py`: replace old schema tests with tests for `build_cs_plot_data` and `build_pip_plot_data`
  - `test_build_cs_plot_data_schema`: verify columns, one row per (sample, method, threshold, l)
  - `test_build_cs_plot_data_cs_sizes_length`: `len(cs_sizes) == len(CS_BETA_GRID)` for every row
  - `test_build_pip_plot_data_schema`: verify columns, one row per (sample, method, threshold)
  - `test_build_pip_plot_data_causal_pips_correct`: marginal_pip at causal indices matches `1 - prod_l(1 - alpha_lj)`

---

## Operational Impact

All existing `plot_ready/` parquets are invalidated. Snakemake pipeline outputs reference `cs_plot_data.parquet` and `pip_plot_data.parquet` — update rule output paths.

## Out of Scope

- `causal_stats` table (per-effect × per-causal × replicate statistics beyond what's in `cs_plot_data`)
- viz_utils aggregation helpers — schema compatible, updated in a follow-on task
