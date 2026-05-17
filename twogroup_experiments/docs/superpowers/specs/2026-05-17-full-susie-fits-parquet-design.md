# Full SuSiE Fit in fits.parquet — Design Spec

## Problem

`core.py` hardcodes `state.single_effects[0]` in `_extract_ser_struct`, `_make_cs_struct`, and `_make_fit_summary_struct`. For L=5 SuSiE fits, effects 1–4 are silently discarded. All downstream plots (power, FDP, CS coverage, causal rank, causal PIP) evaluate only effect 0, making L=5 and L=1 methods comparable only by accident. Marginal PIPs are also wrong: the true marginal PIP is `1 - prod_l(1 - alpha_lj)`, not `alpha_0j`.

## Goal

Store the complete SuSiE posterior in fits.parquet with no loss of per-effect information, while keeping the row-per-(replicate, method, threshold) structure.

## Schema Change

### Columns removed

| Column | Reason |
|---|---|
| `ser_posterior` | Single-effect struct (effect 0 only) |
| `credible_set` | Single-effect struct (effect 0 only) |

### Columns added

| Column | Type | Description |
|---|---|---|
| `single_effects` | `List[Struct]` | L structs, one per effect. Each struct: `{alpha: List[Float64], mu: List[Float64], var: List[Float64], prior_variance: Float64, marginal_log_likelihood: Float64, null_log_likelihood: Float64, ser_log_bf: Float64}` |
| `credible_sets` | `List[Struct]` | L structs, one per effect. Each struct: `{cs: List[Int64], cs_size: Int64, causal_indices: List[Int64], causal_in_cs: Boolean, top_feature: Int64, top_feature_is_causal: Boolean}` |

### Columns modified

| Column | Change |
|---|---|
| `fit_summary` | Gains `marginal_pip: List[Float64]` (length p). `causal_pip` and `max_pip` now derived from `marginal_pip` not `alpha[0]`. All other fields (`n_selected`, `n_iter`, `converged`) unchanged. |

### Columns unchanged

`family_state`, `two_group_state` — global per fit, not per-effect. `replicate`, `method`, `threshold` — keys, unchanged.

## core.py Changes

### `_extract_ser_struct(state, l: int) -> dict`

Add `l` parameter. Use `state.single_effects[l]` and `state.ser_log_bayes_factor[l]`.

### `_make_cs_struct(state, simulation, l: int, coverage=0.95) -> dict`

Add `l` parameter. Use `state.single_effects[l].alpha` and `state.get_credible_sets(coverage)[l]`.

### `_make_fit_summary_struct(state, simulation, n_selected) -> dict`

Compute marginal PIP across all L effects:

```python
L = len(state.single_effects)
alphas = np.stack([np.asarray(state.single_effects[l].alpha, dtype=float) for l in range(L)])
marginal_pip = 1.0 - np.prod(1.0 - alphas, axis=0)
causal_indices = np.asarray(simulation.causal_indices, dtype=int)
return {
    "marginal_pip": marginal_pip.tolist(),
    "causal_pip": float(np.max(marginal_pip[causal_indices])),
    "max_pip": float(np.max(marginal_pip)),
    "n_selected": None if n_selected is None else int(n_selected),
    "n_iter": int(state.n_iter),
    "converged": bool(state.converged),
}
```

### `summarize_cox_method`, `summarize_logistic_method`, `summarize_twogroup_method`

Each returns one dict (unchanged return type). Build `single_effects` and `credible_sets` lists internally:

```python
state = fit_obj["state"]
L = len(state.single_effects)
fit_summary = _make_fit_summary_struct(state, simulation, fit_obj["n_selected"])
return {
    "threshold": fit_obj["threshold"],
    "single_effects": [_extract_ser_struct(state, l) for l in range(L)],
    "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(L)],
    "family_state": _extract_family_state_struct(state),
    "two_group_state": _extract_twogroup_state_struct(state),
    "fit_summary": fit_summary,
}
```

`summarize_method_spec` and `fit_batch_method` in utils.py — **no changes**.

## plot_ready.py Changes

### PIP-based summaries — use `fit_summary.marginal_pip`

`summarize_pip_calibration_per_sample` and `summarize_power_fdp_per_sample`: replace `row["ser_posterior"]["alpha"]` with `row["fit_summary"]["marginal_pip"]`.

`summarize_causal_pip_per_sample`: no code change — already reads `fit_summary.causal_pip` via Polars struct field access. The **value changes** from `max(alpha_0[causal_indices])` to `max(marginal_pip[causal_indices])` because `_make_fit_summary_struct` now computes it from the marginal PIP. This is the correct behavior.

### Alpha/CS-based summaries — loop over `single_effects` / `credible_sets`

`summarize_cs_beta_trace`, `summarize_cs_raw_per_sample`, `summarize_cs_size_histogram_observations`, `summarize_ser_log_bf_histogram_observations`:

Add inner loop over effects:

```python
for row in fits_with_sid.iter_rows(named=True):
    for l, effect in enumerate(row["single_effects"]):
        alpha = np.asarray(effect["alpha"], dtype=float)
        cs_struct = row["credible_sets"][l]
        ser_log_bf = float(effect["ser_log_bf"])
        # ... existing computation ...
        rows.append({..., "l": l})
```

Each output schema gains `"l": pl.Int64`.

### `summarize_cs_beta_trace` — revert `mass_above_causal`

Remove `mass_above_causal` column added in a prior session. This will be handled by a separate `causal_stats` table (future work).

## viz_utils.py / dashboard.py — No changes required

`make_causal_rank_summary` groups by `(collection_name, sample_id, method, threshold)` and takes `min(cs_size)` where `covered=True` — naturally finds the best effect across all L. No change needed.

`make_cs_beta_trace_summary` and all chart renderers — unchanged.

Dashboard cells referencing `cs_beta_trace`, `cs_raw`, histograms — unchanged (the `l` column is transparent to aggregation).

## Operational Impact

All existing `fits.parquet` files are **invalidated** by this schema change and must be regenerated. All downstream `plot_ready/` parquets must also be regenerated. The snakemake pipeline handles this via output path dependencies — delete existing fits to trigger re-run, or use `snakemake --rerun-incomplete`.

## Out of Scope

- `causal_stats` table (per-effect × per-causal statistics: causal rank, mass above causal, causal alpha) — separate spec and implementation
- Threshold absorbed into method name — decided against
