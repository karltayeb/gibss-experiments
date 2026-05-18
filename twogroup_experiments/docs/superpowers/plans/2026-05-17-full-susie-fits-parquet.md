# Full SuSiE Fit in fits.parquet — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-effect `ser_posterior`/`credible_set` columns in fits.parquet with `single_effects`/`credible_sets` list-of-struct columns that store all L SuSiE components, and compute marginal PIP correctly across all L effects.

**Architecture:** core.py extraction helpers are parameterized by effect index `l`; summarize functions build lists of L structs; plot_ready.py PIP-based summaries read `fit_summary.marginal_pip`; alpha-based summaries loop over `single_effects`/`credible_sets` lists and emit one row per effect with an `l` column. Legacy `build_plot_data_frames` in utils.py is removed (not used in production pipeline). Previously-added `mass_above_causal` column in `cs_beta_trace` is reverted (future `causal_stats` table).

**Tech Stack:** Python 3.12, NumPy, Polars, pytest, gibss (IBSS engine)

---

## File Map

| File | Change |
|---|---|
| `core.py` | Parameterize `_extract_ser_struct(state, l)`, `_make_cs_struct(state, simulation, l)`; update `_make_fit_summary_struct` for marginal PIP; update all three `summarize_*` functions to build list columns |
| `plot_ready.py` | Update `load_collection_fits_with_specs` column select; update PIP-based summaries to use `fit_summary.marginal_pip`; update alpha-based summaries to loop over `single_effects`/`credible_sets`, add `l` column; revert `mass_above_causal` |
| `viz_utils.py` | Remove `make_mass_above_causal_summary`, `_plot_mass_above_causal_on_ax`, `render_mass_above_causal_chart` |
| `notebooks/dashboard.py` | Remove `mass_above_causal_heading_cell` and `mass_above_causal_cell` |
| `utils.py` | Remove legacy `build_plot_data_frames` (reads old `ser_posterior` column, not used in production) |
| `tests/test_twogroup_experiments.py` | Remove four `test_build_plot_data_frames_*` tests; update `test_fit_batch_method_returns_one_row_per_replicate` |
| `tests/test_plot_ready.py` | Update `test_summarize_cs_raw_per_sample_columns`; add `test_summarize_cs_beta_trace_has_l_column`; add `test_summarize_pip_calibration_uses_marginal_pip` |

---

### Task 1: Write failing tests for new fits.parquet schema

**Files:**
- Modify: `tests/test_twogroup_experiments.py`

Establish what the new schema must look like before changing any production code.

- [ ] **Step 1: Add schema tests to `tests/test_twogroup_experiments.py`**

Add these two tests after `test_fit_batch_method_returns_one_row_per_replicate` (line ~244):

```python
def test_fit_batch_method_schema_has_single_effects_list():
    from config import _logistic_threshold_method_spec
    df = fit_batch_method(
        _tiny_simulation_spec(),
        method_spec=_logistic_threshold_method_spec(threshold=2.0, L=1),
        replicates=(0,),
    )
    assert "single_effects" in df.columns
    assert "credible_sets" in df.columns
    assert "ser_posterior" not in df.columns
    assert "credible_set" not in df.columns
    row = df.row(0, named=True)
    assert isinstance(row["single_effects"], list)
    assert isinstance(row["credible_sets"], list)
    assert len(row["single_effects"]) == 1
    assert len(row["credible_sets"]) == 1
    effect = row["single_effects"][0]
    assert "alpha" in effect
    assert "ser_log_bf" in effect
    cs = row["credible_sets"][0]
    assert "cs_size" in cs
    assert "causal_in_cs" in cs


def test_fit_batch_method_L5_has_5_effects():
    from config import _logistic_threshold_method_spec
    df = fit_batch_method(
        _tiny_simulation_spec(),
        method_spec=_logistic_threshold_method_spec(threshold=2.0, L=5),
        replicates=(0,),
    )
    row = df.row(0, named=True)
    assert len(row["single_effects"]) == 5
    assert len(row["credible_sets"]) == 5


def test_fit_batch_method_marginal_pip_correct_for_L1():
    """For L=1, marginal_pip == alpha (product formula is identity)."""
    from config import _logistic_threshold_method_spec
    import numpy as np
    df = fit_batch_method(
        _tiny_simulation_spec(),
        method_spec=_logistic_threshold_method_spec(threshold=2.0, L=1),
        replicates=(0,),
    )
    row = df.row(0, named=True)
    alpha = np.asarray(row["single_effects"][0]["alpha"])
    marginal_pip = np.asarray(row["fit_summary"]["marginal_pip"])
    assert marginal_pip.shape == alpha.shape
    np.testing.assert_allclose(marginal_pip, alpha, atol=1e-12)


def test_fit_batch_method_marginal_pip_correct_for_L5():
    """For L=5, marginal_pip >= any single alpha_l (never less)."""
    from config import _logistic_threshold_method_spec
    import numpy as np
    df = fit_batch_method(
        _tiny_simulation_spec(),
        method_spec=_logistic_threshold_method_spec(threshold=2.0, L=5),
        replicates=(0,),
    )
    row = df.row(0, named=True)
    marginal_pip = np.asarray(row["fit_summary"]["marginal_pip"])
    assert marginal_pip.shape[0] == 3  # identity_design_sampler gives p=3
    assert np.all(marginal_pip >= 0.0)
    assert np.all(marginal_pip <= 1.0)
    for effect in row["single_effects"]:
        alpha_l = np.asarray(effect["alpha"])
        assert np.all(marginal_pip >= alpha_l - 1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_twogroup_experiments.py::test_fit_batch_method_schema_has_single_effects_list tests/test_twogroup_experiments.py::test_fit_batch_method_L5_has_5_effects tests/test_twogroup_experiments.py::test_fit_batch_method_marginal_pip_correct_for_L1 tests/test_twogroup_experiments.py::test_fit_batch_method_marginal_pip_correct_for_L5 -v
```

Expected: 4 FAILED (KeyError or AssertionError on column names)

---

### Task 2: Update core.py extraction helpers

**Files:**
- Modify: `core.py:168-232`

- [ ] **Step 1: Write the updated `_extract_ser_struct`**

Replace:
```python
def _extract_ser_struct(state: Any) -> dict[str, Any]:
    effect = state.single_effects[0]
    return {
        "mu": _to_python(effect.mu),
        "var": _to_python(effect.var),
        "alpha": _to_python(effect.alpha),
        "prior_variance": float(effect.prior_variance),
        "marginal_log_likelihood": float(effect.marginal_log_likelihood),
        "null_log_likelihood": float(effect.null_log_likelihood),
        "ser_log_bf": float(np.asarray(state.ser_log_bayes_factor[0])),
    }
```

With:
```python
def _extract_ser_struct(state: Any, l: int) -> dict[str, Any]:
    effect = state.single_effects[l]
    return {
        "mu": _to_python(effect.mu),
        "var": _to_python(effect.var),
        "alpha": _to_python(effect.alpha),
        "prior_variance": float(effect.prior_variance),
        "marginal_log_likelihood": float(effect.marginal_log_likelihood),
        "null_log_likelihood": float(effect.null_log_likelihood),
        "ser_log_bf": float(np.asarray(state.ser_log_bayes_factor[l])),
    }
```

- [ ] **Step 2: Write the updated `_make_cs_struct`**

Replace:
```python
def _make_cs_struct(
    state: Any, simulation: TwoGroupSimulation, coverage: float = 0.95
) -> dict[str, Any]:
    alpha = np.asarray(state.single_effects[0].alpha, dtype=float)
    cs = tuple(int(idx) for idx in state.get_credible_sets(coverage=coverage)[0])
    top_feature = int(np.argmax(alpha))
    causal_indices = [int(idx) for idx in simulation.causal_indices]
    return {
        "cs": list(cs),
        "cs_size": len(cs),
        "causal_indices": causal_indices,
        "causal_in_cs": any(idx in cs for idx in causal_indices),
        "top_feature": top_feature,
        "top_feature_is_causal": top_feature in causal_indices,
    }
```

With:
```python
def _make_cs_struct(
    state: Any, simulation: TwoGroupSimulation, l: int, coverage: float = 0.95
) -> dict[str, Any]:
    alpha = np.asarray(state.single_effects[l].alpha, dtype=float)
    cs = tuple(int(idx) for idx in state.get_credible_sets(coverage=coverage)[l])
    top_feature = int(np.argmax(alpha))
    causal_indices = [int(idx) for idx in simulation.causal_indices]
    return {
        "cs": list(cs),
        "cs_size": len(cs),
        "causal_indices": causal_indices,
        "causal_in_cs": any(idx in cs for idx in causal_indices),
        "top_feature": top_feature,
        "top_feature_is_causal": top_feature in causal_indices,
    }
```

- [ ] **Step 3: Write the updated `_make_fit_summary_struct`**

Replace:
```python
def _make_fit_summary_struct(
    state: Any, simulation: TwoGroupSimulation, n_selected: int | None
) -> dict[str, Any]:
    alpha = np.asarray(state.single_effects[0].alpha, dtype=float)
    causal_alpha = alpha[np.asarray(simulation.causal_indices, dtype=int)]
    return {
        "causal_pip": float(np.max(causal_alpha)),
        "max_pip": float(np.max(alpha)),
        "n_selected": None if n_selected is None else int(n_selected),
        "n_iter": int(state.n_iter),
        "converged": bool(state.converged),
    }
```

With:
```python
def _make_fit_summary_struct(
    state: Any, simulation: TwoGroupSimulation, n_selected: int | None
) -> dict[str, Any]:
    L = len(state.single_effects)
    alphas = np.stack(
        [np.asarray(state.single_effects[l].alpha, dtype=float) for l in range(L)]
    )
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

- [ ] **Step 4: Run schema tests — still failing (summarize functions not updated yet)**

```bash
uv run pytest tests/test_twogroup_experiments.py::test_fit_batch_method_schema_has_single_effects_list -v
```

Expected: still FAIL — `ser_posterior` still in schema

---

### Task 3: Update core.py summarize functions

**Files:**
- Modify: `core.py:259-366`

All three `summarize_*` functions return one dict with `single_effects` and `credible_sets` as lists. `fit_summary` is computed once and shared.

- [ ] **Step 1: Replace `summarize_cox_method`**

Replace:
```python
def summarize_cox_method(
    fit_obj,
    simulation: TwoGroupSimulation,
    *,
    threshold,
    time_sign,
    L=1,
):
    del time_sign, threshold, L
    state = fit_obj["state"]
    return {
        "threshold": fit_obj["threshold"],
        "ser_posterior": _extract_ser_struct(state),
        "credible_set": _make_cs_struct(state, simulation),
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(
            state, simulation, fit_obj["n_selected"]
        ),
    }
```

With:
```python
def summarize_cox_method(
    fit_obj,
    simulation: TwoGroupSimulation,
    *,
    threshold,
    time_sign,
    L=1,
):
    del time_sign, threshold, L
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }
```

- [ ] **Step 2: Replace `summarize_logistic_method`**

Replace:
```python
def summarize_logistic_method(
    fit_obj,
    simulation: TwoGroupSimulation,
    *,
    response_source,
    threshold=None,
    L=1,
):
    del response_source, threshold, L
    state = fit_obj["state"]
    return {
        "threshold": fit_obj["threshold"],
        "ser_posterior": _extract_ser_struct(state),
        "credible_set": _make_cs_struct(state, simulation),
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(
            state, simulation, fit_obj["n_selected"]
        ),
    }
```

With:
```python
def summarize_logistic_method(
    fit_obj,
    simulation: TwoGroupSimulation,
    *,
    response_source,
    threshold=None,
    L=1,
):
    del response_source, threshold, L
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }
```

- [ ] **Step 3: Replace `summarize_twogroup_method`**

Replace:
```python
def summarize_twogroup_method(fit_obj, simulation: TwoGroupSimulation, *, f1, L=1):
    del f1, L
    state = fit_obj["state"]
    return {
        "threshold": fit_obj["threshold"],
        "ser_posterior": _extract_ser_struct(state),
        "credible_set": _make_cs_struct(state, simulation),
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(
            state, simulation, fit_obj["n_selected"]
        ),
    }
```

With:
```python
def summarize_twogroup_method(fit_obj, simulation: TwoGroupSimulation, *, f1, L=1):
    del f1, L
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }
```

- [ ] **Step 4: Run schema tests — all four should now pass**

```bash
uv run pytest tests/test_twogroup_experiments.py::test_fit_batch_method_schema_has_single_effects_list tests/test_twogroup_experiments.py::test_fit_batch_method_L5_has_5_effects tests/test_twogroup_experiments.py::test_fit_batch_method_marginal_pip_correct_for_L1 tests/test_twogroup_experiments.py::test_fit_batch_method_marginal_pip_correct_for_L5 -v
```

Expected: 4 PASSED

- [ ] **Step 5: Run full test suite — check no regressions in previously-passing tests**

```bash
uv run pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: same 50 passing as baseline (7 pre-existing failures unchanged)

- [ ] **Step 6: Commit**

```bash
git add core.py tests/test_twogroup_experiments.py
git commit -m "feat: store full SuSiE fit in fits.parquet via single_effects/credible_sets list columns"
```

---

### Task 4: Update plot_ready.py — load function and PIP-based summaries

**Files:**
- Modify: `plot_ready.py:15-36` (load function)
- Modify: `plot_ready.py:142-280` (pip calibration, power_fdp, causal_pip)
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing test for PIP-based summary using marginal_pip**

Add to `tests/test_plot_ready.py`:

```python
def test_summarize_pip_calibration_uses_marginal_pip():
    """marginal_pip from fit_summary drives pip calibration, not single_effects[0].alpha."""
    import numpy as np
    p = 3
    # marginal_pip has high probability on variable 1
    marginal_pip = [0.05, 0.90, 0.05]
    fits_df = pl.DataFrame({
        "method": ["logistic_threshold_L1"],
        "threshold": [1.0],
        "batch_hash": ["abc"],
        "replicate": [0],
        "single_effects": [[{"alpha": [0.33, 0.34, 0.33], "ser_log_bf": 1.0,
                              "mu": [0.0]*p, "var": [1.0]*p,
                              "prior_variance": 1.0,
                              "marginal_log_likelihood": -1.0,
                              "null_log_likelihood": -2.0}]],
        "fit_summary": [{"marginal_pip": marginal_pip, "causal_pip": 0.9,
                         "max_pip": 0.9, "n_selected": 10,
                         "n_iter": 5, "converged": True}],
    })
    sample_metadata = pl.DataFrame({
        "sample_id": ["abc::0"],
        "batch_hash": ["abc"],
        "replicate": [0],
    })
    simulations_by_batch = {
        "abc": pl.DataFrame({
            "replicate": [0],
            "simulation": [{"causal_indices": [1]}],
        })
    }

    result = plot_ready.summarize_pip_calibration_per_sample(
        fits_df, sample_metadata, simulations_by_batch
    )

    # variable 1 has marginal_pip=0.90 → pip_bin_index = min(int(0.90*20), 19) = 18
    causal_bin = result.filter(pl.col("pip_bin_index") == 18)
    assert causal_bin.height > 0
    assert causal_bin["n_causal_exact"].sum() == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_plot_ready.py::test_summarize_pip_calibration_uses_marginal_pip -v
```

Expected: FAIL — `KeyError: 'ser_posterior'`

- [ ] **Step 3: Update `load_collection_fits_with_specs` column select**

In `plot_ready.py`, replace:
```python
            frames.append(fits_df.select(
                "method", "threshold", "method_spec", "batch_hash", "replicate",
                "ser_posterior", "fit_summary", "credible_set",
            ))
```

With:
```python
            frames.append(fits_df.select(
                "method", "threshold", "method_spec", "batch_hash", "replicate",
                "single_effects", "fit_summary", "credible_sets",
            ))
```

- [ ] **Step 4: Update `summarize_pip_calibration_per_sample`**

Replace the loop body inside `summarize_pip_calibration_per_sample` (the line that reads `alpha`):
```python
        alpha = np.asarray(row["ser_posterior"]["alpha"], dtype=float)
```

With:
```python
        alpha = np.asarray(row["fit_summary"]["marginal_pip"], dtype=float)
```

- [ ] **Step 5: Update `summarize_power_fdp_per_sample`**

Replace:
```python
        alpha = np.asarray(row["ser_posterior"]["alpha"], dtype=float)
```

With:
```python
        alpha = np.asarray(row["fit_summary"]["marginal_pip"], dtype=float)
```

- [ ] **Step 6: Run pip calibration test — should now pass**

```bash
uv run pytest tests/test_plot_ready.py::test_summarize_pip_calibration_uses_marginal_pip -v
```

Expected: PASS

- [ ] **Step 7: Run full test suite — verify no regressions**

```bash
uv run pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: 50 passing (same baseline)

- [ ] **Step 8: Commit**

```bash
git add plot_ready.py tests/test_plot_ready.py
git commit -m "feat: PIP-based plot_ready summaries use marginal_pip across all L effects"
```

---

### Task 5: Update plot_ready.py — alpha-based summaries

**Files:**
- Modify: `plot_ready.py:320-430` (cs_beta_trace, cs_raw, histograms)
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Write failing test for `summarize_cs_raw_per_sample` with new schema**

Replace the existing `test_summarize_cs_raw_per_sample_columns` test in `tests/test_plot_ready.py`:

```python
def test_summarize_cs_raw_per_sample_columns():
    p = 3
    fits_df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L1"],
            "threshold": [1.0],
            "batch_hash": ["abc"],
            "replicate": [0],
            "credible_sets": [[
                {"causal_in_cs": True, "cs_size": 3,
                 "cs": [0, 1, 2], "causal_indices": [1],
                 "top_feature": 1, "top_feature_is_causal": True}
            ]],
            "single_effects": [[
                {"alpha": [0.1, 0.8, 0.1], "ser_log_bf": 2.5,
                 "mu": [0.0]*p, "var": [1.0]*p,
                 "prior_variance": 1.0,
                 "marginal_log_likelihood": -1.0,
                 "null_log_likelihood": -2.0}
            ]],
            "fit_summary": [{"marginal_pip": [0.1, 0.8, 0.1], "causal_pip": 0.8,
                             "max_pip": 0.8, "n_selected": 5,
                             "n_iter": 3, "converged": True}],
        }
    )
    sample_metadata = pl.DataFrame(
        {
            "sample_id": ["abc::0"],
            "batch_hash": ["abc"],
            "replicate": [0],
        }
    )

    result = plot_ready.summarize_cs_raw_per_sample(fits_df, sample_metadata)

    assert set(result.columns) >= {"sample_id", "method", "threshold", "l",
                                    "causal_in_cs", "cs_size", "ser_log_bf"}
    assert result["causal_in_cs"].dtype == pl.Boolean
    assert result["cs_size"].dtype == pl.Int64
    assert result["ser_log_bf"].dtype == pl.Float64
    assert result["l"].dtype == pl.Int64
    assert result.height == 1  # one effect


def test_summarize_cs_beta_trace_has_l_column():
    p = 3
    fits_df = pl.DataFrame(
        {
            "method": ["logistic_threshold_L1"],
            "threshold": [1.0],
            "batch_hash": ["abc"],
            "replicate": [0],
            "credible_sets": [[
                {"causal_in_cs": True, "cs_size": 1,
                 "cs": [1], "causal_indices": [1],
                 "top_feature": 1, "top_feature_is_causal": True}
            ]],
            "single_effects": [[
                {"alpha": [0.05, 0.90, 0.05], "ser_log_bf": 3.0,
                 "mu": [0.0]*p, "var": [1.0]*p,
                 "prior_variance": 1.0,
                 "marginal_log_likelihood": -0.5,
                 "null_log_likelihood": -2.0}
            ]],
            "fit_summary": [{"marginal_pip": [0.05, 0.90, 0.05], "causal_pip": 0.9,
                             "max_pip": 0.9, "n_selected": 5,
                             "n_iter": 3, "converged": True}],
        }
    )
    sample_metadata = pl.DataFrame(
        {
            "sample_id": ["abc::0"],
            "batch_hash": ["abc"],
            "replicate": [0],
        }
    )

    result = plot_ready.summarize_cs_beta_trace(fits_df, sample_metadata)

    assert "l" in result.columns
    assert result["l"].dtype == pl.Int64
    # one effect × 50 beta grid points
    from utils import CS_BETA_GRID
    assert result.height == len(CS_BETA_GRID)
    assert result["l"].unique().to_list() == [0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_plot_ready.py::test_summarize_cs_raw_per_sample_columns tests/test_plot_ready.py::test_summarize_cs_beta_trace_has_l_column -v
```

Expected: both FAIL

- [ ] **Step 3: Rewrite `summarize_cs_beta_trace` in `plot_ready.py`**

Replace the entire function:

```python
def summarize_cs_beta_trace(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold, l, beta) for all betas in CS_BETA_GRID."""
    from utils import CS_BETA_GRID

    empty_schema = {
        "sample_id": pl.String,
        "method": pl.String,
        "threshold": pl.Float64,
        "l": pl.Int64,
        "beta": pl.Float64,
        "cs_size": pl.Int64,
        "covered": pl.Boolean,
        "ser_log_bf": pl.Float64,
    }
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        causal_set = set(row["credible_sets"][0]["causal_indices"])
        for l, effect in enumerate(row["single_effects"]):
            alpha = np.asarray(effect["alpha"], dtype=float)
            ser_log_bf = float(effect["ser_log_bf"])
            order = np.argsort(-alpha)
            cumulative = np.cumsum(alpha[order])
            ordered_features = order.tolist()
            for beta in CS_BETA_GRID:
                cs_size = int(np.searchsorted(cumulative, beta, side="left") + 1)
                covered = bool(set(ordered_features[:cs_size]) & causal_set)
                rows.append({
                    "sample_id": row["sample_id"],
                    "method": row["method"],
                    "threshold": row["threshold"],
                    "l": l,
                    "beta": float(beta),
                    "cs_size": cs_size,
                    "covered": covered,
                    "ser_log_bf": ser_log_bf,
                })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)
```

Note: `causal_set` is the same for all effects (it's the simulation ground truth), taken from `credible_sets[0]["causal_indices"]`.

- [ ] **Step 4: Rewrite `summarize_cs_raw_per_sample` in `plot_ready.py`**

Replace the entire function:

```python
def summarize_cs_raw_per_sample(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold, l) with CS data per effect."""
    empty_schema = {
        "sample_id": pl.String,
        "method": pl.String,
        "threshold": pl.Float64,
        "l": pl.Int64,
        "causal_in_cs": pl.Boolean,
        "cs_size": pl.Int64,
        "ser_log_bf": pl.Float64,
    }
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        for l, (cs_struct, effect) in enumerate(
            zip(row["credible_sets"], row["single_effects"])
        ):
            rows.append({
                "sample_id": row["sample_id"],
                "method": row["method"],
                "threshold": row["threshold"],
                "l": l,
                "causal_in_cs": bool(cs_struct["causal_in_cs"]),
                "cs_size": int(cs_struct["cs_size"]),
                "ser_log_bf": float(effect["ser_log_bf"]),
            })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)
```

- [ ] **Step 5: Update `summarize_cs_size_histogram_observations`**

Replace:

```python
def summarize_cs_size_histogram_observations(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Build raw method x threshold x cs_size observations."""
    fits_with_sample_id = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    return fits_with_sample_id.select(
        "method",
        "threshold",
        pl.col("credible_set").struct.field("cs_size").alias("cs_size"),
    )
```

With:

```python
def summarize_cs_size_histogram_observations(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Build raw method x threshold x l x cs_size observations."""
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        for l, cs_struct in enumerate(row["credible_sets"]):
            rows.append({
                "method": row["method"],
                "threshold": row["threshold"],
                "l": l,
                "cs_size": int(cs_struct["cs_size"]),
            })
    if not rows:
        return pl.DataFrame(schema={
            "method": pl.String, "threshold": pl.Float64,
            "l": pl.Int64, "cs_size": pl.Int64,
        })
    return pl.from_dicts(rows, schema={
        "method": pl.String, "threshold": pl.Float64,
        "l": pl.Int64, "cs_size": pl.Int64,
    })
```

- [ ] **Step 6: Update `summarize_ser_log_bf_histogram_observations`**

Replace:

```python
def summarize_ser_log_bf_histogram_observations(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Build raw method x threshold x ser_log_bf observations."""
    fits_with_sample_id = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    return fits_with_sample_id.select(
        "method",
        "threshold",
        pl.col("ser_posterior").struct.field("ser_log_bf").alias("ser_log_bf"),
    )
```

With:

```python
def summarize_ser_log_bf_histogram_observations(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Build raw method x threshold x l x ser_log_bf observations."""
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        for l, effect in enumerate(row["single_effects"]):
            rows.append({
                "method": row["method"],
                "threshold": row["threshold"],
                "l": l,
                "ser_log_bf": float(effect["ser_log_bf"]),
            })
    if not rows:
        return pl.DataFrame(schema={
            "method": pl.String, "threshold": pl.Float64,
            "l": pl.Int64, "ser_log_bf": pl.Float64,
        })
    return pl.from_dicts(rows, schema={
        "method": pl.String, "threshold": pl.Float64,
        "l": pl.Int64, "ser_log_bf": pl.Float64,
    })
```

- [ ] **Step 6b: Verify `finalize_cs_size_histogram` and `finalize_ser_log_bf_histogram` need no changes**

Both functions select specific columns and drop `l` — intentionally pooling all effects into one distribution for histogram display. No code change needed; verify:

```bash
grep -n "finalize_cs_size_histogram\|finalize_ser_log_bf_histogram" plot_ready.py
```

They select `("method", "threshold", "cs_size")` and `("method", "threshold", "ser_log_bf")` — `l` is dropped, all L effects pool together. This is correct behavior.

- [ ] **Step 7: Run alpha-based summary tests — should now pass**

```bash
uv run pytest tests/test_plot_ready.py::test_summarize_cs_raw_per_sample_columns tests/test_plot_ready.py::test_summarize_cs_beta_trace_has_l_column -v
```

Expected: both PASS

- [ ] **Step 8: Run full test suite**

```bash
uv run pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: 50 passing (baseline unchanged)

- [ ] **Step 9: Commit**

```bash
git add plot_ready.py tests/test_plot_ready.py
git commit -m "feat: alpha-based plot_ready summaries loop over all L effects, emit l column"
```

---

### Task 6: Remove mass_above_causal from viz_utils.py and dashboard.py

**Files:**
- Modify: `viz_utils.py`
- Modify: `notebooks/dashboard.py`

These were added in a prior session and depend on a `mass_above_causal` column in `cs_beta_trace` that no longer exists.

- [ ] **Step 1: Remove three functions from `viz_utils.py`**

Remove `make_mass_above_causal_summary`, `_plot_mass_above_causal_on_ax`, and `render_mass_above_causal_chart` entirely. These are the ~110 lines added between `render_causal_rank_chart` and `explode_cs_component_beta_rows`.

- [ ] **Step 2: Remove two cells from `notebooks/dashboard.py`**

Remove `mass_above_causal_heading_cell` and `mass_above_causal_cell` — the two `@app.cell` blocks added between `causal_rank_cell` and `cs_summary_heading_cell`.

- [ ] **Step 3: Verify dashboard notebook still loads**

```bash
uv run pytest tests/test_plot_ready.py::test_dashboard_notebook_module_loads -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add viz_utils.py notebooks/dashboard.py
git commit -m "revert: remove mass_above_causal from viz_utils and dashboard (pending causal_stats table)"
```

---

### Task 7: Remove legacy `build_plot_data_frames` from utils.py

**Files:**
- Modify: `utils.py`
- Modify: `tests/test_twogroup_experiments.py`

`build_plot_data_frames` reads `row["ser_posterior"]["alpha"]` (old schema) and is not imported by the snakemake pipeline or any production code. Its four tests can be removed.

- [ ] **Step 1: Remove `build_plot_data_frames` and `symlink_plot_data_outputs` imports/functions from `utils.py`**

Remove the functions `build_plot_data_frames` (lines ~195–360) and `symlink_plot_data_outputs` from `utils.py`.

Also remove `CS_BETA_GRID` import from `utils.py` if it was only used by `build_plot_data_frames`. Check with:

```bash
grep -n "CS_BETA_GRID\|build_plot_data_frames\|symlink_plot_data" /Users/ktayeb/research/gibss-experiments/twogroup_experiments/utils.py
```

- [ ] **Step 2: Remove the four associated tests from `tests/test_twogroup_experiments.py`**

Remove:
- `test_build_plot_data_frames_returns_expected_tables_and_shapes`
- `test_build_plot_data_frames_propagates_specs_to_all_outputs`
- `test_build_plot_data_frames_keeps_spec_columns_for_empty_outputs`
- `test_build_plot_data_frames_keeps_threshold_dtype_for_empty_oracle_outputs`
- `test_symlink_plot_data_outputs_links_all_plot_data_files`

Also remove the import of `build_plot_data_frames` and `symlink_plot_data_outputs` from the test file's import block.

- [ ] **Step 3: Run full test suite — verify no new failures**

```bash
uv run pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: 50 passing (or more, if pre-existing failures were in the removed tests — check the pre-existing failure list: `test_build_plot_data_frames_returns_expected_tables_and_shapes` was already failing, so removing it reduces the failure count)

- [ ] **Step 4: Commit**

```bash
git add utils.py tests/test_twogroup_experiments.py
git commit -m "remove: legacy build_plot_data_frames reads old ser_posterior schema"
```

---

### Task 8: Final validation

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v 2>&1 | tail -30
```

Expected: All previously-passing tests still pass. Pre-existing failures (unrelated to this change) may remain.

- [ ] **Step 2: Verify core.py imports clean**

```bash
uv run python -c "import core; print('core OK')"
uv run python -c "import plot_ready; print('plot_ready OK')"
uv run python -c "import viz_utils; print('viz_utils OK')"
```

Expected: all three print OK

- [ ] **Step 3: Smoke-test a real L=5 fit end-to-end**

```bash
uv run python -c "
from functools import partial
from gibss.distributions import Normal, PointMass
from core import SimulationSpec, identity_design_sampler, uniform_single_effect, fit_logistic_method, summarize_logistic_method, simulate
import numpy as np

spec = SimulationSpec(
    name='smoke',
    design_sampler=identity_design_sampler,
    effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
    intercept=-1.0,
    f0=PointMass(0.0),
    f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
    base_seed=99,
)
sim = simulate(spec, 0)
fit_obj = fit_logistic_method(sim, response_source='z', threshold=None, L=5)
result = summarize_logistic_method(fit_obj, sim, response_source='z', threshold=None, L=5)
print('single_effects count:', len(result['single_effects']))
print('credible_sets count:', len(result['credible_sets']))
print('marginal_pip sum:', sum(result['fit_summary']['marginal_pip']))
print('causal_pip:', result['fit_summary']['causal_pip'])
print('OK')
"
```

Expected output:
```
single_effects count: 5
credible_sets count: 5
marginal_pip sum: <some float, typically < 5.0>
causal_pip: <float in [0, 1]>
OK
```

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git status
# commit anything uncommitted
```
