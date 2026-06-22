# Right-censored cox (both directions) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a finite `threshold` in the cox fit mean proper right-censoring (arrivals beyond `t` clamped to `t` and retained in risk sets), for both `time_sign` directions, and expose a reversed-censored library method.

**Architecture:** Add a `_right_censored_survival(score, threshold, time_sign)` helper in `fits/cox.py` and call it from `fit_cox_method`'s finite-threshold branch (the `threshold is None` branch is unchanged). Both `cox` and `cox_reversed` already route through `run_cox_method`, so one change covers both directions. Add a `cox_reversed_censored` library method for the new `time_sign=+1` censored coordinates.

**Tech Stack:** Python, NumPy, JAX/gibss (`gibss.cox`, `gibss.engine`), pytest (run via `uv run pytest`), YAML.

## Global Constraints

- Run all Python via `uv run` (e.g. `uv run pytest`), never bare `python`/`pip`.
- Right-censoring formula: `raw = time_sign*score`, `T = time_sign*threshold`, `event_time = min(raw, T)`, `event_type = (raw <= T)`.
- `threshold is None` keeps the current non-censored behavior byte-identical: `event_time = time_sign*score`, `event_type = ones`.
- `time_sign=-1` censored must stay numerically identical to the pre-change `cox` (event partition `score > t`, no clamp) — clamping is a partial-likelihood no-op for a hard threshold.
- Do NOT rename `cox` or migrate experiments 000–007.

---

### Task 1: Right-censoring helper + fit integration

**Files:**
- Modify: `fits/cox.py` (add `_right_censored_survival`; edit `fit_cox_method` threshold branch ~lines 13-22)
- Test: `tests/test_cox_censored.py` (new)

**Interfaces:**
- Consumes: `core._score`, `gibss.cox`, `gibss.engine`, existing `core.run_cox_method`.
- Produces: `fits.cox._right_censored_survival(score: np.ndarray, threshold: float, time_sign: float) -> tuple[np.ndarray, np.ndarray]` returning `(event_time, event_type)`; `fit_cox_method` uses it when `threshold is not None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cox_censored.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import core
from fits.cox import _right_censored_survival


def _tiny_simulation():
    from core import SimulationSpec, simulate
    from functools import partial
    from gibss.distributions import Normal, PointMass
    spec = SimulationSpec(
        design_sampler=partial(core.gaussian_markov_X, n=40, p=10, rho=0.5),
        effect_sampler=partial(core.uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=2.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        error_sampler=None,
        base_seed=1,
        hash="coxcensored",
        name="tiny",
    )
    return simulate(spec, 0)


def test_right_censored_survival_plus_one():
    score = np.array([0.2, 1.0, 2.5, 3.0])
    t = 1.5
    et, ev = _right_censored_survival(score, t, 1.0)
    # +1: event iff |z| <= t; censored arrivals clamped to t
    np.testing.assert_array_equal(ev, np.array([1, 1, 0, 0]))
    np.testing.assert_allclose(et, np.array([0.2, 1.0, 1.5, 1.5]))


def test_right_censored_survival_minus_one():
    score = np.array([0.2, 1.0, 2.5, 3.0])
    t = 1.5
    et, ev = _right_censored_survival(score, t, -1.0)
    # -1: raw=-score, T=-t; event iff -score <= -t  <=>  score >= t
    np.testing.assert_array_equal(ev, np.array([0, 0, 1, 1]))
    # event_time = min(-score, -t) = -max(score, t)
    np.testing.assert_allclose(et, np.array([-1.5, -1.5, -2.5, -3.0]))


def test_cox_minus_one_threshold_matches_pre_change_behavior():
    # No-regression guard: new censored cox (time_sign=-1) == a direct gibss
    # fit using the OLD construction (event_type = score>t, event_time = -score).
    from gibss import cox, engine
    sim = _tiny_simulation()
    score = np.abs(np.asarray(sim.thetahat) / np.asarray(sim.se))
    t = float(np.median(score))  # ensure both events and censored exist

    data_old = cox.prep_data(
        sim.X, event_time=-1.0 * score, event_type=(score > t).astype(int)
    )
    st_old = cox.initialize_state(
        data_old, L=1, family_state_kwargs={"estimate_prior_variance": False}
    )
    fit_old = engine.fit_ibss(data_old, st_old, cox.default_schedule())
    alpha_old = np.asarray(fit_old.single_effects[0].alpha)

    new = core.run_cox_method(sim, threshold=t, time_sign=-1.0, L=1)
    alpha_new = np.asarray(new["single_effects"][0]["alpha"])
    np.testing.assert_allclose(alpha_new, alpha_old, atol=1e-8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cox_censored.py -v`
Expected: FAIL — `ImportError: cannot import name '_right_censored_survival'`.

- [ ] **Step 3: Add the helper**

In `fits/cox.py`, after the imports (below `from gibss import cox, engine`), add:

```python
def _right_censored_survival(score, threshold, time_sign):
    """Right-censor the transformed arrival time ``time_sign*score`` at
    ``time_sign*threshold``: arrivals past the threshold are clamped to it and
    marked censored (``event_type=0``); they stay in the risk set.
    """
    score = np.asarray(score, dtype=float)
    raw = float(time_sign) * score
    T = float(time_sign) * float(threshold)
    event_time = np.minimum(raw, T)
    event_type = (raw <= T).astype(int)
    return event_time, event_type
```

- [ ] **Step 4: Wire it into `fit_cox_method`**

In `fits/cox.py` `fit_cox_method`, replace this block:

```python
    score = _score(simulation)
    if threshold is None:
        event_type = np.ones_like(score, dtype=int)
    else:
        event_type = (score > float(threshold)).astype(int)
    data = cox.prep_data(
        simulation.X,
        event_time=time_sign * score,
        event_type=event_type,
    )
```

with:

```python
    score = _score(simulation)
    if threshold is None:
        event_time = time_sign * score
        event_type = np.ones_like(score, dtype=int)
    else:
        event_time, event_type = _right_censored_survival(score, threshold, time_sign)
    data = cox.prep_data(
        simulation.X,
        event_time=event_time,
        event_type=event_type,
    )
```

(The later `"n_selected": int(event_type.sum())` line is unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cox_censored.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the cox-related suite (no regressions)**

Run: `uv run pytest tests/test_core_run_methods.py tests/test_cox_censored.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add fits/cox.py tests/test_cox_censored.py
git commit -m "feat(cox): right-censoring for finite threshold (both directions)"
```

---

### Task 2: `cox_reversed_censored` library method

**Files:**
- Modify: `experiments/library.yaml` (add `cox_reversed_censored` under `methods:`, after the `cox_reversed` entry ~line 609)
- Test: `tests/test_cox_censored.py` (append)

**Interfaces:**
- Consumes: `fit_cox_method` right-censoring from Task 1; `experiments.loader` (`load_library`, `library_methods`, `run_method`).
- Produces: library method base `cox_reversed_censored` (function `run_cox_method`, `time_sign: 1.0`, `over.threshold` grid, `over.L: [1]`) → coordinates `cox_reversed_censored__threshold=X__L=1`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cox_censored.py`:

```python
def test_cox_reversed_censored_method_registered_and_new():
    from experiments import loader
    lib = loader.load_library()
    methods = loader.library_methods(lib)
    assert "cox_reversed_censored__threshold=2.00__L=1" in methods

    # reversed-censored (+1) makes the bulk (|z| <= t) the events -- the
    # complement of the old-style (score > t) assignment.
    sim = _tiny_simulation()
    score = np.abs(np.asarray(sim.thetahat) / np.asarray(sim.se))
    t = float(np.median(score))
    _, ev_rev = _right_censored_survival(score, t, 1.0)
    np.testing.assert_array_equal(ev_rev, (score <= t).astype(int))

    coord = {"name": "cox_reversed_censored__threshold=2.00__L=1",
             "function": "run_cox_method",
             "kwargs": {"threshold": 2.0, "time_sign": 1.0, "L": 1}}
    row = loader.run_method(coord, sim)
    assert row["method"] == "cox_reversed_censored__threshold=2.00__L=1"
    assert "single_effects" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cox_censored.py::test_cox_reversed_censored_method_registered_and_new -v`
Expected: FAIL — assertion on missing `cox_reversed_censored__threshold=2.00__L=1` in `methods`.

- [ ] **Step 3: Add the library method**

In `experiments/library.yaml`, immediately after the `cox_reversed` method block (which ends with its `over: L: [1]`, ~line 609) and before `cox:`, add:

```yaml
  cox_reversed_censored:
    function: run_cox_method
    template:
      time_sign: 1.0
    over:
      threshold:
      - 0.5
      - 1.0
      - 1.5
      - 2.0
      - 2.5
      - 3.0
      - 3.5
      L:
      - 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cox_censored.py::test_cox_reversed_censored_method_registered_and_new -v`
Expected: PASS.

- [ ] **Step 5: Run loader + cox suites (no regressions)**

Run: `uv run pytest tests/test_loader.py tests/test_cox_censored.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add experiments/library.yaml tests/test_cox_censored.py
git commit -m "feat(cox): add cox_reversed_censored method (reversed right-censored)"
```

---

## Notes for the implementer

- `gibss.cox.prep_data` preserves sparse BCOO `X` and densifies only non-sparse input; `_tiny_simulation` uses a dense gaussian design, which is fine.
- `_score(sim)` returns `|thetahat/se|`; the helper takes that array directly.
- `single_effects[l]` summary dicts include an `"alpha"` key (see `core._extract_ser_struct`), used by the no-regression test.
- Defining the `cox_reversed_censored` threshold grid does not trigger any fits; coordinates are only built when a supercollection references them.
