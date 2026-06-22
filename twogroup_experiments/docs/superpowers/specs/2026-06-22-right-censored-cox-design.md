# Right-censored cox (both directions)

Date: 2026-06-22

## Problem

The cox-family fit (`fits/cox.py`, `run_cox_method`) treats a finite `threshold`
as an event/censor split:

```python
score = |thetahat/se|                       # |z|
event_type = (score > threshold)            # high |z| are events
event_time = time_sign * score              # actual |z|, no clamping
```

This makes the *extreme* |z| the events regardless of direction, which is not
proper right-censoring. We want a censored cox where a threshold `t` means:
observations whose arrival happens *after* `t` (in the chosen time direction)
are right-censored at `t` and remain in the risk set, contributing as "known to
survive past `t`."

Two directions are wanted, one shared implementation:

- `time_sign = -1` (the existing `cox` direction): events = `|z| > t`, the small
  `|z| <= t` are censored at `t` and kept in risk sets.
- `time_sign = +1` (the `cox_reversed` direction): events = `|z| <= t`, the
  extreme `|z| > t` are censored at `t` and kept in risk sets. **New behavior.**

## Background findings

These shaped the design (established during brainstorming):

1. **Clamping is a numerical no-op for a hard threshold.** The Cox partial
   likelihood depends only on event-time rank order and risk-set membership;
   exact times matter only through ties. A single threshold cleanly separates
   events from censored along the time axis, so clamping censored arrivals to
   `t` (vs keeping actual `|z|`) only reorders ties *within* the censored block,
   where there are no events. Risk-set membership for every event is unchanged.
   We clamp anyway for correct survival semantics.

2. **The `time_sign = -1` censored case is numerically identical to today's
   `cox`.** Today's `cox__threshold=t` already has events = `1(|z| > t)` with
   `time_sign = -1`. Combined with finding 1, the new right-censoring produces
   the same fit. So existing `cox` results stay valid; no re-fit, no migration.

3. **The `time_sign = +1` censored case is genuinely new.** Current
   `run_cox_method` with `time_sign = +1, threshold = t` would make the extremes
   events and drop the bulk — the opposite of the desired reversed-censored
   behavior. No existing config uses `cox_reversed` with a threshold.

## Design

### Unified right-censoring

Right-censor the transformed arrival time `time_sign * |z|` at censoring time
`time_sign * t`:

```python
def _right_censored_survival(score, threshold, time_sign):
    raw = time_sign * score
    T = time_sign * threshold
    event_time = np.minimum(raw, T)          # arrivals past t clamped to t
    event_type = (raw <= T).astype(int)      # arrived by t -> event; else censored
    return event_time, event_type
```

- `+1`: `raw = |z|`, `T = t`. Event iff `|z| <= t`; extremes clamped to `t`,
  sit above all events in time -> retained in every event's risk set.
- `-1`: `raw = -|z|`, `T = -t`. Event iff `|z| >= t` (≈ `|z| > t`, boundary is
  measure-zero for continuous |z|); small `|z|` clamped to `-t`, sit above the
  extreme events -> retained.

### Components

1. **`fits/cox.py`** — add `_right_censored_survival(score, threshold, time_sign)`
   as above. In `fit_cox_method`, replace the threshold branch:
   - `threshold is None` -> non-censored: `event_time = time_sign * score`,
     `event_type = ones` (unchanged).
   - `threshold` finite -> `event_time, event_type = _right_censored_survival(
     score, float(threshold), time_sign)`.
   - `n_selected = int(event_type.sum())` (unchanged meaning: number of events).

   Both `cox` and `cox_reversed` library entries already call `run_cox_method`,
   so this single change covers both directions.

2. **`experiments/library.yaml`** — add a censored reversed method entry so the
   reversed-censored coordinates exist:
   ```yaml
   cox_reversed_censored:
     function: run_cox_method
     template:
       time_sign: 1.0
     over:
       threshold: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]   # mirror cox grid
       L: [1]
   ```
   Existing `cox` (time_sign -1, threshold grid) and `cox_reversed`
   (time_sign +1, threshold null, non-censored) are unchanged.

### Blast radius

- Experiments 000–007 reference `cox__threshold=2.00` only. Their behavior and
  cached fits are numerically unchanged (findings 1–2). No edits, no re-fit.
- New capability is opt-in per experiment via `cox_reversed_censored__threshold=
  X__L=1`.

## Naming note (deliberate scope limit)

The conceptual end state is `cox`/`cox_reversed` = non-censored and dedicated
censored names. We are **not** renaming `cox` -> `cox_censored` here: the
threshold suffix (`cox__threshold=X`) already distinguishes the censored variant,
and a rename would orphan existing fits and touch seven experiment files for no
behavioral gain. The reversed-censored variant gets its own name
(`cox_reversed_censored`) because `cox_reversed` is reserved for the
non-censored (threshold-null) case. A full rename can be a separate cleanup.

## Testing

- **Helper unit test:** for a small `score` vector, assert `_right_censored_survival`
  returns the correct `event_time` (clamped at `t`) and `event_type` for both
  `time_sign = +1` and `-1`.
- **No-regression / equivalence:** on a random simulation,
  `cox__threshold=t` (new code) yields `alpha`/PIP `allclose` to the pre-change
  behavior, i.e. to a Cox fit with `event_type = (score > t)` and
  `event_time = -score`. Guards finding 2.
- **Distinctness:** `cox_reversed_censored__threshold=t` produces a different fit
  from the (old-style) `time_sign=+1, event_type=(score>t)` assignment, and from
  `cox__threshold=t`, confirming the new reversed-censored behavior is active.

## Out of scope

- Wiring `cox_reversed_censored` into specific experiments (003/009) — later
  config step.
- Viz display name / color for `cox_reversed_censored` — add when an experiment
  first renders it.
- Full `cox` -> `cox_censored` rename.
