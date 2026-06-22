# 003 depletion supercollection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a depletion twin of `003-hallmark-loc-snr` using a new `ser_bneg2` enrichment (intercept −2, causal_effect −2), mirroring the enrichment SC's methods and outputs.

**Architecture:** Add one library enrichment `ser_bneg2`; refactor `003_loc_snr.yaml` to share signals/default_args/outputs via anchors; add a second supercollection `003-hallmark-loc-snr-depletion` whose only difference is `enrichment: [ser_bneg2, null_b0]`.

**Tech Stack:** Python, polars, pytest (run via `uv run pytest`), YAML.

## Global Constraints

- Run all Python via `uv run`, never bare `python`/`pip`.
- `ser_bneg2`: `uniform_single_effect`, `causal_effect: -2.0`, `intercept: -2.0`. Depletion null reuses existing `null_b0`.
- The enrichment SC `003-hallmark-loc-snr` resolved methods/outputs/analyses must stay unchanged (anchor refactor is behavior-preserving).
- No paired analyses in 003; depletion twin uses the same regular analyses as the enrichment SC.

---

### Task 1: `ser_bneg2` enrichment + depletion twin supercollection

**Files:**
- Modify: `experiments/library.yaml` (add `ser_bneg2` under `enrichments:`)
- Modify: `experiments/003_loc_snr.yaml` (anchor refactor + new depletion SC)
- Test: `tests/test_003_depletion.py` (new)

**Interfaces:**
- Consumes: `loader.load_library`, `loader.load_config`, `loader.resolve_simulation`, `loader.resolve_sc_analyses`, `loader.resolve_methods_for_sc`, `loader.manifest_dict`, `core.simulate`.
- Produces: library enrichment `ser_bneg2`; supercollection `003-hallmark-loc-snr-depletion`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_003_depletion.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import core
from experiments import loader


def test_ser_bneg2_is_depletion():
    lib = loader.load_library()
    ent = lib["enrichments"]["ser_bneg2"]
    assert ent["intercept"] == -2.0
    assert ent["arguments"]["causal_effect"] == -2.0
    spec = loader.resolve_simulation(lib, "hallmark", "ser_bneg2", "loc_2.0", "gaussian")
    assert spec.intercept == -2.0
    sim = core.simulate(spec, 0)
    b = np.asarray(sim.b)
    assert b.min() == -2.0  # one causal column with a negative (depleting) effect


def test_003_has_depletion_twin():
    cfg = loader.load_config()
    scs = cfg["supercollections"]
    assert "003-hallmark-loc-snr-depletion" in scs
    # depletion collections use ser_bneg2 + null_b0
    dep = scs["003-hallmark-loc-snr-depletion"]
    enrich_list = dep["collections"]["template"]["enrichment"]
    assert enrich_list == ["ser_bneg2", "null_b0"]
    # twin mirrors the enrichment SC's analyses and methods
    base = "003-hallmark-loc-snr"
    twin = "003-hallmark-loc-snr-depletion"
    assert set(loader.resolve_sc_analyses(cfg, twin)) == set(loader.resolve_sc_analyses(cfg, base))
    assert (set(loader.resolve_methods_for_sc(cfg["library"], scs[twin]))
            == set(loader.resolve_methods_for_sc(cfg["library"], scs[base])))


def test_003_enrichment_sc_unchanged():
    cfg = loader.load_config()
    base = cfg["supercollections"]["003-hallmark-loc-snr"]
    # enrichment SC still uses ser_b2 + null_b0
    assert base["collections"]["template"]["enrichment"] == ["ser_b2", "null_b0"]
    # manifest still builds with both SCs present
    m = loader.manifest_dict(cfg["library"], cfg)
    assert m["batches"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_003_depletion.py -v`
Expected: FAIL — `KeyError: 'ser_bneg2'` (not in enrichments) / `003-hallmark-loc-snr-depletion` absent.

- [ ] **Step 3: Add the `ser_bneg2` enrichment**

In `experiments/library.yaml`, under `enrichments:`, add after the `ser_b2` block:

```yaml
  ser_bneg2:
    function: uniform_single_effect
    arguments:
      causal_effect: -2.0
    intercept: -2.0
```

- [ ] **Step 4: Refactor 003 anchors and add the depletion twin**

Replace the entire contents of `experiments/003_loc_snr.yaml` with:

```yaml
_anchors:
  # minimal-loc output: one method per family at threshold 2.0
  base_methods: &base_methods
    [twogroup_oracle__L=1, twogroup__L=1, twogroup_loc_fam__L=1,
     cox_reversed__L=1, cox__threshold=2.00__L=1,
     logistic_threshold__threshold=2.00__L=1]
  # threshold-sweep output: swept lines (thresholded) + horizontals (threshold-free)
  sweep_methods: &sweep_methods
    [cox__threshold=1.00__L=1, cox__threshold=2.00__L=1,
     cox__threshold=3.00__L=1, cox__threshold=4.00__L=1,
     logistic_threshold__threshold=1.00__L=1, logistic_threshold__threshold=2.00__L=1,
     logistic_threshold__threshold=3.00__L=1, logistic_threshold__threshold=4.00__L=1,
     cox_reversed_censored__threshold=1.00__L=1, cox_reversed_censored__threshold=2.00__L=1,
     cox_reversed_censored__threshold=3.00__L=1, cox_reversed_censored__threshold=4.00__L=1,
     cox_uncensored__L=1, cox_reversed__L=1,
     twogroup_oracle__L=1, twogroup__L=1, twogroup_loc_fam__L=1]
  loc_signals: &loc_signals [loc_0.5, loc_1.0, loc_1.5, loc_2.0, loc_2.5, loc_3.0]
  default_args: &default_args {min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}
  outputs: &outputs
    - name: minimal-loc
      method_filter: *base_methods
      analyses: [pip, cs, logbf, pip_non_null, cs_non_null]
    - name: minimal-loc-threshold-sweep
      method_filter: *sweep_methods
      analyses: [threshold_sweep]

supercollections:
  003-hallmark-loc-snr:
    collections:
      template: {design: hallmark, enrichment: [ser_b2, null_b0], error: gaussian}
      over: {signal: *loc_signals}
    methods: *sweep_methods
    default_args: *default_args
    outputs: *outputs

  003-hallmark-loc-snr-depletion:
    collections:
      template: {design: hallmark, enrichment: [ser_bneg2, null_b0], error: gaussian}
      over: {signal: *loc_signals}
    methods: *sweep_methods
    default_args: *default_args
    outputs: *outputs
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_003_depletion.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run loader suite + full config build (no regressions)**

Run: `uv run pytest tests/test_loader.py tests/test_003_depletion.py -q`
Expected: PASS (all).

Then:
```bash
uv run python -c "
from experiments import loader
cfg = loader.load_config()
assert {'003-hallmark-loc-snr','003-hallmark-loc-snr-depletion'} <= set(cfg['supercollections'])
m = loader.manifest_dict(cfg['library'], cfg)
print('ok; SCs + manifest build; batches:', len(m['batches']))
"
```
Expected: prints the ok line, no error.

- [ ] **Step 7: Commit**

```bash
git add experiments/library.yaml experiments/003_loc_snr.yaml tests/test_003_depletion.py
git commit -m "feat(003): depletion twin supercollection (ser_bneg2 mirror)"
```

---

## Notes for the implementer

- `uniform_single_effect` (`simulations/effect/effects.py`) returns the single causal index for any nonzero `causal_effect`, so `-2.0` works unchanged; `core.simulate` sets `b[causal] = -2.0`, giving in-set logit `-4` and background `-2`.
- The shared YAML `outputs` anchor is referenced by both supercollections; the loader only reads it, so sharing one list object is safe.
- The `loc_2.0` signal used in the enrichment test is an existing signal in the library.
