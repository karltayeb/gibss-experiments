# Threshold-sweep plot methods (003) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `cox_reversed_censored` (swept line) and uncensored/twogroup horizontal references to the 003 `minimal-loc-threshold-sweep` plots, with shared family colors.

**Architecture:** No plot-rendering code changes — `_plot_causal_pip_on_ax`/`_plot_mass_above_causal_on_ax` already draw thresholded methods as lines and threshold-null methods as horizontals. We add a threshold-null `cox_uncensored` library method, two color/label map entries, and wire the method sets into 003.

**Tech Stack:** Python, polars, pytest (run via `uv run pytest`), YAML.

## Global Constraints

- Run all Python via `uv run`, never bare `python`/`pip`.
- Shared family colors: `cox_reversed_censored` → `#E69F00` (same as `cox_reversed`); `cox_uncensored` → `#009E73` (same as `cox`).
- Labels: `cox_reversed_censored` → `Cox reversed (censored)`; `cox_uncensored` → `Cox (uncensored)`.
- Do NOT change existing `cox` / `cox_reversed` map entries or any other experiment.
- No changes to `causal_pip` / `mass_above_causal` rendering code.

---

### Task 1: `cox_uncensored` method + color/label map entries

**Files:**
- Modify: `experiments/library.yaml` (add `cox_uncensored` under `methods:`, after `cox_reversed:`)
- Modify: `viz_utils.py` (`method_family_color_map` ~lines 34-49; `method_family_label_map` ~lines 13-27)
- Test: `tests/test_threshold_sweep_methods.py` (new)

**Interfaces:**
- Consumes: `viz_utils.method_color`, `viz_utils.method_family_label_map`, `experiments.loader.load_library`, `loader.library_methods`.
- Produces: library method `cox_uncensored` → coordinate `cox_uncensored__L=1`; color/label map entries for families `cox_reversed_censored` and `cox_uncensored`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_threshold_sweep_methods.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import viz_utils
from experiments import loader


def test_new_method_families_color_and_label():
    assert viz_utils.method_color("cox_reversed_censored__threshold=2.00__L=1") == "#E69F00"
    assert viz_utils.method_color("cox_uncensored__L=1") == "#009E73"
    labels = viz_utils.method_family_label_map()
    assert labels["cox_reversed_censored"] == "Cox reversed (censored)"
    assert labels["cox_uncensored"] == "Cox (uncensored)"


def test_cox_uncensored_registered():
    lib = loader.load_library()
    methods = loader.library_methods(lib)
    assert "cox_uncensored__L=1" in methods
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_threshold_sweep_methods.py -v`
Expected: FAIL — `method_color` returns `#888888` for the new families and `cox_uncensored__L=1` is not in `methods`.

- [ ] **Step 3: Add the `cox_uncensored` library method**

In `experiments/library.yaml`, immediately after the `cox_reversed:` method block (which ends with `over: L: [1]`) and before `cox_reversed_censored:`, add:

```yaml
  cox_uncensored:
    function: run_cox_method
    template:
      time_sign: -1.0
      threshold: null
    over:
      L:
      - 1
```

- [ ] **Step 4: Add the color map entries**

In `viz_utils.py` `method_family_color_map`, add these two entries inside the returned dict (e.g. after the `cox_reversed` line):

```python
        "cox_reversed_censored": "#E69F00",  # share cox_reversed color
        "cox_uncensored":        "#009E73",  # share cox color
```

- [ ] **Step 5: Add the label map entries**

In `viz_utils.py` `method_family_label_map`, add these two entries inside the returned dict (e.g. after the `cox_reversed` line):

```python
        "cox_reversed_censored": "Cox reversed (censored)",
        "cox_uncensored":        "Cox (uncensored)",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_threshold_sweep_methods.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Run loader suite (no regressions)**

Run: `uv run pytest tests/test_loader.py -v`
Expected: PASS (all).

- [ ] **Step 8: Commit**

```bash
git add experiments/library.yaml viz_utils.py tests/test_threshold_sweep_methods.py
git commit -m "feat(viz): cox_uncensored method + shared colors/labels for censored cox families"
```

---

### Task 2: Wire sweep methods into 003

**Files:**
- Modify: `experiments/003_loc_snr.yaml` (anchors + supercollection methods + sweep output method_filter)
- Test: `tests/test_threshold_sweep_methods.py` (append)

**Interfaces:**
- Consumes: `cox_uncensored__L=1` and the color/label entries from Task 1; existing `cox_reversed_censored` library method (threshold grid 0.5–3.5); `loader.load_config`, `loader.resolve_methods_for_sc`, `loader.method_metadata`.
- Produces: 003 `minimal-loc-threshold-sweep` output renders 3 thresholded line methods + 5 threshold-free horizontal methods.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_threshold_sweep_methods.py`:

```python
def test_003_sweep_method_threshold_split():
    import polars as pl
    cfg = loader.load_config()
    sc = cfg["supercollections"]["003-hallmark-loc-snr"]
    methods = loader.resolve_methods_for_sc(cfg["library"], sc)
    meta = loader.method_metadata(methods)

    line_methods = [
        "cox__threshold=2.00__L=1",
        "cox_reversed_censored__threshold=2.00__L=1",
        "logistic_threshold__threshold=2.00__L=1",
    ]
    horizontal_methods = [
        "cox_uncensored__L=1", "cox_reversed__L=1",
        "twogroup_oracle__L=1", "twogroup__L=1", "twogroup_loc_fam__L=1",
    ]
    for m in line_methods:
        row = meta.filter(pl.col("method") == m)
        assert row.height == 1 and bool(row["is_thresholded"][0]) is True, m
    for m in horizontal_methods:
        row = meta.filter(pl.col("method") == m)
        assert row.height == 1 and bool(row["is_thresholded"][0]) is False, m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_threshold_sweep_methods.py::test_003_sweep_method_threshold_split -v`
Expected: FAIL — `cox_reversed_censored__threshold=2.00__L=1` and `cox_uncensored__L=1` are not yet in the 003 method set (`row.height == 0`).

- [ ] **Step 3: Rewrite the 003 anchors and method sets**

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
    [cox__threshold=0.50__L=1, cox__threshold=1.00__L=1, cox__threshold=1.50__L=1,
     cox__threshold=2.00__L=1, cox__threshold=2.50__L=1, cox__threshold=3.00__L=1,
     cox__threshold=3.50__L=1,
     logistic_threshold__threshold=0.50__L=1, logistic_threshold__threshold=1.00__L=1,
     logistic_threshold__threshold=1.50__L=1, logistic_threshold__threshold=2.00__L=1,
     logistic_threshold__threshold=2.50__L=1, logistic_threshold__threshold=3.00__L=1,
     logistic_threshold__threshold=3.50__L=1,
     cox_reversed_censored__threshold=0.50__L=1, cox_reversed_censored__threshold=1.00__L=1,
     cox_reversed_censored__threshold=1.50__L=1, cox_reversed_censored__threshold=2.00__L=1,
     cox_reversed_censored__threshold=2.50__L=1, cox_reversed_censored__threshold=3.00__L=1,
     cox_reversed_censored__threshold=3.50__L=1,
     cox_uncensored__L=1, cox_reversed__L=1,
     twogroup_oracle__L=1, twogroup__L=1, twogroup_loc_fam__L=1]

supercollections:
  003-hallmark-loc-snr:
    collections:
      template: {design: hallmark, enrichment: [ser_b2, null_b0], error: gaussian}
      over: {signal: [loc_0.5, loc_1.0, loc_1.5, loc_2.0, loc_2.5, loc_3.0]}
    methods: *sweep_methods
    default_args: {min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}
    outputs:
      - name: minimal-loc
        method_filter: *base_methods
        analyses: [pip, cs, logbf, pip_non_null, cs_non_null]
      - name: minimal-loc-threshold-sweep
        method_filter: *sweep_methods
        analyses: [threshold_sweep]
```

(`sweep_methods` is the superset; it contains every `base_methods` entry, so the supercollection `methods` list covers the `minimal-loc` filter too.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_threshold_sweep_methods.py::test_003_sweep_method_threshold_split -v`
Expected: PASS.

- [ ] **Step 5: Verify the full config + manifest still build**

Run:
```bash
uv run python -c "
from experiments import loader
cfg = loader.load_config()
m = loader.manifest_dict(cfg['library'], cfg)
print('manifest batches:', len(m['batches']), 'methods:', len(m['methods']))
print('cox_reversed_censored__threshold=2.00__L=1' in m['methods'])
print('cox_uncensored__L=1' in m['methods'])
"
```
Expected: prints counts and `True` twice, no error.

- [ ] **Step 6: Run loader + new suite (no regressions)**

Run: `uv run pytest tests/test_loader.py tests/test_threshold_sweep_methods.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add experiments/003_loc_snr.yaml tests/test_threshold_sweep_methods.py
git commit -m "feat(003): add cox_reversed_censored sweep + uncensored/twogroup horizontals to threshold-sweep plots"
```

---

## Notes for the implementer

- `method_color(name)` derives family as `name.split("__")[0]`; so `cox_reversed_censored__threshold=2.00__L=1` → family `cox_reversed_censored`, and `cox_uncensored__L=1` → family `cox_uncensored`. Both must be in the color/label maps.
- `loader.method_metadata` sets `is_thresholded = threshold is not None`; `cox_uncensored` (template `threshold: null`) and `cox_reversed`/twogroup (no threshold) resolve to `is_thresholded=False` → drawn as horizontals.
- Defining the methods in the library/anchors does not fit anything until the manifest is built and the pipeline run for 003.
