# Reciprocal-rate pairing (paired analyses) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give 009 a uniform depletion/enrichment treatment — facet by `max(rate,1/rate)`, color by `rate<1` — by adding a "paired" analysis family that wraps existing renderers via a bundle transform.

**Architecture:** A new `analyses/paired.py` exposes `<name>_paired` renderers that transform the loaded bundle (rewrite `collection_name`→pair label, `method`→`depletion`/`enrichment`, rebuild `method_metadata`) then call the existing `_make_*` function unchanged. The loader is made suffix-aware so paired analyses need no duplicated library `analyses:` entries.

**Tech Stack:** Python, polars, matplotlib, pytest (run via `uv run pytest`), YAML, snakemake.

## Global Constraints

- Run all Python via `uv run`, never bare `python`/`pip`.
- Do NOT modify any existing `_make_*` renderer or the generic facet/color logic.
- Transform: `method` → `"depletion"` if `rate<1` else `"enrichment"`; `collection_name` → formatted `max(rate, 1/rate)`; `method_filter` overridden to `["depletion","enrichment"]`.
- Rate parsed from alias `lambda=a/b`; non-parseable alias must raise, not mislabel.
- Only 009 uses paired analyses; other experiments untouched.

---

### Task 1: Loader suffix-awareness for `_paired` analyses

**Files:**
- Modify: `experiments/loader.py` (`flatten_analyses` ~251; `analysis_simulation_filter` ~412; `analysis_requires` ~478; `analysis_family` ~592; add `_base_analysis` helper)
- Test: `tests/test_paired_analyses.py` (new)

**Interfaces:**
- Produces: `loader._base_analysis(name) -> str` (strips a single `_paired` suffix); `analysis_family`, `analysis_requires`, `analysis_simulation_filter`, `flatten_analyses` all accept `<base>_paired` names by resolving to `<base>`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_paired_analyses.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import loader


def test_loader_suffix_aware_for_paired():
    lib = loader.load_library()
    cfg = {"library": lib}
    # family resolves to "paired" by suffix
    assert loader.analysis_family("causal_pip_paired") == "paired"
    # requires + simulation_filter resolve to the base analysis
    assert loader.analysis_requires(cfg, "causal_pip_paired") == loader.analysis_requires(cfg, "causal_pip")
    assert (loader.analysis_simulation_filter(lib, "causal_pip_paired")
            == loader.analysis_simulation_filter(lib, "causal_pip"))
    # flatten accepts a paired name (validates via base)
    assert loader.flatten_analyses(lib, ["causal_pip_paired"]) == ["causal_pip_paired"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paired_analyses.py::test_loader_suffix_aware_for_paired -v`
Expected: FAIL — `analysis_family("causal_pip_paired")` raises `KeyError` (not in any family RENDERERS).

- [ ] **Step 3: Add `_base_analysis` helper**

In `experiments/loader.py`, add near the other analysis helpers (e.g. just above `flatten_analyses` at line 251):

```python
def _base_analysis(analysis: str) -> str:
    """Strip a single trailing '_paired' suffix (paired analyses reuse base metadata)."""
    return analysis[: -len("_paired")] if analysis.endswith("_paired") else analysis
```

- [ ] **Step 4: Make `flatten_analyses` validate via base**

In `flatten_analyses`, replace the validation line:

```python
            if n not in analyses:
                raise KeyError(f"Unknown analysis: {n!r}")
```

with:

```python
            if _base_analysis(n) not in analyses:
                raise KeyError(f"Unknown analysis: {n!r}")
```

- [ ] **Step 5: Make requires + simulation_filter resolve the base**

Replace `analysis_simulation_filter`:

```python
def analysis_simulation_filter(library: dict[str, Any], analysis: str) -> str | None:
    return library["analyses"][_base_analysis(analysis)].get("simulation_filter")
```

Replace `analysis_requires`:

```python
def analysis_requires(config: dict[str, Any], analysis: str) -> list[str]:
    return list(config["library"]["analyses"][_base_analysis(analysis)].get("requires", []))
```

- [ ] **Step 6: Add the `paired` branch to `analysis_family`**

In `analysis_family`, add at the top of the function body (before the `pip` check):

```python
    if analysis.endswith("_paired"):
        return "paired"
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_paired_analyses.py::test_loader_suffix_aware_for_paired -v`
Expected: PASS.

- [ ] **Step 8: Run loader suite (no regressions)**

Run: `uv run pytest tests/test_loader.py -v`
Expected: PASS (all).

- [ ] **Step 9: Commit**

```bash
git add experiments/loader.py tests/test_paired_analyses.py
git commit -m "feat(loader): suffix-aware metadata/family lookup for _paired analyses"
```

---

### Task 2: `analyses/paired.py` transform + wrappers + maps

**Files:**
- Create: `analyses/paired.py`
- Modify: `analyses/__init__.py` (register `paired.RENDERERS`)
- Modify: `viz_utils.py` (`method_family_color_map`, `method_family_label_map` — add `depletion`, `enrichment`)
- Test: `tests/test_paired_analyses.py` (append)

**Interfaces:**
- Consumes: `loader._base_analysis`/`analysis_family` from Task 1; `analyses.pip/cs/logbf` RENDERERS; `generate_plots.ANALYSIS_RENDERERS`.
- Produces: `paired.pair_reciprocal(bundle) -> dict`; `paired.RENDERERS` with keys `<name>_paired`; `generate_plots.ANALYSIS_RENDERERS["causal_pip_paired"]` etc.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_paired_analyses.py`:

```python
import matplotlib
matplotlib.use("Agg")
import polars as pl


def test_pair_reciprocal_transform():
    from analyses.paired import pair_reciprocal
    pip_plot = pl.DataFrame(
        {
            "collection_name": ["lambda=2/3", "lambda=3/2"],
            "method": ["cox_reversed__L=1", "cox_reversed__L=1"],
            "threshold": [None, None],
            "causal_pips": [[0.3, 0.4], [0.6, 0.7]],
        },
        schema_overrides={"threshold": pl.Float64},
    )
    bundle = {"pip_plot_data": pip_plot, "method_metadata": pl.DataFrame(),
              "collection_names": ["lambda=2/3", "lambda=3/2"]}
    out = pair_reciprocal(bundle)
    df = out["pip_plot_data"].sort("method")
    # reciprocals collapse to one pair label
    assert df["collection_name"].n_unique() == 1
    # sign assignment
    assert df["method"].to_list() == ["depletion", "enrichment"]
    # method_metadata rebuilt with the two sign rows
    assert set(out["method_metadata"]["method"].to_list()) == {"depletion", "enrichment"}
    assert len(out["collection_names"]) == 1  # 2/3 and 3/2 collapse to one pair


def test_paired_renderer_registered_and_renders():
    import generate_plots
    assert "causal_pip_paired" in generate_plots.ANALYSIS_RENDERERS
    pip_plot = pl.DataFrame(
        {
            "collection_name": ["lambda=2/3", "lambda=3/2"],
            "method": ["cox_reversed__L=1", "cox_reversed__L=1"],
            "threshold": [None, None],
            "causal_pips": [[0.3, 0.4], [0.6, 0.7]],
        },
        schema_overrides={"threshold": pl.Float64},
    )
    bundle = {"pip_plot_data": pip_plot, "method_metadata": pl.DataFrame(),
              "collection_names": ["lambda=2/3", "lambda=3/2"]}
    fig = generate_plots.ANALYSIS_RENDERERS["causal_pip_paired"](bundle, {})
    labels = set(fig.axes[0].get_legend_handles_labels()[1])
    assert labels and labels <= {"Depletion", "Enrichment"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_paired_analyses.py -k "pair_reciprocal or paired_renderer" -v`
Expected: FAIL — `ModuleNotFoundError: analyses.paired`.

- [ ] **Step 3: Create `analyses/paired.py`**

```python
"""Paired analyses: reciprocal-rate pairing wrappers around existing renderers.

Each paired renderer rewrites the bundle so reciprocal-rate collections (e.g.
lambda=2/3 and lambda=3/2) share a facet (max(rate,1/rate)) and are colored by
depletion (rate<1) vs enrichment (rate>1), then calls the existing analysis
renderer unchanged. Intended for single-method experiments (009 / cox_reversed).
"""
from __future__ import annotations

from fractions import Fraction

import polars as pl

from analyses import pip, cs, logbf

_SIGN_METHODS = ("depletion", "enrichment")


def _rate(alias: str) -> Fraction:
    if "=" not in alias:
        raise ValueError(f"Cannot parse rate from collection alias: {alias!r}")
    return Fraction(alias.split("=", 1)[1])


def _pair_label(fr: Fraction) -> str:
    hi = max(fr, 1 / fr)
    return f"{hi.numerator}/{hi.denominator}"


def _sign(fr: Fraction) -> str:
    return "depletion" if fr < 1 else "enrichment"


def _sign_method_metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "method": list(_SIGN_METHODS),
            "method_family": list(_SIGN_METHODS),
            "L": [1, 1],
            "threshold": [None, None],
            "is_thresholded": [False, False],
            "is_oracle": [False, False],
            "method_label_base": ["Depletion", "Enrichment"],
            "method_display": ["Depletion", "Enrichment"],
            "method_display_base": ["Depletion", "Enrichment"],
        },
        schema_overrides={"threshold": pl.Float64},
    )


def pair_reciprocal(bundle: dict) -> dict:
    aliases = list(bundle.get("collection_names", []))
    fr = {a: _rate(a) for a in aliases}
    pair_of = {a: _pair_label(fr[a]) for a in aliases}
    sign_of = {a: _sign(fr[a]) for a in aliases}

    out = dict(bundle)
    for key, df in bundle.items():
        if key.endswith("_plot_data") and isinstance(df, pl.DataFrame) and not df.is_empty():
            out[key] = df.with_columns(
                pl.col("collection_name").replace(sign_of).alias("method"),
                pl.lit(None, dtype=pl.Float64).alias("threshold"),
                pl.col("collection_name").replace(pair_of).alias("collection_name"),
            )
    out["method_metadata"] = _sign_method_metadata()
    out["collection_names"] = sorted({pair_of[a] for a in aliases})
    return out


def _paired(make_fn):
    def renderer(bundle: dict, settings: dict):
        transformed = pair_reciprocal(bundle)
        merged_settings = {**settings, "method_filter": list(_SIGN_METHODS)}
        return make_fn(transformed, merged_settings)
    return renderer


_BASE_RENDERERS = {**pip.RENDERERS, **cs.RENDERERS, **logbf.RENDERERS}
RENDERERS = {f"{name}_paired": _paired(fn) for name, fn in _BASE_RENDERERS.items()}


if "snakemake" in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).parent.parent)
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    import generate_plots
    from experiments import loader as _loader
    _wc = snakemake.wildcards
    _analysis = snakemake.params.analysis
    _cfg_obj = _loader.load_config()
    _bundle = _loader.load_sc_bundle(
        _cfg_obj, _wc.supercollection,
        _loader.analysis_requires(_cfg_obj, _analysis),
        simulation_filter=_loader.analysis_simulation_filter(_cfg_obj["library"], _analysis),
    )
    _args = _loader.resolve_args(_cfg_obj, _wc.supercollection, _wc.args_name)
    generate_plots.render_analysis(_bundle, _args, _analysis, snakemake.output[0])
```

Note: `replace` referencing the original `collection_name` works because polars
evaluates all expressions in one `with_columns` against the input frame, so the
`method` and pair-label expressions both see the original alias.

- [ ] **Step 4: Register `paired.RENDERERS` in the package**

In `analyses/__init__.py`, import `paired` **after** the other families and merge its renderers:

```python
from analyses import pip, cs, logbf, f1
from analyses import paired
from analyses._common import foreground_methods, method_order, set_agg_facecolor

ANALYSIS_RENDERERS: dict = {
    **pip.RENDERERS,
    **cs.RENDERERS,
    **logbf.RENDERERS,
    **f1.RENDERERS,
    **paired.RENDERERS,
}
```

Add `"paired"` to `__all__`.

- [ ] **Step 5: Add depletion/enrichment color + label entries**

In `viz_utils.py` `method_family_color_map`, add inside the returned dict:

```python
        "depletion":             "#0072B2",
        "enrichment":            "#D55E00",
```

In `method_family_label_map`, add:

```python
        "depletion":             "Depletion",
        "enrichment":            "Enrichment",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_paired_analyses.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 7: Run loader + viz suites (no regressions)**

Run: `uv run pytest tests/test_loader.py tests/test_causal_pip_grouping.py -v`
Expected: PASS (all).

- [ ] **Step 8: Commit**

```bash
git add analyses/paired.py analyses/__init__.py viz_utils.py tests/test_paired_analyses.py
git commit -m "feat(analyses): paired reciprocal-rate analysis family (depletion vs enrichment)"
```

---

### Task 3: Wire paired analyses into 009

**Files:**
- Modify: `experiments/library.yaml` (add `paired_009` analysis group)
- Modify: `experiments/009_cox_well_specified.yaml` (`analyses` anchor → `[paired_009]`)
- Test: `tests/test_paired_analyses.py` (append)

**Interfaces:**
- Consumes: suffix-aware loader (Task 1), `paired.RENDERERS` (Task 2).
- Produces: 009 supercollections resolve to the 25 `<base>_paired` analyses.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_paired_analyses.py`:

```python
def test_009_uses_paired_analyses():
    cfg = loader.load_config()
    pairs = loader.resolve_sc_analyses(cfg, "009-hallmark-cox-well-specified")
    analyses = {a for a, _ in pairs}
    assert analyses, "009 resolved no analyses"
    assert all(a.endswith("_paired") for a in analyses)
    assert "causal_pip_paired" in analyses
    assert "agg_causal_pip_paired" in analyses
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paired_analyses.py::test_009_uses_paired_analyses -v`
Expected: FAIL — 009 still resolves the non-paired analyses.

- [ ] **Step 3: Add the `paired_009` analysis group**

In `experiments/library.yaml`, under `analysis_groups:`, add (after the existing groups):

```yaml
  paired_009:
  - pip_calibration_paired
  - agg_pip_calibration_paired
  - power_fdp_paired
  - agg_power_fdp_paired
  - cs_power_size_coverage_trace_paired
  - agg_cs_power_size_coverage_trace_paired
  - cs_power_fdp_paired
  - agg_cs_power_fdp_paired
  - cs_size_power_paired
  - agg_cs_size_power_paired
  - cs_radius_power_paired
  - agg_cs_radius_power_paired
  - log_bf_roc_paired
  - agg_log_bf_roc_paired
  - log_bf_ser_ecdf_paired
  - agg_log_bf_ser_ecdf_paired
  - causal_pip_paired
  - agg_causal_pip_paired
  - mass_above_causal_paired
  - agg_mass_above_causal_paired
  - cs_coverage_size_paired
  - agg_cs_coverage_size_paired
  - cs_coverage_radius_paired
  - agg_cs_coverage_radius_paired
  - cs_calibrated_dot_paired
```

- [ ] **Step 4: Point 009 at the paired group**

In `experiments/009_cox_well_specified.yaml`, change the analyses anchor line:

```yaml
  analyses: &analyses [pip, cs, logbf, pip_non_null, cs_non_null]
```

to:

```yaml
  analyses: &analyses [paired_009]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_paired_analyses.py::test_009_uses_paired_analyses -v`
Expected: PASS.

- [ ] **Step 6: Verify config + manifest build end-to-end**

Run:
```bash
uv run python -c "
from experiments import loader
cfg = loader.load_config()
for sc in [s for s in cfg['supercollections'] if s.startswith('009')]:
    pairs = loader.resolve_sc_analyses(cfg, sc)
    assert pairs and all(a.endswith('_paired') for a,_ in pairs), sc
m = loader.manifest_dict(cfg['library'], cfg)
print('ok; 009 SCs resolve paired analyses; manifest methods:', len(m['methods']))
"
```
Expected: prints the ok line, no assertion error.

- [ ] **Step 7: Run full suite (no regressions)**

Run: `uv run pytest tests/ -q`
Expected: PASS (all).

- [ ] **Step 8: Commit**

```bash
git add experiments/library.yaml experiments/009_cox_well_specified.yaml tests/test_paired_analyses.py
git commit -m "feat(009): use paired reciprocal-rate analyses for all plots"
```

---

## Notes for the implementer

- `loader.load_sc_bundle` sets each `<reduction>_plot_data` frame's `collection_name` to the collection **alias** (e.g. `lambda=2/3`); that is what `pair_reciprocal` parses.
- The transform overwrites `method_metadata` entirely, so the bundle's incoming `method_metadata` is irrelevant to paired rendering (tests pass an empty frame).
- `polars` `Series.replace(mapping)` leaves values absent from the mapping unchanged; all 009 aliases are present in the maps, so every row is rewritten.
- Import order in `analyses/__init__.py` matters: import `pip, cs, logbf, f1` before `paired` (which does `from analyses import pip, cs, logbf`) to avoid a circular import.
