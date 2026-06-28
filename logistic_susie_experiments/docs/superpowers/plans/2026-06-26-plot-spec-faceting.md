# Plot-spec Faceting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate "what gets fit" from "how it's plotted" — derive plot dimensions from batch/method coordinates already in `MANIFEST`, declare faceting/aesthetics/filter in a `plots:` spec, and render through one generic facet engine.

**Architecture:** A new `viz_dims.py` maps raw coordinates → semantic + raw dimension columns, attached to every plot_data row at bundle-load. A new `viz_facet.py` applies a plot spec's filter and splits rows into a facet grid of series. Each analysis is reduced to two pure hooks — `aggregate(rows)->stats` and `draw(ax, stats, *, color, linestyle, label)` — driven by one generic renderer. `agg_*` analyses and `outputs:`/`method_filter`/`collection_name`-faceting are removed.

**Tech Stack:** Python 3.13, polars, matplotlib (pdf backend), PyYAML, Snakemake, pytest (run via `uv run`).

## Global Constraints

- Always run Python/pytest via `uv run` (never bare `python`/`pip`).
- Preserve sparsity: never densify BCOO designs (not touched here, but don't add `.to_dense()`).
- Plot data is ephemeral: no backward-compat shims; change schema and regenerate.
- Reductions (`reductions/pip.py`, `reductions/cs.py`) are NOT modified — they stay per-row; pooling happens at render.
- Spec: `docs/superpowers/specs/2026-06-26-plot-spec-faceting-design.md`.
- Work on branch `feat/plot-spec-faceting`.
- Filter operators are out of scope: scalar=equality, list=membership only.

---

## File Structure

- Create `viz_dims.py` — `method_dims`, `sim_dims` (coordinate → dims). Pure.
- Create `viz_facet.py` — `apply_filter`, `assign_groups`, `FacetGrid` dataclass. Pure.
- Create `tests/conftest.py`, `tests/test_viz_dims.py`, `tests/test_viz_facet.py`, `tests/test_plot_spec_loader.py`.
- Modify `experiments/loader.py` — attach dims in `load_sc_bundle`; add `resolve_plot_specs`/`plot_targets`/`plot_inputs`; drop `collection_name`-as-facet and `resolve_args`/`method_filter`.
- Create `analyses/hooks.py` — `AnalysisHook` protocol + `HOOKS` registry (`aggregate`,`draw` per analysis).
- Modify `analyses/pip.py`, `analyses/cs.py` — replace `_make_*`/`RENDERERS` with hook implementations.
- Modify `generate_plots.py` — `render_plot(bundle, spec, analysis, out)` using `viz_facet` + hooks.
- Modify `viz_utils.py` — add `dim_palette(values)`; keep low-level primitives, retire `facet_by_simulation`/`collection_names` paths as hooks absorb them.
- Modify `experiments/library.yaml` — remove all `agg_*` analyses + `agg_*` analysis_groups.
- Modify `experiments/000_global_local.yaml`, `001_profile_methods.yaml`, `002_global.yaml` — `outputs:` → `plots:`.
- Modify `logistic_susie_experiments.snk` — analyze rule + targets driven by plot specs.

---

## Task 1: Test harness + `viz_dims.py`

**Files:**
- Create: `viz_dims.py`
- Create: `tests/conftest.py`
- Create: `tests/test_viz_dims.py`

**Interfaces:**
- Produces: `method_dims(coord: dict) -> dict`, `sim_dims(coord: dict) -> dict`.

- [ ] **Step 1: Create `tests/conftest.py`** (put the package root on `sys.path` so `import viz_dims`, `import core` work, mirroring `twogroup_experiments/tests/conftest.py`)

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 2: Write failing tests** in `tests/test_viz_dims.py`

```python
from viz_dims import method_dims, sim_dims


def test_method_dims_irls_converged_fixed_centered():
    coord = {"function": "run_irls_method",
             "kwargs": {"ser_cadence": "block", "n_outer": 50, "L": 1,
                        "center": True, "estimate_prior_variance": False,
                        "prior_variance": 100.0}}
    d = method_dims(coord)
    assert d["family"] == "irls"
    assert d["step"] == "converged"
    assert d["prior"] == "fixed"
    assert d["center"] is True
    assert d["cadence"] == "block"
    assert d["L"] == 1
    assert d["function"] == "run_irls_method"
    assert d["m_n_outer"] == 50
    assert d["m_prior_variance"] == 100.0


def test_method_dims_one_step_eb_globaljj():
    coord = {"function": "run_globaljj_method",
             "kwargs": {"ser_cadence": "block", "n_outer": 1, "L": 1}}
    d = method_dims(coord)
    assert d["family"] == "globaljj"
    assert d["step"] == "one_step"
    assert d["prior"] == "eb"      # estimate_prior_variance absent -> EB
    assert d["center"] is False    # absent -> False


def test_method_dims_logistic_impl_family():
    coord = {"function": "run_logistic_method", "kwargs": {"impl": "globaljj", "L": 1}}
    assert method_dims(coord)["family"] == "globaljj"


def test_sim_dims_signal_and_raw():
    coord = {"design": {"function": "gaussian_markov_X",
                        "arguments": {"n": 500, "p": 100, "rho": 0.9}},
             "enrichment": {"function": "uniform_single_effect",
                            "arguments": {"causal_effect": 1.0}, "intercept": -2.0},
             "base_seed": 1}
    d = sim_dims(coord)
    assert d["design"] == "gaussian"
    assert d["intercept"] == -2.0
    assert d["b"] == 1.0
    assert d["signal"] is True
    assert d["d_rho"] == 0.9
    assert d["e_causal_effect"] == 1.0


def test_sim_dims_null_is_not_signal():
    coord = {"design": {"function": "c4_gene_sets_X", "arguments": {}},
             "enrichment": {"function": "uniform_single_effect",
                            "arguments": {"causal_effect": 0.0}, "intercept": -2.0},
             "base_seed": 1}
    d = sim_dims(coord)
    assert d["design"] == "c4"
    assert d["signal"] is False
    assert d["b"] == 0.0
```

- [ ] **Step 3: Run, verify fail**

Run: `uv run pytest tests/test_viz_dims.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'viz_dims'`).

- [ ] **Step 4: Implement `viz_dims.py`**

```python
"""Map raw MANIFEST coordinates to plot-facing dimensions.

Single source of truth for the dimension columns attached to every plot_data row.
Returns clean SEMANTIC dims plus flattened RAW args (m_*, d_*, e_*). Adding a knob
= one line here. No name/alias parsing.
"""
from __future__ import annotations

_DESIGN_NAMES = {"gaussian_markov_X": "gaussian", "uniform_markov_X": "uniform",
                 "c4_gene_sets_X": "c4", "hallmark_gene_sets_X": "hallmark",
                 "msigdb_gene_sets_X": "msigdb"}


def method_dims(coord: dict) -> dict:
    k = dict(coord.get("kwargs", {}))
    fn = coord.get("function", "")
    impl = k.get("impl", "")
    if "irls" in fn or impl == "irls":
        family = "irls"
    elif "globaljj" in fn or impl == "globaljj":
        family = "globaljj"
    elif "localjj" in fn or impl == "localjj":
        family = "localjj"
    elif "score" in fn:
        family = "score"
    else:
        family = impl or fn
    semantic = {
        "family": family,
        "step": "one_step" if k.get("n_outer") == 1 else "converged",
        "prior": "fixed" if k.get("estimate_prior_variance") is False else "eb",
        "center": bool(k.get("center", False)),
        "cadence": k.get("ser_cadence", "block"),
        "L": int(k.get("L", 1)),
        "function": fn,
    }
    raw = {f"m_{key}": v for key, v in k.items()}
    return {**semantic, **raw}


def sim_dims(coord: dict) -> dict:
    d = coord["design"]
    e = coord["enrichment"]
    b = float(e["arguments"].get("causal_effect", 0.0))
    semantic = {
        "design": _DESIGN_NAMES.get(d["function"], d["function"]),
        "intercept": float(e["intercept"]),
        "b": b,
        "signal": b != 0.0,
    }
    raw = {f"d_{key}": v for key, v in (d.get("arguments") or {}).items()}
    raw |= {f"e_{key}": v for key, v in (e.get("arguments") or {}).items()}
    return {**semantic, **raw}
```

- [ ] **Step 5: Run, verify pass**

Run: `uv run pytest tests/test_viz_dims.py -q`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add viz_dims.py tests/conftest.py tests/test_viz_dims.py
git commit -m "feat(viz): viz_dims coordinate->dimension mapping + tests"
```

---

## Task 2: Faceting engine `viz_facet.py`

**Files:**
- Create: `viz_facet.py`
- Create: `tests/test_viz_facet.py`

**Interfaces:**
- Consumes: nothing (operates on `list[dict]` rows).
- Produces:
  - `apply_filter(rows: list[dict], filt: dict) -> list[dict]`
  - `assign_groups(rows, *, facet_row=None, facet_col=None, color=None, linestyle=None) -> FacetGrid`
  - `FacetGrid` dataclass: `.row_keys: list`, `.col_keys: list`, `.cell(row_key, col_key) -> list[Series]`; `Series` has `.color_key`, `.linestyle_key`, `.rows: list[dict]`.

- [ ] **Step 1: Write failing tests** in `tests/test_viz_facet.py`

```python
from viz_facet import apply_filter, assign_groups

ROWS = [
    {"family": "irls", "step": "one_step", "design": "gaussian", "v": 1},
    {"family": "irls", "step": "converged", "design": "gaussian", "v": 2},
    {"family": "irls", "step": "converged", "design": "c4", "v": 3},
    {"family": "globaljj", "step": "converged", "design": "c4", "v": 4},
]


def test_apply_filter_scalar_equality():
    out = apply_filter(ROWS, {"family": "irls"})
    assert {r["v"] for r in out} == {1, 2, 3}


def test_apply_filter_list_membership():
    out = apply_filter(ROWS, {"design": ["c4"], "step": ["converged"]})
    assert {r["v"] for r in out} == {3, 4}


def test_apply_filter_empty_filter_passes_all():
    assert len(apply_filter(ROWS, {})) == 4


def test_assign_groups_facet_col_and_color():
    grid = assign_groups(ROWS, facet_col="design", color="family")
    assert grid.row_keys == [None]
    assert set(grid.col_keys) == {"gaussian", "c4"}
    c4 = grid.cell(None, "c4")
    colors = {s.color_key for s in c4}
    assert colors == {"irls", "globaljj"}
    irls_series = next(s for s in c4 if s.color_key == "irls")
    assert {r["v"] for r in irls_series.rows} == {3}


def test_assign_groups_series_pools_remaining_rows():
    # color only -> one facet cell, series by family pools across step/design
    grid = assign_groups(ROWS, color="family")
    cell = grid.cell(None, None)
    irls = next(s for s in cell if s.color_key == "irls")
    assert {r["v"] for r in irls.rows} == {1, 2, 3}  # pooled
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_viz_facet.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'viz_facet'`).

- [ ] **Step 3: Implement `viz_facet.py`**

```python
"""Pure faceting/aesthetic engine for the plot spec.

apply_filter: equality (scalar) / membership (list) over dimension columns.
assign_groups: split rows into a facet grid (facet_row x facet_col) of series
(color x linestyle); everything not on a channel is pooled into its series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


def apply_filter(rows: list[dict], filt: dict | None) -> list[dict]:
    if not filt:
        return list(rows)
    def keep(r: dict) -> bool:
        for key, want in filt.items():
            val = r.get(key)
            if isinstance(want, (list, tuple, set)):
                if val not in want:
                    return False
            elif val != want:
                return False
        return True
    return [r for r in rows if keep(r)]


@dataclass(frozen=True)
class Series:
    color_key: object
    linestyle_key: object
    rows: list[dict]


@dataclass
class FacetGrid:
    row_keys: list
    col_keys: list
    cells: dict = field(default_factory=dict)  # (row_key, col_key) -> list[Series]

    def cell(self, row_key, col_key) -> list:
        return self.cells.get((row_key, col_key), [])


def _distinct(rows, key):
    if key is None:
        return [None]
    seen = []
    for r in rows:
        v = r.get(key)
        if v not in seen:
            seen.append(v)
    return seen


def assign_groups(rows, *, facet_row=None, facet_col=None, color=None, linestyle=None) -> FacetGrid:
    row_keys = _distinct(rows, facet_row)
    col_keys = _distinct(rows, facet_col)
    color_keys = _distinct(rows, color)
    ls_keys = _distinct(rows, linestyle)

    def matches(r, key, val):
        return key is None or r.get(key) == val

    cells: dict = {}
    for rk, ck in product(row_keys, col_keys):
        series_list = []
        for colk, lsk in product(color_keys, ls_keys):
            sub = [r for r in rows
                   if matches(r, facet_row, rk) and matches(r, facet_col, ck)
                   and matches(r, color, colk) and matches(r, linestyle, lsk)]
            if sub:
                series_list.append(Series(colk, lsk, sub))
        cells[(rk, ck)] = series_list
    return FacetGrid(row_keys, col_keys, cells)
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_viz_facet.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add viz_facet.py tests/test_viz_facet.py
git commit -m "feat(viz): viz_facet filter + facet/series grouping engine + tests"
```

---

## Task 3: Attach dimensions in `load_sc_bundle` + palette

**Files:**
- Modify: `experiments/loader.py` (`load_sc_bundle`, ~lines 619-649)
- Modify: `viz_utils.py` (add `dim_palette`)
- Modify: `tests/conftest.py` (no change expected; verify import path)
- Create: `tests/test_bundle_dims.py`

**Interfaces:**
- Consumes: `viz_dims.method_dims`, `viz_dims.sim_dims`; `MANIFEST` via `manifest_cache.load_manifest_cached()`.
- Produces: every `*_plot_data` frame in the bundle gains dimension columns; `viz_utils.dim_palette(values: list) -> dict[value, hex]`.

- [ ] **Step 1: Write failing test** `tests/test_bundle_dims.py`

```python
import polars as pl
from experiments import loader


def test_bundle_rows_carry_method_and_sim_dims():
    cfg = loader.load_config()
    bundle = loader.load_sc_bundle(cfg, "002_global", ["pip"])
    df = bundle["pip_plot_data"]
    for col in ("family", "step", "prior", "center", "design", "intercept", "b", "signal"):
        assert col in df.columns, f"missing dim column {col}"
    fams = set(df["family"].unique().to_list())
    assert fams <= {"irls", "globaljj"}
    assert set(df["design"].unique().to_list()) <= {"gaussian", "c4"}
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_bundle_dims.py -q`
Expected: FAIL (dim columns absent).

- [ ] **Step 3: Add dim attachment in `experiments/loader.py`**

At top of file add import:
```python
from viz_dims import method_dims, sim_dims
```

In `load_sc_bundle`, replace the per-collection row assembly so each sub-frame gets dims joined by `method` and `batch_hash`. Concretely, after building `cmp = collection_method_pairs(...)` and before the reduction loop, build lookup dicts from the manifest:

```python
    from manifest_cache import load_manifest_cached
    manifest = load_manifest_cached()
    method_dim_by_hash = {h: method_dims(m) for h, m in manifest["methods"].items()}
    sim_dim_by_batch = {h: sim_dims(b["coordinate"]) for h, b in manifest["batches"].items()}
    # map method NAME -> dims (rows carry method name, not hash)
    method_dim_by_name = {m["name"]: method_dim_by_hash[h] for h, m in manifest["methods"].items()}
```

Then, where each reduction sub-frame is read per `(bh, mh, mname, ...)`, attach columns. Replace the existing `sub.append(pl.read_parquet(path))` with:

```python
                frame = pl.read_parquet(path)
                dims = {**method_dim_by_name.get(mname, {}), **sim_dim_by_batch.get(bh, {})}
                # only scalar dims become columns (skip nested); cast via pl.lit
                lit_cols = [pl.lit(v).alias(k) for k, v in dims.items()
                            if not isinstance(v, (list, dict, tuple))]
                sub.append(frame.with_columns(lit_cols))
```

Keep `collection_name` for now (harmless), but it is no longer required for faceting.

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_bundle_dims.py -q`
Expected: PASS.

- [ ] **Step 5: Add `dim_palette` to `viz_utils.py`** (near `method_family_color_map`)

```python
def dim_palette(values: list) -> dict:
    """Stable colorblind-safe palette keyed on a dimension's distinct values.
    Booleans/strings/numbers all supported; order-stable for legend determinism."""
    okabe_ito = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
                 "#56B4E9", "#AA4499", "#882255", "#332288", "#999999"]
    out = {}
    for i, v in enumerate(values):
        out[v] = okabe_ito[i % len(okabe_ito)]
    return out
```

- [ ] **Step 6: Commit**

```bash
git add experiments/loader.py viz_utils.py tests/test_bundle_dims.py
git commit -m "feat(viz): attach method/sim dims to bundle rows + dim_palette"
```

---

## Task 4: Plot-spec resolution in loader

**Files:**
- Modify: `experiments/loader.py` (replace `resolve_args`/`resolve_sc_analyses`; add `resolve_plot_specs`, `plot_analyses`, `plot_targets`, `plot_inputs`)
- Create: `tests/test_plot_spec_loader.py`

**Interfaces:**
- Consumes: a supercollection's `plots:` block.
- Produces:
  - `resolve_plot_specs(config, sc_name) -> dict[str, dict]` (plot name -> spec dict with `analysis`, channel keys, `filter`).
  - `plot_analyses(config, sc_name) -> list[tuple[analysis, plot_name]]` (drives Snakefile targets).
  - `resolve_plot_spec(config, sc_name, plot_name) -> dict`.

- [ ] **Step 1: Add a `plots:` block to `002_global.yaml`** (temporary minimal, expanded in Task 8) so the loader has something to resolve. Replace the `outputs:` block with:

```yaml
    plots:
      family: {analysis: pip_calibration, color: family, facet_col: design, filter: {step: converged, prior: fixed, center: false}}
```

- [ ] **Step 2: Write failing tests** `tests/test_plot_spec_loader.py`

```python
from experiments import loader


def test_resolve_plot_specs_reads_plots_block():
    cfg = loader.load_config()
    specs = loader.resolve_plot_specs(cfg, "002_global")
    assert "family" in specs
    assert specs["family"]["analysis"] == "pip_calibration"
    assert specs["family"]["color"] == "family"
    assert specs["family"]["filter"]["center"] is False


def test_plot_analyses_pairs():
    cfg = loader.load_config()
    pairs = loader.plot_analyses(cfg, "002_global")
    assert ("pip_calibration", "family") in pairs
```

- [ ] **Step 3: Run, verify fail**

Run: `uv run pytest tests/test_plot_spec_loader.py -q`
Expected: FAIL (`AttributeError: ... resolve_plot_specs`).

- [ ] **Step 4: Implement loader functions** in `experiments/loader.py`

```python
_CHANNEL_KEYS = ("color", "linestyle", "facet_row", "facet_col")


def resolve_plot_specs(config: dict, sc_name: str) -> dict:
    sc = config["supercollections"][sc_name]
    out = {}
    for name, spec in (sc.get("plots") or {}).items():
        if "analysis" not in spec:
            raise KeyError(f"plot {name!r} in {sc_name!r} missing 'analysis'")
        out[name] = {
            "name": name,
            "analysis": spec["analysis"],
            "filter": dict(spec.get("filter") or {}),
            **{k: spec.get(k) for k in _CHANNEL_KEYS},
        }
    return out


def resolve_plot_spec(config: dict, sc_name: str, plot_name: str) -> dict:
    specs = resolve_plot_specs(config, sc_name)
    if plot_name not in specs:
        raise KeyError(f"No plot named {plot_name!r} in {sc_name!r}")
    return specs[plot_name]


def plot_analyses(config: dict, sc_name: str) -> list[tuple]:
    return [(s["analysis"], s["name"]) for s in resolve_plot_specs(config, sc_name).values()]
```

Delete `resolve_args` and `resolve_sc_analyses` (replaced); update any references found via grep in Task 8.

- [ ] **Step 5: Run, verify pass**

Run: `uv run pytest tests/test_plot_spec_loader.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/loader.py experiments/002_global.yaml tests/test_plot_spec_loader.py
git commit -m "feat(loader): resolve plots: specs (replaces outputs/method_filter)"
```

---

## Task 5: Analysis hook protocol + generic driver (with `pip_calibration`)

**Files:**
- Create: `analyses/hooks.py`
- Modify: `analyses/pip.py` (add `pip_calibration` hook; keep old `_make_*` until Task 6/7 remove them)
- Modify: `generate_plots.py` (`render_plot`)
- Create: `tests/test_render_plot.py`

**Interfaces:**
- Consumes: `viz_facet.apply_filter`, `viz_facet.assign_groups`; `viz_utils.dim_palette`; bundle with dim columns.
- Produces:
  - `analyses/hooks.py`: `HOOKS: dict[str, AnalysisHook]`; `AnalysisHook` has `requires: str` (reduction key, e.g. "pip"), `aggregate(rows: list[dict]) -> object`, `draw(ax, stats, *, color, linestyle, label) -> None`.
  - `generate_plots.render_plot(bundle: dict, spec: dict, output_path: str) -> None`.

- [ ] **Step 1: Write failing test** `tests/test_render_plot.py`

```python
from pathlib import Path
from experiments import loader
import generate_plots


def test_render_plot_writes_pdf(tmp_path):
    cfg = loader.load_config()
    spec = loader.resolve_plot_spec(cfg, "002_global", "family")
    bundle = loader.load_sc_bundle(cfg, "002_global", [spec_reduction(spec)])
    out = tmp_path / "family.pdf"
    generate_plots.render_plot(bundle, spec, str(out))
    assert out.exists() and out.stat().st_size > 0


def spec_reduction(spec):
    from analyses.hooks import HOOKS
    return HOOKS[spec["analysis"]].requires
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_render_plot.py -q`
Expected: FAIL (`No module named 'analyses.hooks'`).

- [ ] **Step 3: Create `analyses/hooks.py`**

```python
"""Per-analysis geometry hooks + registry.

Each analysis provides: a `requires` reduction key, an `aggregate(rows)->stats`
that pools a series' rows into the plotted statistic (must be grouping-invariant),
and a `draw(ax, stats, *, color, linestyle, label)` that renders one series.
The generic driver (generate_plots.render_plot) owns filtering and faceting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AnalysisHook:
    requires: str
    aggregate: Callable
    draw: Callable


HOOKS: dict[str, AnalysisHook] = {}


def register(name: str, requires: str):
    def deco(_unused=None):
        return None
    return deco


def add_hook(name: str, requires: str, aggregate, draw):
    HOOKS[name] = AnalysisHook(requires=requires, aggregate=aggregate, draw=draw)
```

- [ ] **Step 4: Add the `pip_calibration` hook in `analyses/pip.py`**

Read the existing data shaping in `viz_utils.expand_pip_calibration_from_compact` and the per-curve drawing in `viz_utils.render_pip_calibration`. The reduction emits per-row `pip_bin_counts` and `pip_bin_causal_counts` (lists). The hook:

```python
import numpy as np
from analyses.hooks import add_hook


def _pip_calibration_aggregate(rows):
    # pool bin counts across all rows in the series (grouping-invariant: sum)
    total = None
    causal = None
    for r in rows:
        c = np.asarray(r["pip_bin_counts"], dtype=float)
        k = np.asarray(r["pip_bin_causal_counts"], dtype=float)
        total = c if total is None else total + c
        causal = k if causal is None else causal + k
    n_bins = len(total)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        emp = np.where(total > 0, causal / total, np.nan)
    return {"centers": centers, "empirical": emp, "counts": total}


def _pip_calibration_draw(ax, stats, *, color, linestyle, label):
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, zorder=0)  # y=x reference
    m = stats["counts"] > 0
    ax.plot(stats["centers"][m], stats["empirical"][m],
            color=color, linestyle=linestyle, marker="o", ms=3, lw=1.5, label=label)
    ax.set_xlabel("Predicted PIP"); ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


add_hook("pip_calibration", "pip", _pip_calibration_aggregate, _pip_calibration_draw)
```

Ensure `analyses/__init__.py` imports `analyses.pip` and `analyses.cs` so `add_hook` runs at import (registry populated). Verify/append:

```python
from analyses import pip as _pip   # noqa: F401  (populates HOOKS)
from analyses import cs as _cs     # noqa: F401
```

- [ ] **Step 5: Implement `render_plot` in `generate_plots.py`**

```python
import matplotlib.pyplot as plt
from pathlib import Path

import viz_utils
from analyses.hooks import HOOKS
from viz_facet import apply_filter, assign_groups

_LINESTYLES = ["-", "--", ":", "-."]


def render_plot(bundle: dict, spec: dict, output_path: str) -> None:
    hook = HOOKS[spec["analysis"]]
    df = bundle[f"{hook.requires}_plot_data"]
    rows = df.to_dicts() if hasattr(df, "to_dicts") else list(df)
    rows = apply_filter(rows, spec.get("filter"))

    grid = assign_groups(rows, facet_row=spec.get("facet_row"),
                         facet_col=spec.get("facet_col"),
                         color=spec.get("color"), linestyle=spec.get("linestyle"))

    color_vals = sorted({s.color_key for c in grid.cells.values() for s in c}, key=str)
    palette = viz_utils.dim_palette(color_vals)
    ls_vals = sorted({s.linestyle_key for c in grid.cells.values() for s in c}, key=str)
    ls_map = {v: _LINESTYLES[i % len(_LINESTYLES)] for i, v in enumerate(ls_vals)}

    nr, nc = max(1, len(grid.row_keys)), max(1, len(grid.col_keys))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.4 * nr), squeeze=False)
    for i, rk in enumerate(grid.row_keys):
        for j, ck in enumerate(grid.col_keys):
            ax = axes[i][j]
            for s in grid.cell(rk, ck):
                stats = hook.aggregate(s.rows)
                label = " ".join(str(x) for x in (s.color_key, s.linestyle_key) if x is not None)
                hook.draw(ax, stats, color=palette[s.color_key],
                          linestyle=ls_map[s.linestyle_key], label=label or None)
            title = " | ".join(str(x) for x in (rk, ck) if x is not None)
            if title:
                ax.set_title(title, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="center right", fontsize=8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 6: Run, verify pass**

Run: `uv run pytest tests/test_render_plot.py -q`
Expected: PASS (writes a non-empty PDF). Requires existing reductions for 002_global (already present in `results/`).

- [ ] **Step 7: Commit**

```bash
git add analyses/hooks.py analyses/pip.py analyses/__init__.py generate_plots.py tests/test_render_plot.py
git commit -m "feat(viz): analysis hook protocol + generic render_plot (pip_calibration)"
```

---

## Task 6: Convert remaining pip-family analyses to hooks

**Files:**
- Modify: `analyses/pip.py` (add hooks: `power_fdp`, `causal_pip`, `mass_above_causal`; remove `_make_*` + `RENDERERS`)
- Modify: `tests/test_render_plot.py` (parametrize over the 4 pip analyses)

**Interfaces:**
- Consumes: `analyses.hooks.add_hook`; existing data-shaping in `viz_utils.expand_power_fdp_from_compact`, `expand_causal_pip_*`, `expand_mass_above_causal_*` (read current `_make_*` for the exact expand calls).
- Produces: `HOOKS["power_fdp"]`, `HOOKS["causal_pip"]`, `HOOKS["mass_above_causal"]`.

- [ ] **Step 1: Parametrized failing test** — extend `tests/test_render_plot.py`

```python
import pytest
import generate_plots
from experiments import loader
from analyses.hooks import HOOKS


@pytest.mark.parametrize("analysis", ["pip_calibration", "power_fdp", "causal_pip", "mass_above_causal"])
def test_pip_family_hooks_render(tmp_path, analysis):
    assert analysis in HOOKS
    cfg = loader.load_config()
    spec = {"name": "t", "analysis": analysis, "color": "family",
            "facet_col": "design", "filter": {"step": "converged", "prior": "fixed", "center": False}}
    bundle = loader.load_sc_bundle(cfg, "002_global", [HOOKS[analysis].requires])
    out = tmp_path / f"{analysis}.pdf"
    generate_plots.render_plot(bundle, spec, str(out))
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_render_plot.py -q`
Expected: FAIL for `power_fdp`/`causal_pip`/`mass_above_causal` (`KeyError` not in HOOKS).

- [ ] **Step 3: Implement the three hooks** in `analyses/pip.py`.

For each, read the current `_make_<name>` (lines 34-90 of `analyses/pip.py`) and the `viz_utils.expand_*`/`render_*` it calls; move the per-series statistic into `aggregate` and the per-series drawing into `draw`. Specifics:

- `power_fdp` (requires "pip"): `aggregate` builds a power-vs-FDP curve by sweeping the PIP threshold over pooled rows (reuse `viz_utils.expand_power_fdp_from_compact` logic on the series rows; it returns sorted FDP/power arrays). `draw` plots FDP (x) vs power (y), `square_axes`, `xlim=[0,max_fdp]`. `max_fdp` read from `spec` is not available in the hook signature — bake the default 0.5 into `draw` (or read from a module constant); range filtering of FDP stays in `draw`.
- `causal_pip` (requires "pip", causal sims only — enforce in `aggregate` by ignoring rows with empty `causal_indices`): `aggregate` collects causal PIPs; `draw` an ECDF or histogram per the current `render_causal_pip`.
- `mass_above_causal` (requires "cs" per current `_make_mass_above_causal`; confirm by reading it — it reads cs_plot_data): set `requires="cs"`; `aggregate`/`draw` mirror current renderer.

Write the actual code by porting each existing renderer's body (they already contain the full logic). Do not leave any `pass`/TODO.

- [ ] **Step 4: Remove `_make_*` pip functions and the `RENDERERS` dict** from `analyses/pip.py` (now superseded by `HOOKS`).

- [ ] **Step 5: Run, verify pass**

Run: `uv run pytest tests/test_render_plot.py -q`
Expected: PASS (4 analyses render).

- [ ] **Step 6: Commit**

```bash
git add analyses/pip.py tests/test_render_plot.py
git commit -m "feat(viz): port pip-family analyses to hooks; drop _make_/RENDERERS"
```

---

## Task 7: Convert cs-family analyses to hooks

**Files:**
- Modify: `analyses/cs.py` (add 12 hooks; remove `_make_*` + `agg_*` + `RENDERERS`)
- Modify: `tests/test_render_plot.py` (parametrize cs analyses)

**Interfaces:**
- Produces: `HOOKS[...]` for: `causal_rank`, `preceding_posterior_mass_ecdf`, `cs_dot_summary`, `cs_calibrated_dot`, `cs_size_power`, `cs_radius_power`, `cs_power_fdp`, `cs_power_size_coverage_trace`, `cs_coverage_trace`, `cs_coverage_size`, `cs_coverage_radius`, `cs_calibration`. All `requires="cs"`.

- [ ] **Step 1: Parametrized failing test** — extend `tests/test_render_plot.py` with the 12 cs analyses (same shape as Task 6 Step 1, `color="family"`, `facet_col="design"`, `filter={"signal": True, "step": "converged", "prior": "fixed", "center": False}` so coverage-bearing analyses have causal sims).

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_render_plot.py -q -k cs_`
Expected: FAIL (cs analyses not in HOOKS).

- [ ] **Step 3: Implement hooks** by porting each `_make_<name>` in `analyses/cs.py` (read lines for each; they call `viz_utils.expand_*`/`render_*`). For each: `aggregate` pools the series' cs rows into the statistic the current renderer computes (coverage, size, power, rank, ecdf, calibration bins); `draw` renders one series. Map the current `simulation_filter: has_causal` to checking causal presence inside `aggregate` (skip non-causal rows) — the spec `filter: {signal: true}` already restricts, but keep the guard.

  **Grouping-invariance caveat (from spec):** `cs_dot_summary` and `cs_calibrated_dot` are per-method scalar/dot summaries. Verify their `aggregate` is well-defined over a pooled series; if a dot represents one method, ensure `color`/`facet` channels separate methods so each series is one method. If a summary genuinely cannot pool, render it as one dot per series (color/facet must fully identify the method) — document this constraint in a comment in the hook.

  Port the full body of each renderer; no placeholders.

- [ ] **Step 4: Remove all `_make_*`, `agg_*` make-functions, and `RENDERERS`** from `analyses/cs.py`.

- [ ] **Step 5: Run, verify pass**

Run: `uv run pytest tests/test_render_plot.py -q`
Expected: PASS (all 16 analyses render).

- [ ] **Step 6: Commit**

```bash
git add analyses/cs.py tests/test_render_plot.py
git commit -m "feat(viz): port cs-family analyses to hooks; drop _make_/agg_/RENDERERS"
```

---

## Task 8: Library + experiment yaml migration; remove agg_*

**Files:**
- Modify: `experiments/library.yaml` (remove all `agg_*` analyses + `agg_*` group members; analysis_groups may be removed entirely if unused)
- Modify: `experiments/002_global.yaml` (full `plots:` block)
- Modify: `experiments/000_global_local.yaml`, `experiments/001_profile_methods.yaml` (`outputs:` → `plots:`)
- Modify: `experiments/loader.py` (remove dead `flatten_analyses`, `analysis_groups` handling, `reduction_method_filter`/`method_filter` paths if now unused — confirm via grep)
- Create: `tests/test_config_migration.py`

- [ ] **Step 1: Write failing test** `tests/test_config_migration.py`

```python
from experiments import loader


def test_no_agg_analyses_remain():
    cfg = loader.load_config()
    assert not any(a.startswith("agg_") for a in cfg["library"]["analyses"])


def test_all_supercollections_use_plots():
    cfg = loader.load_config()
    for name, sc in cfg["supercollections"].items():
        assert "outputs" not in sc, f"{name} still has outputs:"
        assert "plots" in sc, f"{name} missing plots:"


def test_002_global_pairwise_plots_present():
    cfg = loader.load_config()
    specs = loader.resolve_plot_specs(cfg, "002_global")
    assert {"step", "prior", "center", "family", "all"} <= set(specs)
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_config_migration.py -q`
Expected: FAIL.

- [ ] **Step 3: Edit `experiments/library.yaml`** — delete every `agg_*:` entry under `analyses:` and every `agg_*` line under `analysis_groups:` (or remove `analysis_groups:` wholesale if the new `plots:` don't reference groups).

- [ ] **Step 4: Rewrite `002_global.yaml` `plots:`** (one figure per contrast; each generates across the key analyses by listing them, OR one plot entry per (analysis, contrast)). Use per-analysis explicit entries for clarity:

```yaml
    plots:
      # one-step vs converged (color=step, linestyle=family), per design
      step__pip_calibration: {analysis: pip_calibration, color: step, linestyle: family, facet_col: design, filter: {prior: fixed, center: false}}
      step__power_fdp:        {analysis: power_fdp,        color: step, linestyle: family, facet_col: design, filter: {prior: fixed, center: false}}
      step__cs_calibration:   {analysis: cs_calibration,   color: step, linestyle: family, facet_col: design, filter: {prior: fixed, center: false, signal: true}}
      step__cs_size_power:    {analysis: cs_size_power,     color: step, linestyle: family, facet_col: design, filter: {prior: fixed, center: false, signal: true}}
      # eb vs fixed
      prior__pip_calibration: {analysis: pip_calibration, color: prior, linestyle: family, facet_col: design, filter: {step: converged, center: false}}
      prior__power_fdp:       {analysis: power_fdp,        color: prior, linestyle: family, facet_col: design, filter: {step: converged, center: false}}
      prior__cs_calibration:  {analysis: cs_calibration,   color: prior, linestyle: family, facet_col: design, filter: {step: converged, center: false, signal: true}}
      prior__cs_size_power:   {analysis: cs_size_power,     color: prior, linestyle: family, facet_col: design, filter: {step: converged, center: false, signal: true}}
      # centered vs not (irls only) -- color=center now distinct
      center__pip_calibration: {analysis: pip_calibration, color: center, facet_col: design, filter: {family: irls, step: converged, prior: fixed}}
      center__power_fdp:       {analysis: power_fdp,        color: center, facet_col: design, filter: {family: irls, step: converged, prior: fixed}}
      center__cs_calibration:  {analysis: cs_calibration,   color: center, facet_col: design, filter: {family: irls, step: converged, prior: fixed, signal: true}}
      center__cs_size_power:   {analysis: cs_size_power,     color: center, facet_col: design, filter: {family: irls, step: converged, prior: fixed, signal: true}}
      # irls vs jj
      family__pip_calibration: {analysis: pip_calibration, color: family, facet_col: design, filter: {step: converged, prior: fixed, center: false}}
      family__power_fdp:       {analysis: power_fdp,        color: family, facet_col: design, filter: {step: converged, prior: fixed, center: false}}
      family__cs_calibration:  {analysis: cs_calibration,   color: family, facet_col: design, filter: {step: converged, prior: fixed, center: false, signal: true}}
      family__cs_size_power:   {analysis: cs_size_power,     color: family, facet_col: design, filter: {step: converged, prior: fixed, center: false, signal: true}}
      # overview (all methods; color=family, linestyle=step, facet rows=prior, cols=design)
      all__pip_calibration: {analysis: pip_calibration, color: family, linestyle: step, facet_row: prior, facet_col: design}
      all__power_fdp:       {analysis: power_fdp,        color: family, linestyle: step, facet_row: prior, facet_col: design}
```

Define the test's `{"step","prior","center","family","all"}` membership by treating the plot-name prefix before `__` as the contrast; update `test_002_global_pairwise_plots_present` to check prefixes if needed, or rename plots to bare contrast names with a single analysis each. (Pick one; keep test and yaml consistent.)

- [ ] **Step 5: Convert `000_global_local.yaml` and `001_profile_methods.yaml`** `outputs:` → `plots:`. Port each old `output` (its `method_filter` + analyses) to plot specs: replace `method_filter: [a, b]` with a `filter:` on the dims that select those methods (e.g. `{family: [...], step: ...}`), choosing `color`/`facet` to match the previous comparison. For the L=5 SuSiE collection, include `filter: {L: 5}` where relevant.

- [ ] **Step 6: Remove dead loader code** — grep and delete now-unused `flatten_analyses`, `resolve_sc_analyses` (already gone), `reduction_method_filter`, `method_filter` plumbing, `analysis_groups` handling, if no longer referenced:

Run: `grep -rn "flatten_analyses\|analysis_groups\|method_filter\|resolve_args" experiments/ *.py *.snk`
Delete each hit that is now dead; keep any still used by the Snakefile until Task 9.

- [ ] **Step 7: Run, verify pass**

Run: `uv run pytest tests/test_config_migration.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add experiments/
git commit -m "feat(config): migrate experiments to plots:; drop agg_* analyses"
```

---

## Task 9: Snakefile analyze path on plot specs

**Files:**
- Modify: `logistic_susie_experiments.snk` (analyze rule + plot targets)
- Modify: `analyses/pip.py`, `analyses/cs.py` (replace the `if "snakemake" in globals()` blocks that call `render_analysis` with `render_plot`)

**Interfaces:**
- Consumes: `loader.plot_analyses`, `loader.resolve_plot_spec`, `loader.analysis_requires` (still maps analysis→reduction via `HOOKS[...].requires`), `generate_plots.render_plot`.

- [ ] **Step 1: Inspect current analyze rule + all_plots target**

Run: `grep -n "analyze\|all_plots\|args_name\|render_analysis\|resolve_sc_analyses\|supercollection" logistic_susie_experiments.snk`
Read the rules referenced (around lines 145-180).

- [ ] **Step 2: Update path layout + targets.** Output path becomes
`results/supercollections/{supercollection}/{analysis}/{plot_name}.pdf` (same shape as today, `args_name`→`plot_name`). Build the `all_plots` input list from `loader.plot_analyses(config, sc)` for every supercollection:

```python
def _all_plot_targets():
    paths = []
    for sc in CONFIG["supercollections"]:
        for analysis, plot_name in loader.plot_analyses(CONFIG, sc):
            paths.append(f"{RESULTS_ROOT}/supercollections/{sc}/{analysis}/{plot_name}.pdf")
    return paths
```

- [ ] **Step 3: Update the analyze rule** so its inputs are the reductions required by `HOOKS[analysis].requires` (filtered to the plot's `filter` where it restricts simulations/methods — but simplest correct version: require all reductions for that reduction key across the supercollection, as today). Pass `analysis` and `plot_name` to the script via `params`. Inputs computed by a helper analogous to the existing `analysis_inputs`, keyed on the reduction `HOOKS[analysis].requires`.

- [ ] **Step 4: Update the `if "snakemake" in globals()` blocks** in `analyses/pip.py` and `analyses/cs.py` to:

```python
if "snakemake" in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).parent.parent)
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    import generate_plots
    from experiments import loader as _loader
    _wc = snakemake.wildcards
    _cfg = _loader.load_config()
    _spec = _loader.resolve_plot_spec(_cfg, _wc.supercollection, _wc.plot_name)
    from analyses.hooks import HOOKS
    _bundle = _loader.load_sc_bundle(_cfg, _wc.supercollection, [HOOKS[_spec["analysis"]].requires])
    generate_plots.render_plot(_bundle, _spec, snakemake.output[0])
```

(Only one such block is needed; consolidate into a single analyze script if the rule currently dispatches per-family. If the rule maps analysis→family module, route by `analysis_family` still, but both call `render_plot`.)

- [ ] **Step 5: Dry-run**

Run: `PYTHONPATH=. uv run snakemake -s logistic_susie_experiments.snk -n`
Expected: DAG builds, no errors; analyze targets use `{plot_name}.pdf`.

- [ ] **Step 6: Build the 002 plots for real**

Run: `PYTHONPATH=. uv run snakemake -s logistic_susie_experiments.snk --cores 4 all_plots`
Expected: completes; `results/supercollections/002_global/pip_calibration/family.pdf` (etc.) regenerated.

- [ ] **Step 7: Commit**

```bash
git add logistic_susie_experiments.snk analyses/pip.py analyses/cs.py
git commit -m "feat(pipeline): analyze rule renders plot specs via render_plot"
```

---

## Task 10: Browser + symlinks + end-to-end sanity

**Files:**
- Modify: `scripts/symlink_plots.py` (rename `args_name`→`plot_name` in comments/vars only; path shape unchanged)
- Modify: `scripts/plot_browser.py` (confirm "Comparison" axis = `plot_name`; no logic change expected)

- [ ] **Step 1: Refresh symlinks**

Run: `PYTHONPATH=. uv run python scripts/symlink_plots.py`
Expected: creates `by_type/{analysis}/{plot_name}/{sc}.pdf` and `by_sc/{sc}/{plot_name}/{analysis}.pdf`.

- [ ] **Step 2: Full test suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (viz_dims, viz_facet, bundle_dims, plot_spec_loader, render_plot×16, config_migration).

- [ ] **Step 3: Visual spot check** — open `results/supercollections/002_global/pip_calibration/center__pip_calibration.pdf` and confirm the two `center` series have distinct colors and the design facets are present.

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "chore(plot): refresh symlinks + browser naming for plot specs"
```

---

## Self-Review notes

- **Spec coverage:** viz_dims (Task 1) ✓; dims-as-columns + attachment (Task 3) ✓; plot spec schema + filter (Task 4) ✓; renderer hooks + generic driver (Task 5) ✓; agg elimination (Task 8) ✓; full migration of analyses (Tasks 6-7) + experiments (Task 8) + single Snakefile path (Task 9) ✓; filter eq+membership only (Task 2) ✓; browser/symlink (Task 10) ✓; grouping-invariance caveat (Task 7 Step 3) ✓.
- **Open risk carried into execution:** Tasks 6/7 port existing renderer bodies — the implementer must read each current `_make_*`/`viz_utils.expand_*`/`render_*` and move its logic into `aggregate`/`draw`. These are refactors of existing, working code (not new logic), with exact source pointers given.
- **`max_fdp`/settings:** previously came from `default_args`; now bake sensible defaults into the relevant `draw` hooks (power_fdp max_fdp=0.5). If per-plot control is needed later, add an optional `params:` map to the plot spec (out of scope now).
