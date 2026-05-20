# Plot Export Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate all dashboard plots as vector PDFs via Snakemake, driven by a restructured `plot_config.yaml` with separate supercollection and settings definitions.

**Architecture:** `plot_config.yaml` gains two top-level keys — `supercollections` (data + defaults) and `settings` (method comparison presets). A new `generate_plots.py` module reuses `viz_utils` render functions to produce PDFs. Snakemake rules wire data dependencies. Dashboard is simplified: prepare-collections section removed, config selection updated to two dropdowns (supercollection + plot_settings).

**Tech Stack:** Python, matplotlib (vector PDF backend), Snakemake, Polars, existing `viz_utils.py` and `plot_ready.py`

---

## File Structure

- **Modify:** `notebooks/plot_config.yaml` — migrate to two-key format
- **Create:** `generate_plots.py` — `make_plot(supercollection, plot_settings, plot_type, output_path)` and helpers
- **Modify:** `twogroup_experiments.snk` — add `plot_settings` wildcard, `supercollection_plot` rule, `all_plots` rule, update config loading
- **Modify:** `notebooks/dashboard.py` — remove prepare-collections cells, update config loading to new format, add plot_settings dropdown

---

## Section 1: `plot_config.yaml` restructure

Two top-level keys replace the current flat structure:

```yaml
supercollections:
  hallmark-signal-loc:
    collections:
      - {name: design=hallmark__enrichment=ser_enrich__signal=loc_0.50, alias: "mu=0.50"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=loc_1.00, alias: "mu=1.00"}
    default_settings:
      threshold: 2.0
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

settings:
  all_methods:
    method_families:
      - cox_heavy
      - logistic_oracle
      - logistic_threshold
      - cox_light_threshold
      - twogroup
      - twogroup_oracle
  cox_light_vs_logistic:
    method_families: [logistic_threshold, cox_light_threshold]
  cox_heavy_vs_twogroup:
    method_families: [cox_heavy, twogroup]
```

- Each supercollection has `collections` (list of name+alias) and `default_settings`
- Each settings entry has any subset of settings keys that override defaults
- Merge rule: `{**default_settings, **settings[plot_settings]}` produces effective settings
- All existing supercollection names in the current flat config migrate as-is under `supercollections:`
- The `_defaults` YAML anchor is removed (no longer needed — settings presets replace it)

---

## Section 2: Snakemake rules

### Constants (at snk startup)

```python
PLOT_TYPES = [
    "pip_calibration", "power_fdp", "causal_pip", "causal_rank",
    "mass_above_causal", "cs_dot_summary", "cs_power_fdp", "cs_beta_trace",
]
_PLOT_CONFIG = yaml.safe_load(Path(PLOT_CONFIG_PATH).read_text()) or {}
SUPERCOLLECTION_NAMES = sorted(_PLOT_CONFIG.get("supercollections", {}).keys())
PLOT_SETTINGS_NAMES = sorted(_PLOT_CONFIG.get("settings", {}).keys())
```

### Wildcard constraint (add to existing block)

```python
wildcard_constraints:
    plot_settings = r"[A-Za-z0-9_\-]+",
```

### New rules

```python
rule supercollection_plot:
    input:
        f"{RESULTS_ROOT}/supercollections/{{supercollection}}/out.txt",
    output:
        f"{RESULTS_ROOT}/plots/{{supercollection}}/{{plot_settings}}/{{plot_type}}.pdf",
    run:
        import generate_plots
        generate_plots.make_plot(
            wildcards.supercollection,
            wildcards.plot_settings,
            wildcards.plot_type,
            output[0],
        )

rule all_plots:
    input:
        expand(
            f"{RESULTS_ROOT}/plots/{{supercollection}}/{{plot_settings}}/{{plot_type}}.pdf",
            supercollection=SUPERCOLLECTION_NAMES,
            plot_settings=PLOT_SETTINGS_NAMES,
            plot_type=PLOT_TYPES,
        ),
```

`supercollection_plot` depends on `supercollections/{supercollection}/out.txt`, which is already produced by `rule materialize_supercollection` (all plot-ready parquets present). Snakemake parallelizes across plot types.

PDFs are vector graphics — matplotlib's PDF backend renders all lines, axes, and text as vectors. Scales without quality loss at any zoom or print size.

---

## Section 3: `generate_plots.py`

New top-level module. Reuses all existing `viz_utils` expand/make/render functions — no new rendering logic.

### Public API

```python
def make_plot(supercollection: str, plot_settings: str, plot_type: str, output_path: str) -> None:
    """Generate one plot type PDF for a (supercollection, plot_settings) combo."""
    cfg = _load_plot_config()
    settings = _resolve_settings(cfg, supercollection, plot_settings)
    combined_data = _load_supercollection_data(cfg, supercollection)
    fig = _PLOT_DISPATCH[plot_type](combined_data, settings)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
```

### Helpers

```python
def _load_plot_config() -> dict:
    path = Path(__file__).parent / "notebooks" / "plot_config.yaml"
    return yaml.safe_load(path.read_text()) or {}

def _resolve_settings(cfg: dict, supercollection: str, plot_settings: str) -> dict:
    defaults = cfg["supercollections"][supercollection].get("default_settings", {})
    overrides = cfg["settings"][plot_settings]
    return {**defaults, **overrides}

def _load_supercollection_data(cfg: dict, supercollection: str) -> dict:
    """Load plot-ready parquets for all collections in supercollection.
    Returns combined_data dict matching dashboard structure."""
    coll_list = cfg["supercollections"][supercollection]["collections"]
    collection_alias_root = Path(__file__).parent / "results" / "collections"
    aliases = {item["name"]: item.get("alias", item["name"]) for item in coll_list}
    bundles = {
        name: plot_ready.load_plot_ready_collection(collection_alias_root / name)
        for item in coll_list
        for name in [item["name"]]
    }
    combined_method_metadata = (
        pl.concat([b["method_metadata"] for b in bundles.values()])
        .unique(subset=["method", "threshold"])
    )
    def _tag(key):
        return pl.concat([
            b[key].with_columns(pl.lit(aliases.get(name, name)).alias("collection_name"))
            for name, b in bundles.items()
        ])
    return {
        "method_metadata": combined_method_metadata,
        "collection_names": [aliases.get(item["name"], item["name"]) for item in coll_list],
        "pip_plot_data": _tag("pip_plot_data"),
        "cs_plot_data": _tag("cs_plot_data"),
    }
```

### Dispatch functions

One private function per plot type, each mirroring the corresponding dashboard cell logic:

```python
def _make_pip_calibration(combined_data: dict, settings: dict) -> plt.Figure: ...
def _make_power_fdp(combined_data: dict, settings: dict) -> plt.Figure: ...
def _make_causal_pip(combined_data: dict, settings: dict) -> plt.Figure: ...
def _make_causal_rank(combined_data: dict, settings: dict) -> plt.Figure: ...
def _make_mass_above_causal(combined_data: dict, settings: dict) -> plt.Figure: ...
def _make_cs_dot_summary(combined_data: dict, settings: dict) -> plt.Figure: ...
def _make_cs_power_fdp(combined_data: dict, settings: dict) -> plt.Figure: ...
def _make_cs_beta_trace(combined_data: dict, settings: dict) -> plt.Figure: ...

_PLOT_DISPATCH = {
    "pip_calibration": _make_pip_calibration,
    "power_fdp": _make_power_fdp,
    "causal_pip": _make_causal_pip,
    "causal_rank": _make_causal_rank,
    "mass_above_causal": _make_mass_above_causal,
    "cs_dot_summary": _make_cs_dot_summary,
    "cs_power_fdp": _make_cs_power_fdp,
    "cs_beta_trace": _make_cs_beta_trace,
}
```

Settings keys consumed per plot type:
- All plots: `method_families`, `threshold`, `L`
- Power/FDP plots: also `max_fdp`
- CS plots: also `max_cs_size`, `min_log_bf`, `cs_beta`

---

## Section 4: Dashboard changes

### (i) Remove Prepare Collections

Delete these five cells entirely:
- `prepare_heading_cell`
- `unprepared_cell`
- `snakemake_cores_cell`
- `snakemake_prepare_cell`
- `dry_run_output_cell`

Dashboard becomes view-only. User runs snakemake from terminal before opening dashboard.

### (ii–iii) New config selection

Replace current single config dropdown + Load/Clear/Apply flow with:

```
[supercollection dropdown]  [plot_settings dropdown]  [Apply]
```

`supercollection` dropdown options = keys from `plot_config["supercollections"]`.
`plot_settings` dropdown options = `["(default)"] + keys from plot_config["settings"]`.

When Apply clicked:
- `selected` = all collections from chosen supercollection in config order
- `aliases` = alias map from supercollection config
- `settings` = `merge(default_settings, plot_config["settings"][plot_settings])` — or just `default_settings` if `"(default)"` chosen

Apply result feeds directly into `bundles_cell` (same as today — `apply_btn.value` structure unchanged).

Sliders (threshold, max_fdp, etc.) seed their initial values from the resolved settings. User can still adjust interactively after Apply.

Config save cell is removed (no write-back to config).

---

## What does NOT change

- All `render_*` functions in `viz_utils.py` — unchanged
- All `expand_*` / `make_*` functions — unchanged
- All Snakemake rules for data preparation — unchanged
- `plot_ready.py` — unchanged
- Dashboard plot cells (pip_calibration_cell, power_fdp_cell, etc.) — unchanged
