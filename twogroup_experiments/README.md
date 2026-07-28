# Twogroup Experiments Pipeline

## Quick Reference

```bash
# 1. Add/change simulations or methods → regenerate manifest
uv run python config.py

# 2. Run all fits + build plot-ready data
snakemake --snakefile twogroup_experiments.snk all_collections -j<N>

# 3. Generate all plots for all supercollections
snakemake --snakefile twogroup_experiments.snk all_plots -j<N>

# 4. Generate plots for one supercollection
snakemake --snakefile twogroup_experiments.snk supercollection_all_plots \
  --config supercollection=hallmark-signal-loc -j<N>

# 5. Sync plots from Midway (dereferences symlinks → local copies)
./scripts/sync_plots.sh [user@midway3.rcc.uchicago.edu]

# 6. Browse results interactively
uv run marimo run notebooks/dashboard.py
```

**Key files:**
- `config.py` — simulation and method specs, writes `results/manifest.json`
- `notebooks/plot_config.yaml` — collections, supercollections, plot settings
- `twogroup_experiments.snk` — Snakemake workflow

---

## 1. Adding Simulations and Methods

All simulation and method specs live in `config.py`. After any change, regenerate the manifest:

```bash
uv run python config.py
```

This writes `results/manifest.json`, which Snakemake uses as its config. Existing results
are unaffected — Snakemake only runs rules whose outputs are missing.

### Adding a simulation

Simulations are `SimulationSpec` objects. Each spec needs a unique `name`, a `design_sampler`
(generates the feature matrix X), an `effect_sampler`, a prior `f0`/`f1`, and `base_seed`.

Example — adding a new Markov design at `rho=0.7`:

```python
from functools import partial
new_sim = SimulationSpec(
    name="design=gaussian_markov_rho_0.70_n_features_100__enrichment=ser_enrich__signal=loc_1.50",
    design_sampler=partial(gaussian_markov_X, n=500, p=100, rho=0.7),
    effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
    intercept=-2.0,
    f0=F0,
    f1=fixed_normal(loc=1.5, scale=0.1),
    base_seed=BASE_SEED,
)
REGISTRY.register_simulations((new_sim,))
```

Then register batches for it:

```python
new_batches = batch_specs_for_simulation(
    new_sim,
    replicates_per_batch=REPLICATES_PER_BATCH,
    n_batches=N_BATCHES,
)
REGISTRY.register_batches(new_batches)
```

### Adding a method

Methods are `MethodSpec` objects. Each needs a `name`, a `fit_function`,
a `summarize_function`, and `kwargs` forwarded to the fitter.

```python
new_method = MethodSpec(
    name="my_method_L1",
    fit_function=fit_twogroup_method,
    summarize_function=summarize_twogroup_method,
    kwargs={"f1": F1INIT, "L": 1, "n_null_iter": 20, "n_intercept_iter": 20},
)
REGISTRY.register_methods((new_method,))
```

The manifest hash is derived from the full spec, so renaming or changing kwargs
produces a new hash and invalidates old results for that method.

---

## 2. Requesting Simulations: Collections and Supercollections

Fits and plot-ready data are organized around two levels defined in
`notebooks/plot_config.yaml`.

### Collections

A **collection** is a set of simulation batches × methods to fit. Each collection
maps to one directory under `results/collections/{name}/plot_ready/`.

Collections are defined under the `collections:` key. If no entry exists for a
simulation name, the collection defaults to using the `default` method collection
and the simulation spec matching that name.

**Custom collection** — override method selection:

```yaml
collections:
  design=hallmark__enrichment=ser_enrich__signal=loc_1.50:
    method_collections:
      - twogroup_ser      # only fit twogroup-family methods
```

**Union collection** — combine multiple simulations into one collection
(useful for aggregating across simulation scenarios in a single plot):

```yaml
collections:
  hallmark-loc-weak:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=loc_0.50
      - design=hallmark__enrichment=ser_enrich__signal=loc_1.00
    method_collections:
      - default
```

### Method collections

`method_collections:` defines named groups of methods to fit. Each entry filters
the full method bank by `method_families`, `L`, and/or `thresholds`:

```yaml
method_collections:
  default:
    method_families: [cox_heavy, logistic_oracle, logistic_threshold,
                      cox_light_threshold, twogroup, twogroup_scale_fam,
                      twogroup_loc_fam, twogroup_oracle]
    L: [1]
    thresholds: [0.0, 1.0, 2.0, 3.0, 4.0]
  twogroup_ser:
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init,
                      twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
```

### Supercollections

A **supercollection** groups multiple collections for joint plotting. Each collection
entry has a `name` (matching a key in `collections:` or an implicit single-sim name)
and an `alias` used as the x-axis label in plots.

```yaml
supercollections:
  hallmark-signal-loc:
    collections:
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_0.50
        alias: mu=0.50
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_1.00
        alias: mu=1.00
      ...
    default_settings:
      thresholds: [2.0]
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots:
      - settings: [all_methods, twogroup_methods]
        plot_type_groups: [standard, cs]
```

Run fits + plot data for a single supercollection:

```bash
snakemake --snakefile twogroup_experiments.snk \
  results/supercollections/hallmark-signal-loc/out.txt -j<N>
```

---

## 3. Plots

Plots are PDFs at `results/supercollections/{sc}/{plot_type}/{settings}.pdf`.
Symlinks at `results/plots/{plot_type}/{settings}/{sc}.pdf` allow browsing
across supercollections (created by `scripts/symlink_plots.py`).

### Plot types

| Group | Types |
|---|---|
| `standard` | `pip_calibration`, `power_fdp`, `causal_pip`, `causal_rank`, `mass_above_causal` + `agg_*` variants |
| `cs` | `cs_dot_summary`, `cs_power_fdp`, `cs_beta_trace` + `agg_*` variants |

### Specifying plots in a supercollection

The `plots:` key is a list of entries. Each entry specifies which settings names
and plot type groups to generate. All (settings × plot_types) combinations are produced:

```yaml
plots:
  - settings: [all_methods, twogroup_methods]
    plot_type_groups: [standard, cs]
  - settings: [threshold_methods]
    plot_types: [pip_calibration]      # explicit types also work
```

### Plot settings

Settings are defined under the top-level `settings:` key. Each named setting
is a dict of overrides applied on top of the supercollection's `default_settings`.
`null` means "no overrides" (use defaults as-is).

| Name | Effect |
|---|---|
| `all_methods` | All fit methods, threshold=2.0 (from default_settings) |
| `threshold_methods` | All methods, all threshold levels |
| `twogroup_methods` | Twogroup family only, threshold=2.0 |
| `heavy_vs_light` | Cox/logistic threshold methods only, all thresholds |
| `twogroup_vs_heavy` | Twogroup + cox_heavy, threshold=2.0 |

**Adding a new setting:**

```yaml
settings:
  my_setting:
    method_families: [twogroup, cox_heavy]
    thresholds: [1.0, 2.0, 3.0]
    max_fdp: 0.2
```

Then reference it in a supercollection's `plots:` list.

### `default_settings` keys

| Key | Effect |
|---|---|
| `thresholds` | List of threshold values to show; `null` = all |
| `method_families` | Restrict displayed methods to these families |
| `max_fdp` | FDP axis limit in power/FDP plots |
| `min_log_bf` | CS filtering threshold (log Bayes factor) |
| `max_cs_size` | CS filtering threshold (max CS size) |

---

## 4. Helper Scripts

| Script | When to use |
|---|---|
| `uv run python config.py` | After adding/changing simulations or methods — regenerates `results/manifest.json` |
| `scripts/symlink_plots.py` | After generating plots — creates `results/plots/{type}/{settings}/{sc}.pdf` symlinks for cross-supercollection browsing |
| `scripts/sync_plots.sh [host]` | Rsync `results/plots/` from Midway to local; dereferences remote symlinks so local gets real PDF copies |
| `scripts/invalidate_fits.py <prefix>` | Delete fit outputs for methods matching a name prefix, forcing Snakemake to rerun them. Use `--dry-run` first |

**Invalidate twogroup fits (dry run first):**

```bash
uv run python scripts/invalidate_fits.py twogroup --dry-run
uv run python scripts/invalidate_fits.py twogroup
```

---

## 5. Interactive Dashboard

```bash
uv run marimo run notebooks/dashboard.py
```

Select a supercollection and plot settings from the dropdowns. The dashboard
reads plot-ready parquet files from `results/collections/` and renders plots
live with the same code used to produce the PDFs.

---

## Pipeline Overview

```
config.py  →  results/manifest.json
                      │
          ┌───────────▼────────────┐
          │  by_batch/{h}/         │
          │    simulations.parquet │  ← materialize_twogroup_experiment_batch
          │    fits/{mh}/          │  ← fit_twogroup_experiment_batch_method
          │      fits.parquet      │
          └───────────┬────────────┘
                      │
          ┌───────────▼────────────┐
          │  collections/{alias}/  │
          │    plot_ready/*.parquet│  ← twogroup_experiments_target
          └───────────┬────────────┘
                      │
          ┌───────────▼────────────────────────────┐
          │  supercollections/{sc}/{type}/{s}.pdf  │  ← supercollection_plot
          └───────────┬────────────────────────────┘
                      │
          ┌───────────▼────────────────────────────┐
          │  plots/{type}/{settings}/{sc}.pdf      │  ← symlink_plots.py
          └────────────────────────────────────────┘
```
