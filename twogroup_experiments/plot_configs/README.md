# Plot Configuration Guide

All `*.yaml` files in this directory are merged at load time. Keys are merged shallowly (later files alphabetically win on conflicts within the same top-level key). Add a new file here to define new analyses without touching existing configs.

## File structure

```
plot_configs/
  main.yaml       # shared: method_collections, plot_type_groups, settings, bulk supercollections
  non_null.yaml   # coverage-size diagnostic supercollections
  your_new.yaml   # add new analyses here
```

## Top-level keys

| Key | Purpose |
|-----|---------|
| `method_collections` | Named groups of methods (used by Snakemake to determine which fits to run per collection) |
| `collections` | Per-collection overrides: custom `simulations` list or non-default `method_collections`. Omit if the collection name matches a single simulation spec and uses the default method set. |
| `supercollections` | Named groups of collections for plotting. This is what you define to add a new analysis. |
| `plot_type_groups` | Named lists of plot types, referenced from supercollection `plots:` blocks. |
| `settings` | Named setting presets (thresholds, min_log_bf, etc.) referenced by name from supercollection `plots:`. |

## Defining a supercollection

```yaml
supercollections:
  my-analysis:
    collections:
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_2.00
        alias: hallmark loc=2
      - name: design=c4__enrichment=ser_enrich__signal=loc_2.00
        alias: c4 loc=2
    default_settings:         # applied unless overridden by a named settings preset
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
    plots:
      - settings:             # list of named settings presets from settings: section
          - twogroup_methods
          - minimal
        plot_type_groups:     # list of named groups from plot_type_groups: section
          - cs
          - standard
```

- `name` must match a simulation spec name (check `config.py` `SIMULATION_BY_NAME` keys).
- `alias` is the display label in plots. Defaults to `name` if omitted.
- Multiple `plots:` entries allowed — each produces a separate set of output PDFs.

## Collection name format

```
design={design}__enrichment={enrichment}__signal={kind}_{value}
```

Common designs: `hallmark`, `c4`, `gaussian_markov_rho_0.90_n_features_100`, `uniform_markov_rho_0.90_n_features_100`

Enrichment: `ser_enrich` (non-null SER), `b0_{b0}_b_{b}` (grid, e.g. `b0_-3.00_b_0.00` for null)

Signal: `loc_{mu}` or `scale_{sigma}`

## Available plot_type_groups

| Group | Contents |
|-------|---------|
| `standard` | pip_calibration, power_fdp, causal_pip, causal_rank, mass_above_causal (+ agg_ variants) |
| `cs` | All CS plots: dot_summary, calibrated_dot, size_power, power_fdp, beta_trace, coverage_trace, coverage_size (+ agg_ variants) |
| `f1` | f1_boxplot, f1_scatter, f1_enrich_scatter |
| `coverage_size` | cs_coverage_size, agg_cs_coverage_size |

## YAML anchors

Anchors (`&name`) and aliases (`*name`) work within a single file but not across files. Define and reference them in the same file.

## Example: new analysis file

```yaml
# notebooks/plot_configs/my_analysis.yaml
supercollections:
  my-new-supercollection:
    collections:
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_2.00
        alias: loc=2
    default_settings:
      min_log_bf: 2.0
      max_cs_size: 10000
    plots:
      - settings:
          - minimal
        plot_type_groups:
          - coverage_size
```

Save the file — it's picked up automatically on the next run. No changes to existing files needed.
