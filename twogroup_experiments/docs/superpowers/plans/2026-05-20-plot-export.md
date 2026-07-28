# Plot Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate all dashboard plots as vector PDFs via Snakemake, driven by a restructured `plot_config.yaml` with separate `supercollections` and `settings` keys.

**Architecture:** Migrate `plot_config.yaml` to two top-level keys; create `generate_plots.py` that reuses all existing `viz_utils` render functions; add two Snakemake rules (`supercollection_plot`, `all_plots`); simplify dashboard config selection to two dropdowns (supercollection + plot_settings) and remove the Prepare Collections section.

**Tech Stack:** Python, matplotlib (PDF backend), Snakemake, Polars, existing `viz_utils.py` and `plot_ready.py`

---

## File Map

- **Modify:** `notebooks/plot_config.yaml` — migrate flat structure to `supercollections:` + `settings:` keys
- **Create:** `generate_plots.py` — `make_plot()` public API + 8 private dispatch functions + helpers
- **Create:** `tests/test_generate_plots.py` — unit tests for helpers and integration test for `make_plot()`
- **Modify:** `twogroup_experiments.snk` — update config loading, add wildcard constraint, add two rules
- **Modify:** `notebooks/dashboard.py` — remove 13 cells, add one `config_select_cell`

---

## Task 1: Migrate `notebooks/plot_config.yaml`

**Files:**
- Modify: `notebooks/plot_config.yaml`

The current file has a `_defaults` anchor and flat top-level keys (one per supercollection).
Migrate to two top-level keys: `supercollections:` and `settings:`.
Each supercollection gets `collections:` (unchanged) and `default_settings:` (was `settings:` under `<<: *defaults`).
The three initial `settings:` presets are added at the top level under `settings:`.
Remove the `_defaults` anchor entirely.

- [ ] **Step 1: Write the migrated YAML**

Replace the entire contents of `notebooks/plot_config.yaml` with the new format. The 22 existing supercollection blocks are preserved verbatim except `settings: {<<: *defaults}` becomes `default_settings:` with the defaults inlined. Three settings presets are added.

```yaml
supercollections:
  hallmark-signal-loc:
    collections:
      - {name: design=hallmark__enrichment=ser_enrich__signal=loc_0.50, alias: "mu=0.50"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=loc_1.00, alias: "mu=1.00"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=loc_1.50, alias: "mu=1.50"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=loc_2.00, alias: "mu=2.00"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=loc_2.50, alias: "mu=2.50"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=loc_3.00, alias: "mu=3.00"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  hallmark-signal-scale:
    collections:
      - {name: design=hallmark__enrichment=ser_enrich__signal=scale_0.75, alias: "sigma=0.75"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=scale_1.00, alias: "sigma=1.00"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=scale_1.50, alias: "sigma=1.50"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=scale_1.75, alias: "sigma=1.75"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=scale_2.00, alias: "sigma=2.00"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=scale_3.00, alias: "sigma=3.00"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=scale_4.00, alias: "sigma=4.00"}
      - {name: design=hallmark__enrichment=ser_enrich__signal=scale_5.00, alias: "sigma=5.00"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  c4-signal-loc:
    collections:
      - {name: design=c4__enrichment=ser_enrich__signal=loc_0.50, alias: "mu=0.50"}
      - {name: design=c4__enrichment=ser_enrich__signal=loc_1.00, alias: "mu=1.00"}
      - {name: design=c4__enrichment=ser_enrich__signal=loc_1.50, alias: "mu=1.50"}
      - {name: design=c4__enrichment=ser_enrich__signal=loc_2.00, alias: "mu=2.00"}
      - {name: design=c4__enrichment=ser_enrich__signal=loc_2.50, alias: "mu=2.50"}
      - {name: design=c4__enrichment=ser_enrich__signal=loc_3.00, alias: "mu=3.00"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  c4-signal-scale:
    collections:
      - {name: design=c4__enrichment=ser_enrich__signal=scale_0.75, alias: "sigma=0.75"}
      - {name: design=c4__enrichment=ser_enrich__signal=scale_1.00, alias: "sigma=1.00"}
      - {name: design=c4__enrichment=ser_enrich__signal=scale_1.50, alias: "sigma=1.50"}
      - {name: design=c4__enrichment=ser_enrich__signal=scale_1.75, alias: "sigma=1.75"}
      - {name: design=c4__enrichment=ser_enrich__signal=scale_2.00, alias: "sigma=2.00"}
      - {name: design=c4__enrichment=ser_enrich__signal=scale_3.00, alias: "sigma=3.00"}
      - {name: design=c4__enrichment=ser_enrich__signal=scale_4.00, alias: "sigma=4.00"}
      - {name: design=c4__enrichment=ser_enrich__signal=scale_5.00, alias: "sigma=5.00"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-rho0.9-signal-loc:
    collections:
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_0.50, alias: "mu=0.50"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.00, alias: "mu=1.00"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "mu=1.50"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "mu=2.00"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.50, alias: "mu=2.50"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_3.00, alias: "mu=3.00"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-rho0.9-signal-scale:
    collections:
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_0.75, alias: "sigma=0.75"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.00, alias: "sigma=1.00"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.50, alias: "sigma=1.50"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "sigma=1.75"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00, alias: "sigma=2.00"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_3.00, alias: "sigma=3.00"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_4.00, alias: "sigma=4.00"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_5.00, alias: "sigma=5.00"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-rho0.9-signal-loc:
    collections:
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_0.50, alias: "mu=0.50"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.00, alias: "mu=1.00"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "mu=1.50"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "mu=2.00"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.50, alias: "mu=2.50"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_3.00, alias: "mu=3.00"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-rho0.9-signal-scale:
    collections:
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_0.75, alias: "sigma=0.75"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.00, alias: "sigma=1.00"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.50, alias: "sigma=1.50"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "sigma=1.75"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00, alias: "sigma=2.00"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_3.00, alias: "sigma=3.00"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_4.00, alias: "sigma=4.00"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_5.00, alias: "sigma=5.00"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-correlation-loc:
    collections:
      - {name: design=gaussian_markov_rho_0.00_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.00"}
      - {name: design=gaussian_markov_rho_0.50_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.50"}
      - {name: design=gaussian_markov_rho_0.80_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.80"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.90"}
      - {name: design=gaussian_markov_rho_0.95_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.95"}
      - {name: design=gaussian_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.99"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-correlation-scale:
    collections:
      - {name: design=gaussian_markov_rho_0.00_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.00"}
      - {name: design=gaussian_markov_rho_0.50_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.50"}
      - {name: design=gaussian_markov_rho_0.80_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.80"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.90"}
      - {name: design=gaussian_markov_rho_0.95_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.95"}
      - {name: design=gaussian_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.99"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-correlation-loc:
    collections:
      - {name: design=uniform_markov_rho_0.00_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.00"}
      - {name: design=uniform_markov_rho_0.50_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.50"}
      - {name: design=uniform_markov_rho_0.80_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.80"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.90"}
      - {name: design=uniform_markov_rho_0.95_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.95"}
      - {name: design=uniform_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "rho=0.99"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-correlation-scale:
    collections:
      - {name: design=uniform_markov_rho_0.00_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.00"}
      - {name: design=uniform_markov_rho_0.50_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.50"}
      - {name: design=uniform_markov_rho_0.80_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.80"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.90"}
      - {name: design=uniform_markov_rho_0.95_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.95"}
      - {name: design=uniform_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "rho=0.99"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-n-features-loc:
    collections:
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "p=100"}
      - {name: design=gaussian_markov_rho_0.90_n_features_200__enrichment=ser_enrich__signal=loc_1.50, alias: "p=200"}
      - {name: design=gaussian_markov_rho_0.90_n_features_400__enrichment=ser_enrich__signal=loc_1.50, alias: "p=400"}
      - {name: design=gaussian_markov_rho_0.90_n_features_800__enrichment=ser_enrich__signal=loc_1.50, alias: "p=800"}
      - {name: design=gaussian_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=loc_1.50, alias: "p=1600"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-n-features-scale:
    collections:
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "p=100"}
      - {name: design=gaussian_markov_rho_0.90_n_features_200__enrichment=ser_enrich__signal=scale_1.75, alias: "p=200"}
      - {name: design=gaussian_markov_rho_0.90_n_features_400__enrichment=ser_enrich__signal=scale_1.75, alias: "p=400"}
      - {name: design=gaussian_markov_rho_0.90_n_features_800__enrichment=ser_enrich__signal=scale_1.75, alias: "p=800"}
      - {name: design=gaussian_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=scale_1.75, alias: "p=1600"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-n-features-loc:
    collections:
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.50, alias: "p=100"}
      - {name: design=uniform_markov_rho_0.90_n_features_200__enrichment=ser_enrich__signal=loc_1.50, alias: "p=200"}
      - {name: design=uniform_markov_rho_0.90_n_features_400__enrichment=ser_enrich__signal=loc_1.50, alias: "p=400"}
      - {name: design=uniform_markov_rho_0.90_n_features_800__enrichment=ser_enrich__signal=loc_1.50, alias: "p=800"}
      - {name: design=uniform_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=loc_1.50, alias: "p=1600"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-n-features-scale:
    collections:
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_1.75, alias: "p=100"}
      - {name: design=uniform_markov_rho_0.90_n_features_200__enrichment=ser_enrich__signal=scale_1.75, alias: "p=200"}
      - {name: design=uniform_markov_rho_0.90_n_features_400__enrichment=ser_enrich__signal=scale_1.75, alias: "p=400"}
      - {name: design=uniform_markov_rho_0.90_n_features_800__enrichment=ser_enrich__signal=scale_1.75, alias: "p=800"}
      - {name: design=uniform_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=scale_1.75, alias: "p=1600"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-correlation-loc-strong:
    collections:
      - {name: design=gaussian_markov_rho_0.00_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.00"}
      - {name: design=gaussian_markov_rho_0.50_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.50"}
      - {name: design=gaussian_markov_rho_0.80_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.80"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.90"}
      - {name: design=gaussian_markov_rho_0.95_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.95"}
      - {name: design=gaussian_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.99"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-correlation-scale-strong:
    collections:
      - {name: design=gaussian_markov_rho_0.00_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.00"}
      - {name: design=gaussian_markov_rho_0.50_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.50"}
      - {name: design=gaussian_markov_rho_0.80_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.80"}
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.90"}
      - {name: design=gaussian_markov_rho_0.95_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.95"}
      - {name: design=gaussian_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.99"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-correlation-loc-strong:
    collections:
      - {name: design=uniform_markov_rho_0.00_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.00"}
      - {name: design=uniform_markov_rho_0.50_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.50"}
      - {name: design=uniform_markov_rho_0.80_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.80"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.90"}
      - {name: design=uniform_markov_rho_0.95_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.95"}
      - {name: design=uniform_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "rho=0.99"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-correlation-scale-strong:
    collections:
      - {name: design=uniform_markov_rho_0.00_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.00"}
      - {name: design=uniform_markov_rho_0.50_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.50"}
      - {name: design=uniform_markov_rho_0.80_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.80"}
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.90"}
      - {name: design=uniform_markov_rho_0.95_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.95"}
      - {name: design=uniform_markov_rho_0.99_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "rho=0.99"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-n-features-loc-strong:
    collections:
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "p=100"}
      - {name: design=gaussian_markov_rho_0.90_n_features_200__enrichment=ser_enrich__signal=loc_2.00, alias: "p=200"}
      - {name: design=gaussian_markov_rho_0.90_n_features_400__enrichment=ser_enrich__signal=loc_2.00, alias: "p=400"}
      - {name: design=gaussian_markov_rho_0.90_n_features_800__enrichment=ser_enrich__signal=loc_2.00, alias: "p=800"}
      - {name: design=gaussian_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=loc_2.00, alias: "p=1600"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  gaussian-n-features-scale-strong:
    collections:
      - {name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "p=100"}
      - {name: design=gaussian_markov_rho_0.90_n_features_200__enrichment=ser_enrich__signal=scale_2.25, alias: "p=200"}
      - {name: design=gaussian_markov_rho_0.90_n_features_400__enrichment=ser_enrich__signal=scale_2.25, alias: "p=400"}
      - {name: design=gaussian_markov_rho_0.90_n_features_800__enrichment=ser_enrich__signal=scale_2.25, alias: "p=800"}
      - {name: design=gaussian_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=scale_2.25, alias: "p=1600"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-n-features-loc-strong:
    collections:
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00, alias: "p=100"}
      - {name: design=uniform_markov_rho_0.90_n_features_200__enrichment=ser_enrich__signal=loc_2.00, alias: "p=200"}
      - {name: design=uniform_markov_rho_0.90_n_features_400__enrichment=ser_enrich__signal=loc_2.00, alias: "p=400"}
      - {name: design=uniform_markov_rho_0.90_n_features_800__enrichment=ser_enrich__signal=loc_2.00, alias: "p=800"}
      - {name: design=uniform_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=loc_2.00, alias: "p=1600"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      L: 1
      max_fdp: 0.5
      method_families:
        - cox_heavy
        - logistic_oracle
        - logistic_threshold
        - cox_light_threshold
        - twogroup
        - twogroup_oracle

  uniform-n-features-scale-strong:
    collections:
      - {name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.25, alias: "p=100"}
      - {name: design=uniform_markov_rho_0.90_n_features_200__enrichment=ser_enrich__signal=scale_2.25, alias: "p=200"}
      - {name: design=uniform_markov_rho_0.90_n_features_400__enrichment=ser_enrich__signal=scale_2.25, alias: "p=400"}
      - {name: design=uniform_markov_rho_0.90_n_features_800__enrichment=ser_enrich__signal=scale_2.25, alias: "p=800"}
      - {name: design=uniform_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=scale_2.25, alias: "p=1600"}
    default_settings:
      threshold: 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
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
    method_families:
      - logistic_threshold
      - cox_light_threshold
  cox_heavy_vs_twogroup:
    method_families:
      - cox_heavy
      - twogroup
```

- [ ] **Step 2: Verify the YAML loads cleanly**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run python -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('notebooks/plot_config.yaml').read_text())
assert 'supercollections' in cfg, 'missing supercollections key'
assert 'settings' in cfg, 'missing settings key'
assert len(cfg['supercollections']) == 22, f'expected 22 supercollections, got {len(cfg[\"supercollections\"])}'
assert len(cfg['settings']) == 3, f'expected 3 settings presets, got {len(cfg[\"settings\"])}'
sc = cfg['supercollections']['hallmark-signal-loc']
assert 'collections' in sc
assert 'default_settings' in sc
assert sc['default_settings']['threshold'] == 2.0
print('OK: plot_config.yaml structure valid')
"
```

Expected: `OK: plot_config.yaml structure valid`

- [ ] **Step 3: Commit**

```bash
git add notebooks/plot_config.yaml
git commit -m "feat: migrate plot_config.yaml to supercollections/settings format"
```

---

## Task 2: Create `generate_plots.py`

**Files:**
- Create: `generate_plots.py`
- Create: `tests/test_generate_plots.py`

`generate_plots.py` is a top-level module (same directory as `viz_utils.py`). It reuses all existing `viz_utils` expand/make/render functions — no new rendering logic.

The module needs access to `plot_ready` and `viz_utils`. It derives `foreground_methods` from settings the same way `selected_methods_cell` does in the dashboard.

- [ ] **Step 1: Write failing tests**

Create `tests/test_generate_plots.py`:

```python
from __future__ import annotations

import pytest
import polars as pl


def test_resolve_settings_merges_overrides():
    import generate_plots

    cfg = {
        "supercollections": {
            "my-sc": {
                "default_settings": {
                    "threshold": 2.0,
                    "L": 1,
                    "max_fdp": 0.5,
                    "method_families": ["twogroup"],
                }
            }
        },
        "settings": {
            "cox_only": {"method_families": ["cox_heavy"]},
        },
    }

    result = generate_plots._resolve_settings(cfg, "my-sc", "cox_only")

    assert result["threshold"] == 2.0
    assert result["method_families"] == ["cox_heavy"]
    assert result["max_fdp"] == 0.5


def test_resolve_settings_default_only():
    import generate_plots

    cfg = {
        "supercollections": {
            "my-sc": {
                "default_settings": {"threshold": 2.0, "L": 1}
            }
        },
        "settings": {
            "all_methods": {},
        },
    }

    result = generate_plots._resolve_settings(cfg, "my-sc", "all_methods")

    assert result == {"threshold": 2.0, "L": 1}


def test_load_plot_config_has_two_keys():
    import generate_plots

    cfg = generate_plots._load_plot_config()

    assert "supercollections" in cfg
    assert "settings" in cfg


def test_foreground_methods_filters_by_family_and_L():
    import generate_plots

    method_metadata = pl.DataFrame({
        "method": ["twogroup_L1", "cox_heavy_L1", "twogroup_L2"],
        "method_family": ["twogroup", "cox_heavy", "twogroup"],
        "L": [1, 1, 2],
        "threshold": [None, None, None],
        "is_thresholded": [False, False, False],
    })

    settings = {"method_families": ["twogroup"], "L": 1, "threshold": 2.0}
    result = generate_plots._foreground_methods(method_metadata, settings)

    assert result == {"twogroup_L1"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_generate_plots.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'generate_plots'`

- [ ] **Step 3: Write `generate_plots.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import plot_ready
import viz_utils


_PLOT_CONFIG_PATH = Path(__file__).parent / "notebooks" / "plot_config.yaml"
_COLLECTION_ALIAS_ROOT = Path(__file__).parent / "results" / "collections"

_PLOT_TYPES = [
    "pip_calibration", "power_fdp", "causal_pip", "causal_rank",
    "mass_above_causal", "cs_dot_summary", "cs_power_fdp", "cs_beta_trace",
]


def make_plot(
    supercollection: str,
    plot_settings: str,
    plot_type: str,
    output_path: str,
) -> None:
    """Generate one plot-type PDF for a (supercollection, plot_settings) combo."""
    cfg = _load_plot_config()
    settings = _resolve_settings(cfg, supercollection, plot_settings)
    combined_data = _load_supercollection_data(cfg, supercollection)
    fig = _PLOT_DISPATCH[plot_type](combined_data, settings)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_plot_config() -> dict:
    return yaml.safe_load(_PLOT_CONFIG_PATH.read_text()) or {}


def _resolve_settings(cfg: dict, supercollection: str, plot_settings: str) -> dict:
    defaults = cfg["supercollections"][supercollection].get("default_settings", {})
    overrides = cfg["settings"].get(plot_settings, {})
    return {**defaults, **overrides}


def _load_supercollection_data(cfg: dict, supercollection: str) -> dict:
    coll_list = cfg["supercollections"][supercollection]["collections"]
    aliases = {item["name"]: item.get("alias", item["name"]) for item in coll_list}
    bundles = {
        item["name"]: plot_ready.load_plot_ready_collection(
            _COLLECTION_ALIAS_ROOT / item["name"]
        )
        for item in coll_list
    }
    combined_method_metadata = (
        pl.concat([b["method_metadata"] for b in bundles.values()])
        .unique(subset=["method", "threshold"])
    )

    def _tag(key: str) -> pl.DataFrame:
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


def _foreground_methods(method_metadata: pl.DataFrame, settings: dict) -> set[str]:
    threshold = settings.get("threshold", 2.0)
    L = settings.get("L", 1)
    method_families = settings.get("method_families", [])
    mask = (
        pl.col("method_family").is_in(method_families)
        & (pl.col("L") == L)
        & (
            ~pl.col("is_thresholded")
            | (pl.col("threshold") == threshold)
            | pl.col("threshold").is_null()
        )
    )
    return set(method_metadata.filter(mask)["method"].to_list())


def _method_order(method_metadata: pl.DataFrame, foreground: set[str]) -> list[str]:
    return (
        method_metadata.filter(pl.col("method").is_in(foreground))
        .select("method", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method"])["method"]
        .to_list()
    )


def _make_pip_calibration(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    threshold = settings.get("threshold", 2.0)
    fg = _foreground_methods(method_meta, settings)
    summary = viz_utils.expand_pip_calibration_from_compact(
        pip_plot.filter(pl.col("method").is_in(fg)),
        method_meta,
        selected_threshold=threshold,
    )
    if summary.is_empty():
        return viz_utils.make_placeholder_chart("No PIP calibration data")
    return viz_utils.render_pip_calibration(
        summary,
        facet_by_simulation=True,
        collection_names=combined_data["collection_names"],
    )


def _make_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    threshold = settings.get("threshold", 2.0)
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
        selected_threshold=threshold,
        show_background_threshold_traces=False,
    )
    if power_fdp.is_empty():
        return viz_utils.make_placeholder_chart("No power/FDP data")
    summary = viz_utils.make_power_fdp_summary(power_fdp)
    return viz_utils.render_power_fdp_chart(
        summary,
        facet=True,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
        collection_names=combined_data["collection_names"],
    )


def _make_causal_pip(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    causal_pip = viz_utils.expand_causal_pip_from_compact(pip_plot, method_meta)
    filtered = causal_pip.filter(pl.col("method").is_in(fg))
    if filtered.is_empty():
        return viz_utils.make_placeholder_chart("No causal PIP data")
    order = _method_order(method_meta, fg)
    summary = viz_utils.make_causal_pip_summary(filtered)
    return viz_utils.render_causal_pip_chart(
        summary,
        facet=True,
        legend_outside=True,
        square_axes=True,
        method_order=order,
        collection_names=combined_data["collection_names"],
    )


def _make_causal_rank(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = _method_order(method_meta, fg)
    rank_summary = viz_utils.make_causal_rank_summary(cs_data, method_meta, selected_methods=fg)
    if rank_summary.is_empty():
        return viz_utils.make_placeholder_chart("No causal rank data")
    return viz_utils.render_causal_rank_chart(
        rank_summary,
        facet=True,
        legend_outside=True,
        square_axes=True,
        method_order=order,
        collection_names=combined_data["collection_names"],
    )


def _make_mass_above_causal(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = _method_order(method_meta, fg)
    expanded = viz_utils.expand_mass_above_causal_from_compact(
        cs_data.filter(pl.col("method").is_in(fg)),
        method_meta,
    )
    if expanded.is_empty():
        return viz_utils.make_placeholder_chart("No mass above causal data")
    summary = viz_utils.make_mass_above_causal_summary(expanded)
    return viz_utils.render_mass_above_causal_chart(
        summary,
        facet=True,
        legend_outside=True,
        square_axes=True,
        method_order=order,
        collection_names=combined_data["collection_names"],
    )


def _make_cs_dot_summary(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    threshold = settings.get("threshold", 2.0)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    cs_beta = settings.get("cs_beta", 0.95)
    collection_names = combined_data["collection_names"]
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_threshold=threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_dot_summary_chart(
        summary,
        collection_names=collection_names,
        selected_beta=round(cs_beta, 2),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_cs_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    _BETA_095_IDX = 45  # CS_BETA_GRID[45] == 0.95
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    threshold = settings.get("threshold", 2.0)
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)

    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")

    cs_raw = cs_data.with_columns(
        pl.col("cs_sizes").list.get(_BETA_095_IDX).alias("cs_size"),
        pl.when(pl.col("rank_of_causal").list.len() > 0)
        .then(pl.col("rank_of_causal").list.min() < pl.col("cs_sizes").list.get(_BETA_095_IDX))
        .otherwise(False)
        .alias("causal_in_cs"),
    ).select(
        "collection_name", "sample_id", "method", "threshold",
        "l", "cs_size", "causal_in_cs", "ser_log_bf",
    )

    raw = (
        cs_raw.filter(
            pl.col("method").is_in(fg)
            & (pl.col("threshold").is_null() | (pl.col("threshold") == threshold))
        )
        .join(
            method_meta.select("method", "threshold", "method_display", "is_thresholded"),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
    )

    if raw.is_empty():
        return viz_utils.make_placeholder_chart("No CS power/FDP data")

    lbf_lo = float(raw["ser_log_bf"].min())
    lbf_hi = float(raw["ser_log_bf"].max())
    lbf_grid = np.linspace(lbf_lo, lbf_hi, 60)[::-1]
    method_groups = (
        raw.select("method", "threshold", "method_display", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method_display"])
    )

    rows = []
    for coll_name in collection_names:
        coll_raw = raw.filter(pl.col("collection_name") == coll_name)
        for mg in method_groups.iter_rows(named=True):
            thresh_filter = (
                pl.col("threshold").is_null()
                if mg["threshold"] is None
                else (pl.col("threshold") == mg["threshold"])
            )
            m_data = coll_raw.filter((pl.col("method") == mg["method"]) & thresh_filter)
            if m_data.is_empty():
                continue
            n_total = m_data.height
            causal_arr = m_data["causal_in_cs"].to_numpy()
            lbf_arr = m_data["ser_log_bf"].to_numpy()
            for t in lbf_grid:
                disc = lbf_arr >= t
                hit = disc & causal_arr
                n_disc = int(disc.sum())
                n_hit = int(hit.sum())
                rows.append({
                    "collection_name": coll_name,
                    "method": mg["method"],
                    "threshold": mg["threshold"],
                    "method_display": mg["method_display"],
                    "is_thresholded": mg["is_thresholded"],
                    "pip_threshold": float(t),
                    "power": float(n_hit / max(n_total, 1)),
                    "fdp": float((n_disc - n_hit) / max(n_disc, 1)),
                })

    cs_pf = pl.from_dicts(
        rows,
        schema={
            "collection_name": pl.String,
            "method": pl.String,
            "threshold": pl.Float64,
            "method_display": pl.String,
            "is_thresholded": pl.Boolean,
            "pip_threshold": pl.Float64,
            "power": pl.Float64,
            "fdp": pl.Float64,
        },
    ).with_columns(
        pl.col("method_display").alias("trace_label"),
        pl.col("method_display").alias("legend_label"),
        pl.lit(True).alias("is_selected_threshold"),
        pl.col("collection_name").alias("simulation_name"),
    )

    return viz_utils.render_power_fdp_chart(
        cs_pf,
        facet=True,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
        collection_names=collection_names,
    )


def _make_cs_beta_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    threshold = settings.get("threshold", 2.0)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_threshold=threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_beta_trace_chart(
        beta_summary,
        collection_names=collection_names,
        selected_threshold=threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


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

- [ ] **Step 4: Run unit tests to verify they pass**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_generate_plots.py -v
```

Expected: all 4 unit tests pass.

- [ ] **Step 5: Smoke-test `make_plot()` end-to-end**

This requires plot-ready data on disk. Skip if collections are not yet materialized (Task 3 snakemake run would materialize them). If data exists, run:

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run python -c "
import generate_plots, tempfile, os
from pathlib import Path

# Use first available supercollection
cfg = generate_plots._load_plot_config()
sc = next(iter(cfg['supercollections']))
ps = next(iter(cfg['settings']))
coll_names = [item['name'] for item in cfg['supercollections'][sc]['collections']]
# Check at least one collection is plot-ready
alias_root = Path('results/collections')
ready = [n for n in coll_names if (alias_root / n / 'plot_ready' / 'out.txt').exists()]
if not ready:
    print(f'SKIP: no plot-ready collections in {sc}')
else:
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        out = f.name
    generate_plots.make_plot(sc, ps, 'power_fdp', out)
    size = os.path.getsize(out)
    os.unlink(out)
    print(f'OK: PDF written, {size} bytes')
"
```

Expected: `OK: PDF written, <N> bytes` or `SKIP: no plot-ready collections in <name>`.

- [ ] **Step 6: Commit**

```bash
git add generate_plots.py tests/test_generate_plots.py
git commit -m "feat: add generate_plots module with make_plot() dispatch"
```

---

## Task 3: Update `twogroup_experiments.snk`

**Files:**
- Modify: `twogroup_experiments.snk`

Three changes: (1) update `SUPERCOLLECTIONS` and `load_supercollection` to read from `supercollections:` key, (2) add `PLOT_TYPES`, `PLOT_SETTINGS_NAMES` constants and `plot_settings` wildcard constraint, (3) add `rule supercollection_plot` and `rule all_plots`.

Current code (lines 55–72) reads `_PLOT_CONFIG` and builds `SUPERCOLLECTIONS` as keys excluding `_`-prefixed ones. This needs to change to `_PLOT_CONFIG.get("supercollections", {}).keys()`.

- [ ] **Step 1: Update config loading and constants**

In `twogroup_experiments.snk`, replace:

```python
_PLOT_CONFIG: dict = _yaml.safe_load(Path(PLOT_CONFIG_PATH).read_text()) or {}
SUPERCOLLECTIONS = sorted(
    key for key in _PLOT_CONFIG if not str(key).startswith("_")
)
```

with:

```python
_PLOT_CONFIG: dict = _yaml.safe_load(Path(PLOT_CONFIG_PATH).read_text()) or {}
SUPERCOLLECTIONS = sorted(_PLOT_CONFIG.get("supercollections", {}).keys())
PLOT_SETTINGS_NAMES = sorted(_PLOT_CONFIG.get("settings", {}).keys())
PLOT_TYPES = [
    "pip_calibration", "power_fdp", "causal_pip", "causal_rank",
    "mass_above_causal", "cs_dot_summary", "cs_power_fdp", "cs_beta_trace",
]
```

- [ ] **Step 2: Update `load_supercollection`**

Replace:

```python
def load_supercollection(supercollection: str) -> dict:
    if supercollection not in _PLOT_CONFIG:
        raise KeyError(supercollection)
    return _PLOT_CONFIG[supercollection]
```

with:

```python
def load_supercollection(supercollection: str) -> dict:
    supercollections = _PLOT_CONFIG.get("supercollections", {})
    if supercollection not in supercollections:
        raise KeyError(supercollection)
    return supercollections[supercollection]
```

- [ ] **Step 3: Add `plot_settings` wildcard constraint**

In the existing `wildcard_constraints:` block, add:

```python
    plot_settings    = r"[A-Za-z0-9_\-]+",
```

The full block becomes:

```python
wildcard_constraints:
    batch_hash       = r"[0-9a-f]{64}",
    method_hash      = r"[0-9a-f]{64}",
    collection_alias = r"[A-Za-z0-9_\-=\.]+",
    supercollection  = r"[A-Za-z0-9_\-\.]+",
    plot_settings    = r"[A-Za-z0-9_\-]+",
```

- [ ] **Step 4: Add `rule supercollection_plot` and `rule all_plots`**

Append after `rule twogroup_experiments_target` (end of file):

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
            supercollection=SUPERCOLLECTIONS,
            plot_settings=PLOT_SETTINGS_NAMES,
            plot_type=PLOT_TYPES,
        ),
```

- [ ] **Step 5: Verify dry-run for all_plots**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run snakemake --snakefile twogroup_experiments.snk --dry-run all_plots 2>&1 | tail -20
```

Expected: Snakemake prints a job plan (or "Nothing to be done" if all PDFs already exist). No `KeyError` or `WildcardError`.

- [ ] **Step 6: Verify `rule all_supercollections` still works (existing test)**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run snakemake --snakefile twogroup_experiments.snk --dry-run all_supercollections 2>&1 | tail -10
```

Expected: No errors.

- [ ] **Step 7: Commit**

```bash
git add twogroup_experiments.snk
git commit -m "feat: add Snakemake supercollection_plot and all_plots rules"
```

---

## Task 4: Simplify `notebooks/dashboard.py`

**Files:**
- Modify: `notebooks/dashboard.py`
- Test: `tests/test_plot_ready.py` (existing `test_dashboard_notebook_module_loads`)

Two sets of changes: (1) delete the Prepare Collections section and all old config selection cells, (2) replace them with a new `config_select_cell` that has two dropdowns.

**Cells to delete entirely:**
- `prepare_heading_cell` (lines 29–33)
- `unprepared_cell` (lines 37–63)
- `snakemake_cores_cell` (lines 66–69)
- `snakemake_prepare_cell` (lines 72–137)
- `dry_run_output_cell` (lines 140–142)
- `config_io_cell` (lines 153–180)
- `collection_selector_cell` (lines 183–192)
- `dirty_state_cell` (lines 195–198)
- `config_state_cell` (lines 201–204)
- `alias_cell` (lines 207–283)
- `apply_status_cell` (lines 286–289)
- `config_save_cell` (lines 293–328)
- `config_save_status_cell` (lines 331–337)

**Cell to add** (replaces all of the above after `view_heading_cell`):

```python
@app.cell
def config_select_cell():
    import yaml as _yaml

    _config_path = Path(__file__).parent / "plot_config.yaml"
    _all_configs: dict = _yaml.safe_load(_config_path.read_text()) or {} if _config_path.exists() else {}
    _supercollections = _all_configs.get("supercollections", {})
    _settings_presets = _all_configs.get("settings", {})

    _sc_names = list(_supercollections.keys())
    _ps_names = ["(default)"] + list(_settings_presets.keys())

    supercollection_dropdown = mo.ui.dropdown(
        options=_sc_names,
        value=_sc_names[0] if _sc_names else None,
        label="supercollection",
    )
    plot_settings_dropdown = mo.ui.dropdown(
        options=_ps_names,
        value=_ps_names[0] if _ps_names else None,
        label="plot settings",
    )

    def _apply(_):
        sc = supercollection_dropdown.value
        ps = plot_settings_dropdown.value
        if not sc:
            return {"selected": [], "aliases": {}, "settings": {}}
        sc_cfg = _supercollections.get(sc, {})
        coll_list = sc_cfg.get("collections", [])
        selected = [item["name"] for item in coll_list]
        aliases = {item["name"]: item.get("alias", item["name"]) for item in coll_list}
        defaults = sc_cfg.get("default_settings", {})
        overrides = _settings_presets.get(ps, {}) if ps != "(default)" else {}
        settings = {**defaults, **overrides}
        return {"selected": selected, "aliases": aliases, "settings": settings}

    _initial_val = _apply(None)
    apply_btn = mo.ui.button(label="Apply", on_click=_apply, value=_initial_val)

    mo.hstack([supercollection_dropdown, plot_settings_dropdown, apply_btn])
    return (apply_btn,)
```

**`bundles_cell` depends on `apply_btn` from `config_select_cell`** — no change to `bundles_cell` itself needed, but its signature must reference `apply_btn` which was previously provided by `alias_cell`. Marimo resolves by function argument name, so `bundles_cell(collection_alias_root, apply_btn)` keeps working as long as `apply_btn` is still exported from a cell.

The `controls_cell` and `histogram_controls_cell` both read `apply_btn.value.get("settings", {})` — this still works because `apply_btn.value["settings"]` is the merged settings dict.

- [ ] **Step 1: Verify existing dashboard test passes before editing**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_plot_ready.py::test_dashboard_notebook_module_loads -v
```

Expected: PASS.

- [ ] **Step 2: Delete the 13 cells from `dashboard.py`**

Delete these cells (identified by their `def <name>` decorators):
- `prepare_heading_cell`
- `unprepared_cell`
- `snakemake_cores_cell`
- `snakemake_prepare_cell`
- `dry_run_output_cell`
- `config_io_cell`
- `collection_selector_cell`
- `dirty_state_cell`
- `config_state_cell`
- `alias_cell`
- `apply_status_cell`
- `config_save_cell`
- `config_save_status_cell`

Delete from line 29 (`@app.cell(hide_code=True)` before `prepare_heading_cell`) through line 337 (end of `config_save_status_cell`), inclusive.

- [ ] **Step 3: Insert `config_select_cell` after `view_heading_cell`**

`view_heading_cell` (lines 145–149 in the original file, will be renumbered after deletions) outputs a heading "## View Collections". Insert `config_select_cell` immediately after it.

The new cell:

```python
@app.cell
def config_select_cell():
    import yaml as _yaml

    _config_path = Path(__file__).parent / "plot_config.yaml"
    _all_configs: dict = _yaml.safe_load(_config_path.read_text()) or {} if _config_path.exists() else {}
    _supercollections = _all_configs.get("supercollections", {})
    _settings_presets = _all_configs.get("settings", {})

    _sc_names = list(_supercollections.keys())
    _ps_names = ["(default)"] + list(_settings_presets.keys())

    supercollection_dropdown = mo.ui.dropdown(
        options=_sc_names,
        value=_sc_names[0] if _sc_names else None,
        label="supercollection",
    )
    plot_settings_dropdown = mo.ui.dropdown(
        options=_ps_names,
        value=_ps_names[0] if _ps_names else None,
        label="plot settings",
    )

    def _apply(_):
        sc = supercollection_dropdown.value
        ps = plot_settings_dropdown.value
        if not sc:
            return {"selected": [], "aliases": {}, "settings": {}}
        sc_cfg = _supercollections.get(sc, {})
        coll_list = sc_cfg.get("collections", [])
        selected = [item["name"] for item in coll_list]
        aliases = {item["name"]: item.get("alias", item["name"]) for item in coll_list}
        defaults = sc_cfg.get("default_settings", {})
        overrides = _settings_presets.get(ps, {}) if ps != "(default)" else {}
        settings = {**defaults, **overrides}
        return {"selected": selected, "aliases": aliases, "settings": settings}

    _initial_val = _apply(None)
    apply_btn = mo.ui.button(label="Apply", on_click=_apply, value=_initial_val)

    mo.hstack([supercollection_dropdown, plot_settings_dropdown, apply_btn])
    return (apply_btn,)
```

- [ ] **Step 4: Update `bundles_cell` signature if needed**

Check the signature of `bundles_cell`. In the original it is:

```python
def bundles_cell(collection_alias_root, apply_btn):
```

`collection_alias_root` was exported by `collection_selector_cell` which is now deleted. It needs to be defined locally instead. Update `bundles_cell`:

```python
@app.cell
def bundles_cell(apply_btn):
    collection_alias_root = Path(__file__).parent.parent / "results" / "collections"
    _settings = apply_btn.value
    _selected = _settings["selected"]
    _aliases: dict[str, str] = _settings["aliases"]

    mo.stop(not _selected, mo.md("Select a supercollection above."))

    _bundles = {
        name: plot_ready.load_plot_ready_collection(collection_alias_root / name)
        for name in _selected
    }

    combined_method_metadata = (
        pl.concat([b["method_metadata"] for b in _bundles.values()])
        .unique(subset=["method", "threshold"])
    )

    def _tag(key):
        return pl.concat([
            b[key].with_columns(
                pl.lit(_aliases.get(name, name)).alias("collection_name")
            )
            for name, b in _bundles.items()
        ])

    combined_data = {
        "method_metadata": combined_method_metadata,
        "collection_names": [_aliases.get(n, n) for n in _selected],
        "pip_plot_data": _tag("pip_plot_data"),
        "cs_plot_data": _tag("cs_plot_data"),
    }
    return (combined_data,)
```

- [ ] **Step 5: Run dashboard module load test**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/test_plot_ready.py::test_dashboard_notebook_module_loads -v
```

Expected: PASS.

- [ ] **Step 6: Run full test suite**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
uv run pytest tests/ -v 2>&1 | tail -20
```

Expected: all previously-passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add notebooks/dashboard.py
git commit -m "feat: simplify dashboard to supercollection/plot_settings dropdowns, remove prepare-collections"
```
