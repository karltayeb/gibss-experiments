# Simulation Config Reorganization Design

## Goal

Reorganize `config.py` around the simulation comparisons intended for the paper while keeping the underlying simulation/method registry broad enough to support future collections without further structural changes.

The design separates:

- a dense bank of registered simulation specs
- atomic collections that each correspond to a single plotted comparison entry
- grouped plot configurations in `notebooks/plot_config.yaml`

## Scope

This reorganization covers three study sections:

- signal simulations
- correlation simulations
- feature-count simulations

For now, all paper-facing collections use:

- `enrichment=ser_enrich`
- batch size `50`

## Simulation Bank

The raw simulation bank should be larger than the set of paper-facing collections.

### Global parameter grids

- `LOC_GRID = (0.25, 0.50, ..., 3.50)`
- `SCALE_GRID = (0.50, 0.75, ..., 5.00)`
- `RHO_GRID = (0.00, 0.10, ..., 0.90, 0.91, ..., 0.99)`
- `N_FEATURE_GRID = (100, 200, 400, 800, 1600)`

### Signal family conventions

- location-family simulations use `f1 = N(mu0, 0.1)`
- scale-family simulations use `f1 = N(0, sigma0)`

### Fixed anchors

These anchors are used for the correlation and feature-count studies:

- location anchor: `f1 = N(1.5, 0.1)`
- scale anchor: `f1 = N(0, 1.75)`

### Design families in the bank

- `hallmark`
- `c4`
- Gaussian Markov designs over `RHO_GRID`
- uniform Markov designs over `RHO_GRID`
- Gaussian Markov feature-count designs over `N_FEATURE_GRID` at fixed `rho=0.90`
- uniform Markov feature-count designs over `N_FEATURE_GRID` at fixed `rho=0.90`

## Atomic Collection Naming

Atomic collection names should be flat and self-describing.

Top-level fields are separated by `__` and assigned with `=`.

### Signal and correlation studies

- `design={design}__enrichment={enrichment}__signal=loc_{value:.2f}`
- `design={design}__enrichment={enrichment}__signal=scale_{value:.2f}`

Examples:

- `design=hallmark__enrichment=ser_enrich__signal=loc_1.50`
- `design=c4__enrichment=ser_enrich__signal=scale_1.75`
- `design=gaussian_markov_rho_0.90__enrichment=ser_enrich__signal=loc_2.00`
- `design=uniform_markov_rho_0.95__enrichment=ser_enrich__signal=scale_1.75`

### Feature-count study

- `design={design}_n_features_{p}__enrichment={enrichment}__signal=loc_{value:.2f}`
- `design={design}_n_features_{p}__enrichment={enrichment}__signal=scale_{value:.2f}`

Examples:

- `design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_1.50`
- `design=uniform_markov_rho_0.90_n_features_1600__enrichment=ser_enrich__signal=scale_1.75`

Names are intended to be human-readable and grep-friendly. The full structured meaning lives in metadata, not in parsers built around the names.

## Atomic Collections

Each atomic collection corresponds to one plotted comparison entry. A collection contains:

- one simulation spec
- the default method set for the relevant SER or SuSiE fit family
- all batches for that simulation under the standardized batch-size policy

## Paper-Facing Supercollections

These grouped comparisons should be expressed in `notebooks/plot_config.yaml`, not as separate config-layer collection unions unless later needed for workflow convenience.

### Signal

- `hallmark-signal-loc`
- `hallmark-signal-scale`
- `c4-signal-loc`
- `c4-signal-scale`
- `gaussian-rho0.9-signal-loc`
- `gaussian-rho0.9-signal-scale`
- `uniform-rho0.9-signal-loc`
- `uniform-rho0.9-signal-scale`

The atomic collections inside each group vary over the selected subset of `LOC_GRID` or `SCALE_GRID`.

The signal section uses the following designs:

- `hallmark`
- `c4`
- `gaussian_markov_rho_0.90`
- `uniform_markov_rho_0.90`

### Correlation

- `gaussian-correlation-loc`
- `gaussian-correlation-scale`
- `uniform-correlation-loc`
- `uniform-correlation-scale`

The atomic collections inside each group vary over the selected subset of `RHO_GRID`.

The location group uses the fixed anchor `loc_1.50`. The scale group uses the fixed anchor `scale_1.75`.

### Feature Count

- `gaussian-n-features-loc`
- `gaussian-n-features-scale`
- `uniform-n-features-loc`
- `uniform-n-features-scale`

The atomic collections inside each group vary over `N_FEATURE_GRID`.

These use fixed:

- `rho=0.90`
- location anchor `loc_1.50`
- scale anchor `scale_1.75`

## Collection Subsets Used for Plots

The simulation bank is dense; the paper-facing collection groups should use smaller readable subsets.

The selected subsets should be defined explicitly near the top of `config.py` as globals, separate from the dense bank grids.

Expected globals:

- `SIGNAL_LOC_VALUES`
- `SIGNAL_SCALE_VALUES`
- `CORRELATION_RHO_VALUES`
- `N_FEATURE_VALUES`

These subsets are the values actually materialized as atomic paper-facing collections.

## Config Structure

`config.py` should be reorganized into four layers:

1. shared globals
2. simulation-bank builders
3. atomic paper-collection registration helpers
4. grouped plot-facing metadata export support

Recommended helper responsibilities:

- build dense design kwargs
- build dense `f1` kwargs
- register raw simulation specs once
- register atomic collections from explicit study definitions
- expose a predictable set of collection names for `plot_config.yaml`

## Registry Behavior

The current registry deduplication behavior should be preserved. Re-registering the same simulation or method spec from multiple study definitions must remain harmless and should not create duplicate registry entries.

## Out of Scope

- adding `ser_dep` paper-facing collections
- replacing the existing method families
- changing fit summarization logic
- introducing name parsing as a primary source of metadata
