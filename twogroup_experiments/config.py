from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from gibss.distributions import Normal, PointMass

from config_builders import batch_specs_for_simulation, fixed_normal, format_float
from config_registry import ConfigRegistry
from core import (
    HASH_KEY,
    MethodSpec,
    SimulationSpec,
    c4_gene_sets_X,
    dehydrate_hashed,
    fit_cox_method,
    fit_linear_method,
    fit_logistic_method,
    fit_twogroup_method,
    gaussian_markov_X,
    hallmark_gene_sets_X,
    null_enrich_X,
    summarize_cox_method,
    summarize_linear_method,
    summarize_logistic_method,
    summarize_twogroup_method,
    t_error_sampler,
    uniform_markov_X,
    uniform_single_effect,
)
from utils import BatchSpec

__all__ = [
    "THRESHOLDS",
    "THRESHOLDS_SMALL",
    "REPLICATES_PER_BATCH",
    "N_BATCHES",
    "LOC_GRID",
    "SCALE_GRID",
    "RHO_GRID",
    "N_FEATURE_GRID",
    "SIGNAL_LOC_VALUES",
    "SIGNAL_SCALE_VALUES",
    "CORRELATION_RHO_VALUES",
    "N_FEATURE_VALUES",
    "ConfigRegistry",
    "SIMULATION_SPECS",
    "THRESHOLD_SWEEP_SER_SPECS",
    "THRESHOLD_SWEEP_SUSIE_SPECS",
    "DEFAULT_SER_SPECS",
    "DEFAULT_SUSIE_SPECS",
    "DEFAULT_METHOD_SPECS",
    "BANK_BATCH_SPECS",
    "BATCH_SPECS",
    "NULL_ENRICH_SIMULATION_SPECS",
    "NULL_ENRICH_METHOD_SPECS",
    "T_ERROR_SIMULATION_SPECS",
    "NULL_SIMULATION_SPECS",
    "NULL_INTERCEPT_VALUES",
    "NULL_METHOD_SPECS",
    "GRID_SIMULATION_SPECS",
    "GRID_EFFECT_LOC_VALUES",
    "GRID_EFFECT_SCALE_VALUES",
    "GRID_INTERCEPT_VALUES",
]

REGISTRY = ConfigRegistry()

THRESHOLDS = (
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
)
THRESHOLDS_SMALL = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

REPLICATES_PER_BATCH = 50
N_BATCHES = 1
BASE_SEED = 20260501

LOC_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5)
SCALE_GRID = (
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.25,
    4.5,
    4.75,
    5.0,
)
RHO_GRID = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
)
N_FEATURE_GRID = (100, 200, 400, 800, 1600)

SIGNAL_LOC_VALUES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SIGNAL_SCALE_VALUES = (0.75, 1.0, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0)
CORRELATION_RHO_VALUES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99)
N_FEATURE_VALUES = N_FEATURE_GRID

F0 = PointMass(0.0)
F1INIT = Normal(loc=0.0, scale=1.0, estimate_loc=True, estimate_scale=True)
F1INIT_SCALE_FAM = Normal(loc=0.0, scale=1.0, estimate_loc=False, estimate_scale=True)
F1INIT_LOC_FAM = Normal(loc=0.0, scale=0.1, estimate_loc=True, estimate_scale=False)
SER_ENRICH = "ser_enrich"
NULL_ENRICH = "null_enrich"
NULL_ENRICH_N = 4384
LOC_SCALE_FIXED = 0.1
CORRELATION_LOC_ANCHOR = 1.5
CORRELATION_LOC_ANCHOR_STRONG = 2.0
CORRELATION_SCALE_ANCHOR = 1.75
CORRELATION_SCALE_ANCHOR_STRONG = 2.25
SIGNAL_RHO = 0.90
SIGNAL_N_FEATURES = 100
SAMPLE_SIZE_RHO = 0.90


def _logistic_threshold_method_spec(threshold: float, *, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"logistic_threshold_L{L}",
        fit_function=fit_logistic_method,
        summarize_function=summarize_logistic_method,
        kwargs={
            "response_source": "score_threshold",
            "threshold": float(threshold),
            "L": int(L),
        },
    )


def _cox_light_threshold_method_spec(threshold: float, *, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"cox_light_threshold_L{L}",
        fit_function=fit_cox_method,
        summarize_function=summarize_cox_method,
        kwargs={
            "threshold": float(threshold),
            "time_sign": -1.0,
            "L": int(L),
        },
    )


def _cox_heavy_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"cox_heavy_L{L}",
        fit_function=fit_cox_method,
        summarize_function=summarize_cox_method,
        kwargs={"threshold": None, "time_sign": 1.0, "L": int(L)},
    )


def _logistic_oracle_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"logistic_oracle_L{L}",
        fit_function=fit_logistic_method,
        summarize_function=summarize_logistic_method,
        kwargs={"response_source": "z", "threshold": None, "L": int(L)},
    )


_TG_ITER_KWARGS = {"n_null_iter": 20, "n_intercept_iter": 20}


def _twogroup_oracle_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"twogroup_oracle_L{L}",
        fit_function=fit_twogroup_method,
        summarize_function=summarize_twogroup_method,
        kwargs={"f1": None, "L": int(L), **_TG_ITER_KWARGS},
    )


def _twogroup_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"twogroup_L{L}",
        fit_function=fit_twogroup_method,
        summarize_function=summarize_twogroup_method,
        kwargs={"f1": F1INIT, "L": int(L), **_TG_ITER_KWARGS},
    )


def _twogroup_oracle_init_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"twogroup_oracle_init_L{L}",
        fit_function=fit_twogroup_method,
        summarize_function=summarize_twogroup_method,
        kwargs={"f1": F1INIT, "oracle_init": True, "L": int(L), **_TG_ITER_KWARGS},
    )


def _twogroup_scale_fam_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"twogroup_scale_fam_L{L}",
        fit_function=fit_twogroup_method,
        summarize_function=summarize_twogroup_method,
        kwargs={"f1": F1INIT_SCALE_FAM, "L": int(L), **_TG_ITER_KWARGS},
    )


def _twogroup_loc_fam_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"twogroup_loc_fam_L{L}",
        fit_function=fit_twogroup_method,
        summarize_function=summarize_twogroup_method,
        kwargs={"f1": F1INIT_LOC_FAM, "L": int(L), **_TG_ITER_KWARGS},
    )


def _linear_fixed_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"linear_fixed_L{L}",
        fit_function=fit_linear_method,
        summarize_function=summarize_linear_method,
        kwargs={"estimate_residual_variance": False, "L": int(L)},
    )


def _linear_estimated_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"linear_estimated_L{L}",
        fit_function=fit_linear_method,
        summarize_function=summarize_linear_method,
        kwargs={"estimate_residual_variance": True, "L": int(L)},
    )


def _default_method_specs(
    *, L: int, thresholds: tuple[float, ...]
) -> tuple[MethodSpec, ...]:
    return (
        _cox_heavy_method_spec(L=L),
        _logistic_oracle_method_spec(L=L),
        _twogroup_oracle_method_spec(L=L),
        _twogroup_method_spec(L=L),
        _twogroup_oracle_init_method_spec(L=L),
        _twogroup_scale_fam_method_spec(L=L),
        _twogroup_loc_fam_method_spec(L=L),
        *tuple(
            _cox_light_threshold_method_spec(threshold, L=L) for threshold in thresholds
        ),
        *tuple(
            _logistic_threshold_method_spec(threshold, L=L) for threshold in thresholds
        ),
        _linear_fixed_method_spec(L=L),
        _linear_estimated_method_spec(L=L),
    )


THRESHOLD_SWEEP_SER_SPECS = _default_method_specs(L=1, thresholds=THRESHOLDS)
THRESHOLD_SWEEP_SUSIE_SPECS = _default_method_specs(L=5, thresholds=THRESHOLDS)
DEFAULT_SER_SPECS = _default_method_specs(L=1, thresholds=THRESHOLDS_SMALL)
DEFAULT_SUSIE_SPECS = _default_method_specs(L=5, thresholds=THRESHOLDS_SMALL)
DEFAULT_METHOD_SPECS = DEFAULT_SER_SPECS + DEFAULT_SUSIE_SPECS
REGISTRY.register_methods(THRESHOLD_SWEEP_SER_SPECS + THRESHOLD_SWEEP_SUSIE_SPECS)


def _signal_name(kind: str, value: float) -> str:
    return f"{kind}_{format_float(value)}"


def _simulation_name(*, design: str, enrichment: str, signal: str, error: str | None = None, intercept: float | None = None) -> str:
    base = f"design={design}__enrichment={enrichment}__signal={signal}"
    if intercept is not None:
        base = f"{base}__intercept={format_float(intercept)}"
    return base if error is None else f"{base}__error={error}"


def _markov_design_name(*, family: str, rho: float, n_features: int) -> str:
    return f"{family}_markov_rho_{format_float(rho)}_n_features_{int(n_features)}"


def _make_simulation(
    *,
    design_name: str,
    design_sampler,
    signal_kind: str,
    signal_value: float,
) -> SimulationSpec:
    if signal_kind == "loc":
        f1 = fixed_normal(loc=signal_value, scale=LOC_SCALE_FIXED)
    elif signal_kind == "scale":
        f1 = fixed_normal(loc=0.0, scale=signal_value)
    else:
        raise ValueError(f"Unknown signal kind: {signal_kind}")

    return SimulationSpec(
        name=_simulation_name(
            design=design_name,
            enrichment=SER_ENRICH,
            signal=_signal_name(signal_kind, signal_value),
        ),
        design_sampler=design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-2.0,
        f0=F0,
        f1=f1,
        base_seed=BASE_SEED,
    )


def _dense_signal_values() -> tuple[tuple[str, float], ...]:
    return tuple(("loc", value) for value in LOC_GRID) + tuple(
        ("scale", value) for value in SCALE_GRID
    )


def _build_design_kwargs() -> dict[str, dict[str, object]]:
    design_kwargs: dict[str, dict[str, object]] = {
        "hallmark": {"design_sampler": hallmark_gene_sets_X},
        "c4": {"design_sampler": c4_gene_sets_X},
    }

    for family, sampler in (
        ("gaussian", gaussian_markov_X),
        ("uniform", uniform_markov_X),
    ):
        for rho in RHO_GRID:
            design_name = _markov_design_name(
                family=family,
                rho=rho,
                n_features=SIGNAL_N_FEATURES,
            )
            design_kwargs[design_name] = {
                "design_sampler": partial(
                    sampler,
                    n=500,
                    p=SIGNAL_N_FEATURES,
                    rho=rho,
                )
            }
        for n_features in N_FEATURE_GRID:
            design_name = _markov_design_name(
                family=family,
                rho=SAMPLE_SIZE_RHO,
                n_features=n_features,
            )
            design_kwargs[design_name] = {
                "design_sampler": partial(
                    sampler,
                    n=500,
                    p=n_features,
                    rho=SAMPLE_SIZE_RHO,
                )
            }

    return design_kwargs


DESIGN_KWARGS = _build_design_kwargs()

RAW_SIMULATION_SPECS = tuple(
    _make_simulation(
        design_name=design_name,
        design_sampler=kwargs["design_sampler"],
        signal_kind=signal_kind,
        signal_value=signal_value,
    )
    for design_name, kwargs in DESIGN_KWARGS.items()
    for signal_kind, signal_value in _dense_signal_values()
)
SIMULATION_BY_NAME = {spec.name: spec for spec in RAW_SIMULATION_SPECS}

REGISTRY.register_simulations(RAW_SIMULATION_SPECS)

BANK_BATCH_SPECS = tuple(
    batch
    for simulation_spec in RAW_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        simulation_spec,
        replicates_per_batch=REPLICATES_PER_BATCH,
        n_batches=N_BATCHES,
    )
)
REGISTRY.register_batches(BANK_BATCH_SPECS)


def _named_simulation(name: str) -> SimulationSpec:
    try:
        return SIMULATION_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unknown simulation name: {name}") from exc


def _atomic_collection_name(
    *, design: str, signal_kind: str, signal_value: float
) -> str:
    return _simulation_name(
        design=design,
        enrichment=SER_ENRICH,
        signal=_signal_name(signal_kind, signal_value),
    )


SIGNAL_DESIGNS = (
    "hallmark",
    "c4",
    _markov_design_name(
        family="gaussian", rho=SIGNAL_RHO, n_features=SIGNAL_N_FEATURES
    ),
    _markov_design_name(family="uniform", rho=SIGNAL_RHO, n_features=SIGNAL_N_FEATURES),
)

CORRELATION_GAUSSIAN_DESIGNS = tuple(
    _markov_design_name(family="gaussian", rho=rho, n_features=SIGNAL_N_FEATURES)
    for rho in CORRELATION_RHO_VALUES
)
CORRELATION_UNIFORM_DESIGNS = tuple(
    _markov_design_name(family="uniform", rho=rho, n_features=SIGNAL_N_FEATURES)
    for rho in CORRELATION_RHO_VALUES
)
N_FEATURE_GAUSSIAN_DESIGNS = tuple(
    _markov_design_name(family="gaussian", rho=SAMPLE_SIZE_RHO, n_features=n_features)
    for n_features in N_FEATURE_VALUES
)
N_FEATURE_UNIFORM_DESIGNS = tuple(
    _markov_design_name(family="uniform", rho=SAMPLE_SIZE_RHO, n_features=n_features)
    for n_features in N_FEATURE_VALUES
)


def _make_null_enrich_simulation(
    *,
    signal_kind: str,
    signal_value: float,
) -> SimulationSpec:
    if signal_kind == "loc":
        f1 = fixed_normal(loc=signal_value, scale=LOC_SCALE_FIXED)
    elif signal_kind == "scale":
        f1 = fixed_normal(loc=0.0, scale=signal_value)
    else:
        raise ValueError(f"Unknown signal kind: {signal_kind}")
    return SimulationSpec(
        name=_simulation_name(
            design=NULL_ENRICH,
            enrichment=NULL_ENRICH,
            signal=_signal_name(signal_kind, signal_value),
        ),
        design_sampler=partial(null_enrich_X, n=NULL_ENRICH_N),
        effect_sampler=partial(uniform_single_effect, causal_effect=0.0),
        intercept=-2.0,
        f0=F0,
        f1=f1,
        base_seed=BASE_SEED,
    )


_NULL_ENRICH_SIGNAL_VALUES = (
    [("loc", v) for v in SIGNAL_LOC_VALUES]
    + [("scale", v) for v in SIGNAL_SCALE_VALUES]
)

NULL_ENRICH_SIMULATION_SPECS = tuple(
    _make_null_enrich_simulation(signal_kind=kind, signal_value=value)
    for kind, value in _NULL_ENRICH_SIGNAL_VALUES
)
SIMULATION_BY_NAME.update({spec.name: spec for spec in NULL_ENRICH_SIMULATION_SPECS})

NULL_ENRICH_METHOD_SPECS = (
    _twogroup_oracle_method_spec(L=1),
    _twogroup_method_spec(L=1),
    _twogroup_oracle_init_method_spec(L=1),
    _twogroup_scale_fam_method_spec(L=1),
    _twogroup_loc_fam_method_spec(L=1),
)

_NULL_ENRICH_BATCH_SPECS = tuple(
    batch
    for sim in NULL_ENRICH_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        sim,
        replicates_per_batch=REPLICATES_PER_BATCH,
        n_batches=N_BATCHES,
    )
)
REGISTRY.register_simulations(NULL_ENRICH_SIMULATION_SPECS)
REGISTRY.register_batches(_NULL_ENRICH_BATCH_SPECS)


T_ERROR_DFS = (3, 5, 10, 30)

T_ERROR_SIGNAL_VALUES: dict[str, dict[str, float]] = {
    "hallmark": {"loc": 2.0, "scale": 2.0},
    "c4": {"loc": 2.0, "scale": 2.0},
    _markov_design_name(
        family="gaussian", rho=SIGNAL_RHO, n_features=SIGNAL_N_FEATURES
    ): {"loc": 2.0, "scale": 2.0},
    _markov_design_name(
        family="uniform", rho=SIGNAL_RHO, n_features=SIGNAL_N_FEATURES
    ): {"loc": 2.0, "scale": 2.0},
}


def _make_t_error_simulation(
    *,
    design_name: str,
    design_sampler,
    signal_kind: str,
    signal_value: float,
    error_df: int,
) -> SimulationSpec:
    if signal_kind == "loc":
        f1 = fixed_normal(loc=signal_value, scale=LOC_SCALE_FIXED)
    elif signal_kind == "scale":
        f1 = fixed_normal(loc=0.0, scale=signal_value)
    else:
        raise ValueError(f"Unknown signal kind: {signal_kind!r}")
    return SimulationSpec(
        name=_simulation_name(
            design=design_name,
            enrichment=SER_ENRICH,
            signal=_signal_name(signal_kind, signal_value),
            error=f"t_df_{error_df}",
        ),
        design_sampler=design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-2.0,
        f0=F0,
        f1=f1,
        base_seed=BASE_SEED,
        error_sampler=partial(t_error_sampler, df=error_df),
    )


T_ERROR_SIMULATION_SPECS = tuple(
    _make_t_error_simulation(
        design_name=design_name,
        design_sampler=DESIGN_KWARGS[design_name]["design_sampler"],
        signal_kind=signal_kind,
        signal_value=T_ERROR_SIGNAL_VALUES[design_name][signal_kind],
        error_df=df,
    )
    for design_name in T_ERROR_SIGNAL_VALUES
    for signal_kind in ("loc", "scale")
    for df in T_ERROR_DFS
)

SIMULATION_BY_NAME.update({spec.name: spec for spec in T_ERROR_SIMULATION_SPECS})
REGISTRY.register_simulations(T_ERROR_SIMULATION_SPECS)
REGISTRY.register_batches(tuple(
    batch
    for spec in T_ERROR_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        spec,
        replicates_per_batch=REPLICATES_PER_BATCH,
        n_batches=N_BATCHES,
    )
))


NULL_METHOD_SPECS = DEFAULT_SER_SPECS
NULL_SIM = "null"
NULL_INTERCEPT_VALUES = (-3.0, -2.0, -1.0, 0.0)
_NULL_SIGNAL_VALUES: list[tuple[str, float]] = (
    [("loc", v) for v in SIGNAL_LOC_VALUES]
    + [("scale", v) for v in SIGNAL_SCALE_VALUES]
)


def _make_null_simulation(
    *,
    design_name: str,
    design_sampler,
    signal_kind: str,
    signal_value: float,
    intercept: float,
) -> SimulationSpec:
    if signal_kind == "loc":
        f1 = fixed_normal(loc=signal_value, scale=LOC_SCALE_FIXED)
    elif signal_kind == "scale":
        f1 = fixed_normal(loc=0.0, scale=signal_value)
    else:
        raise ValueError(f"Unknown signal kind: {signal_kind}")
    return SimulationSpec(
        name=_simulation_name(
            design=design_name,
            enrichment=NULL_SIM,
            signal=_signal_name(signal_kind, signal_value),
            intercept=intercept,
        ),
        design_sampler=design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=0.0),
        intercept=float(intercept),
        f0=F0,
        f1=f1,
        base_seed=BASE_SEED,
    )


NULL_SIMULATION_SPECS = tuple(
    _make_null_simulation(
        design_name=design_name,
        design_sampler=DESIGN_KWARGS[design_name]["design_sampler"],
        signal_kind=signal_kind,
        signal_value=signal_value,
        intercept=intercept,
    )
    for design_name in SIGNAL_DESIGNS
    for signal_kind, signal_value in _NULL_SIGNAL_VALUES
    for intercept in NULL_INTERCEPT_VALUES
)

SIMULATION_BY_NAME.update({spec.name: spec for spec in NULL_SIMULATION_SPECS})
REGISTRY.register_simulations(NULL_SIMULATION_SPECS)
REGISTRY.register_batches(tuple(
    batch
    for spec in NULL_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        spec,
        replicates_per_batch=REPLICATES_PER_BATCH,
        n_batches=N_BATCHES,
    )
))


# --- full b0 x b enrichment grid ---
# Naming: design=X__enrichment=b0_{b0}_b_{b}__signal={kind}_{|b|}
# b=0 is the null case (same ser_enrich function, causal_effect=0; zero-effect
# causal vars are removed before reporting). scale requires b >= 0.
GRID_EFFECT_LOC_VALUES: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
GRID_EFFECT_SCALE_VALUES: tuple[float, ...] = (0.0, 1.0, 2.0)
GRID_INTERCEPT_VALUES: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0)


def _grid_enrichment_name(b0: float, b: float) -> str:
    return f"b0_{format_float(b0)}_b_{format_float(b)}"


def _make_grid_simulation(
    *,
    design_name: str,
    design_sampler,
    signal_kind: str,
    b: float,
    b0: float,
) -> SimulationSpec:
    abs_b = abs(b)
    if signal_kind == "loc":
        f1 = fixed_normal(loc=b, scale=LOC_SCALE_FIXED)
        signal = _signal_name("loc", b)
    elif signal_kind == "scale":
        scale_val = abs_b if abs_b > 0 else 1.0  # scale must be positive; b=0 uses prior=1.0
        f1 = fixed_normal(loc=0.0, scale=scale_val)
        signal = _signal_name("scale", abs_b)
    else:
        raise ValueError(f"Unknown signal kind: {signal_kind!r}")
    return SimulationSpec(
        name=f"design={design_name}__enrichment={_grid_enrichment_name(b0, b)}__signal={signal}",
        design_sampler=design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=b),
        intercept=float(b0),
        f0=F0,
        f1=f1,
        base_seed=BASE_SEED,
    )


def _grid_signal_effect_pairs() -> list[tuple[str, float]]:
    return (
        [("loc", b) for b in GRID_EFFECT_LOC_VALUES]
        + [("scale", b) for b in GRID_EFFECT_SCALE_VALUES]
    )


GRID_SIMULATION_SPECS = tuple(
    _make_grid_simulation(
        design_name=design_name,
        design_sampler=DESIGN_KWARGS[design_name]["design_sampler"],
        signal_kind=signal_kind,
        b=b,
        b0=b0,
    )
    for design_name in SIGNAL_DESIGNS
    for signal_kind, b in _grid_signal_effect_pairs()
    for b0 in GRID_INTERCEPT_VALUES
)

SIMULATION_BY_NAME.update({spec.name: spec for spec in GRID_SIMULATION_SPECS})
REGISTRY.register_simulations(GRID_SIMULATION_SPECS)
REGISTRY.register_batches(tuple(
    batch
    for spec in GRID_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        spec,
        replicates_per_batch=REPLICATES_PER_BATCH,
        n_batches=N_BATCHES,
    )
))


TINY_TEST_SIMULATION = _atomic_collection_name(
    design="hallmark",
    signal_kind="loc",
    signal_value=SIGNAL_LOC_VALUES[0],
)
TINY_TEST_BATCH = BatchSpec(
    name="tiny_test",
    simulation_spec=_named_simulation(TINY_TEST_SIMULATION),
    replicates=(0,),
)
REGISTRY.register_batches((TINY_TEST_BATCH,))


def manifest_dict() -> dict[str, object]:
    manifest: dict[str, object] = {
        "simulation_specs": {},
        "method_specs": {},
        "batches": {},
    }
    simulation_specs = manifest["simulation_specs"]
    method_specs = manifest["method_specs"]
    batches = manifest["batches"]

    assert isinstance(simulation_specs, dict)
    assert isinstance(method_specs, dict)
    assert isinstance(batches, dict)

    for simulation_spec in REGISTRY.simulations:
        simulation_node = dehydrate_hashed(simulation_spec)
        simulation_specs[simulation_node[HASH_KEY]] = simulation_node

    for method_spec in REGISTRY.methods:
        method_node = dehydrate_hashed(method_spec)
        method_specs[method_node[HASH_KEY]] = method_node

    for batch in REGISTRY.batches:
        simulation_node = dehydrate_hashed(batch.simulation_spec)
        batch_node = {
            "name": batch.name,
            "simulation_spec": simulation_node,
            "replicates": list(batch.replicates),
        }
        batch_record = {**batch_node, HASH_KEY: dehydrate_hashed(batch)[HASH_KEY]}
        batches[batch_record[HASH_KEY]] = batch_record

    return manifest


def write_manifest(path: str | Path | None = None) -> Path:
    if path is not None:
        destination = Path(path)
    else:
        destination = Path(__file__).resolve().parent / "results" / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest_dict(), indent=2), encoding="utf-8")
    return destination


SIMULATION_SPECS = REGISTRY.simulations
BATCH_SPECS = REGISTRY.batches


if __name__ == "__main__":
    manifest_path = write_manifest()
    print(f"Manifest written to {manifest_path}")
