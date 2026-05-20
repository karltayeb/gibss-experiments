from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Iterable

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
    fit_logistic_method,
    fit_twogroup_method,
    gaussian_markov_X,
    hallmark_gene_sets_X,
    summarize_cox_method,
    summarize_logistic_method,
    summarize_twogroup_method,
    uniform_markov_X,
    uniform_single_effect,
)
from utils import BatchSpec, CollectionSpec

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
    "COLLECTION_SPECS",
    "collection_yaml_node",
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
SER_ENRICH = "ser_enrich"
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


def _twogroup_oracle_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"twogroup_oracle_L{L}",
        fit_function=fit_twogroup_method,
        summarize_function=summarize_twogroup_method,
        kwargs={"f1": None, "L": int(L)},
    )


def _twogroup_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"twogroup_L{L}",
        fit_function=fit_twogroup_method,
        summarize_function=summarize_twogroup_method,
        kwargs={"f1": F1INIT, "L": int(L)},
    )


def _default_method_specs(
    *, L: int, thresholds: tuple[float, ...]
) -> tuple[MethodSpec, ...]:
    return (
        _cox_heavy_method_spec(L=L),
        _logistic_oracle_method_spec(L=L),
        _twogroup_oracle_method_spec(L=L),
        _twogroup_method_spec(L=L),
        *tuple(
            _cox_light_threshold_method_spec(threshold, L=L) for threshold in thresholds
        ),
        *tuple(
            _logistic_threshold_method_spec(threshold, L=L) for threshold in thresholds
        ),
    )


THRESHOLD_SWEEP_SER_SPECS = _default_method_specs(L=1, thresholds=THRESHOLDS)
THRESHOLD_SWEEP_SUSIE_SPECS = _default_method_specs(L=5, thresholds=THRESHOLDS)
DEFAULT_SER_SPECS = _default_method_specs(L=1, thresholds=THRESHOLDS_SMALL)
DEFAULT_SUSIE_SPECS = _default_method_specs(L=5, thresholds=THRESHOLDS_SMALL)
DEFAULT_METHOD_SPECS = DEFAULT_SER_SPECS + DEFAULT_SUSIE_SPECS


def _signal_name(kind: str, value: float) -> str:
    return f"{kind}_{format_float(value)}"


def _simulation_name(*, design: str, enrichment: str, signal: str) -> str:
    return f"design={design}__enrichment={enrichment}__signal={signal}"


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


def _register_atomic_collection(
    *,
    design: str,
    signal_kind: str,
    signal_value: float,
    methods: Iterable[MethodSpec] = DEFAULT_METHOD_SPECS,
) -> CollectionSpec:
    collection_name = _atomic_collection_name(
        design=design,
        signal_kind=signal_kind,
        signal_value=signal_value,
    )
    return REGISTRY.register_collection(
        name=collection_name,
        simulations=(_named_simulation(collection_name),),
        methods=tuple(methods),
        n_batches=N_BATCHES,
        replicates_per_batch=REPLICATES_PER_BATCH,
        batch_builder=batch_specs_for_simulation,
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


def _register_signal_collections() -> tuple[CollectionSpec, ...]:
    collections: list[CollectionSpec] = []
    for design in SIGNAL_DESIGNS:
        for loc in SIGNAL_LOC_VALUES:
            collections.append(
                _register_atomic_collection(
                    design=design,
                    signal_kind="loc",
                    signal_value=loc,
                )
            )
        for scale in SIGNAL_SCALE_VALUES:
            collections.append(
                _register_atomic_collection(
                    design=design,
                    signal_kind="scale",
                    signal_value=scale,
                )
            )
    return tuple(collections)


def _register_correlation_collections() -> tuple[CollectionSpec, ...]:
    collections: list[CollectionSpec] = []
    for design in CORRELATION_GAUSSIAN_DESIGNS + CORRELATION_UNIFORM_DESIGNS:
        for loc in (CORRELATION_LOC_ANCHOR, CORRELATION_LOC_ANCHOR_STRONG):
            collections.append(
                _register_atomic_collection(design=design, signal_kind="loc", signal_value=loc)
            )
        for scale in (CORRELATION_SCALE_ANCHOR, CORRELATION_SCALE_ANCHOR_STRONG):
            collections.append(
                _register_atomic_collection(design=design, signal_kind="scale", signal_value=scale)
            )
    return tuple(collections)


def _register_n_feature_collections() -> tuple[CollectionSpec, ...]:
    collections: list[CollectionSpec] = []
    for design in N_FEATURE_GAUSSIAN_DESIGNS + N_FEATURE_UNIFORM_DESIGNS:
        for loc in (CORRELATION_LOC_ANCHOR, CORRELATION_LOC_ANCHOR_STRONG):
            collections.append(
                _register_atomic_collection(design=design, signal_kind="loc", signal_value=loc)
            )
        for scale in (CORRELATION_SCALE_ANCHOR, CORRELATION_SCALE_ANCHOR_STRONG):
            collections.append(
                _register_atomic_collection(design=design, signal_kind="scale", signal_value=scale)
            )
    return tuple(collections)


SIGNAL_COLLECTION_SPECS = _register_signal_collections()
CORRELATION_COLLECTION_SPECS = _register_correlation_collections()
N_FEATURE_COLLECTION_SPECS = _register_n_feature_collections()

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
REGISTRY.register_collection(
    name="tiny_test",
    batches=(TINY_TEST_BATCH,),
    methods=(_logistic_oracle_method_spec(L=1),),
)


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


def _flat_batch_node(batch: "BatchSpec") -> dict:
    """Produce a flat batch node matching the file-based collection_spec.yaml format."""
    return {
        HASH_KEY: dehydrate_hashed(batch)[HASH_KEY],
        "name": batch.name,
        "replicates": list(batch.replicates),
        "simulation_spec": dehydrate_hashed(batch.simulation_spec),
    }


def collection_yaml_node(name: str) -> dict[str, object]:
    collection = next(
        (collection for collection in REGISTRY.collections if collection.name == name),
        None,
    )
    if collection is None:
        raise KeyError(f"Unknown collection name: {name}")
    return {
        "name": collection.name,
        "batches": [_flat_batch_node(batch) for batch in collection.batches],
        "method_specs": [
            dehydrate_hashed(method) for method in collection.method_specs
        ],
        HASH_KEY: dehydrate_hashed(collection)[HASH_KEY],
    }


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
COLLECTION_SPECS = REGISTRY.collections


if __name__ == "__main__":
    manifest_path = write_manifest()
    print(f"Manifest written to {manifest_path}")
