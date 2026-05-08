from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Iterable

from gibss.distributions import Normal, PointMass

from config_builders import (
    batch_specs_for_simulation,
    fixed_normal,
    format_float,
    make_product_simulation_specs,
)
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
    uniform_markov_X,
    hallmark_gene_sets_X,
    summarize_cox_method,
    summarize_logistic_method,
    summarize_twogroup_method,
    uniform_single_effect,
)
from utils import BatchSpec, CollectionSpec

__all__ = [
    "THRESHOLDS",
    "HALLMARK_REPLICATES_PER_BATCH",
    "HALLMARK_N_BATCHES",
    "C4_REPLICATES_PER_BATCH",
    "C4_N_BATCHES",
    "ConfigRegistry",
    "SIMULATION_SPECS",
    "THRESHOLD_SWEEP_SER_SPECS",
    "THRESHOLD_SWEEP_SUSIE_SPECS",
    "DEFAULT_SER_SPECS",
    "DEFAULT_SUSIE_SPECS",
    "HALLMARK_BATCH_SPECS",
    "C4_BATCH_SPECS",
    "BATCH_SPECS",
    "COLLECTION_SPECS",
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

THRESHOLDS_SMALL = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
HALLMARK_REPLICATES_PER_BATCH = 50
HALLMARK_N_BATCHES = 2
C4_REPLICATES_PER_BATCH = 20
C4_N_BATCHES = 5
BASE_SEED = 20260501

F0 = PointMass(0.0)
F1INIT = Normal(loc=0.0, scale=1.0, estimate_loc=True, estimate_scale=True)

SEPARATOR = "__"
LOC_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
SCALE_GRID = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


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


MSIGDB_DESIGN_KWARGS = {
    "hallmark": {"design_sampler": hallmark_gene_sets_X},
    "c4": {"design_sampler": c4_gene_sets_X},
}
GAUSSIAN_DESIGN_KWARGS = {
    f"gaussian_markov_{rho:.2f}": {
        "design_sampler": partial(gaussian_markov_X, n=500, p=100, rho=rho)
    }
    for rho in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
}

UNIFORM_DESIGN_KWARGS = {
    f"uniform_markov_{rho:.2f}": {
        "design_sampler": partial(uniform_markov_X, n=500, p=100, rho=rho)
    }
    for rho in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
}

DESIGN_KWARGS = MSIGDB_DESIGN_KWARGS | GAUSSIAN_DESIGN_KWARGS | UNIFORM_DESIGN_KWARGS

REGIME_KWARGS = {
    "ser_enrich": {
        "effect_sampler": partial(uniform_single_effect, causal_effect=2.0),
        "intercept": -2.0,
    },
    "ser_dep": {
        "effect_sampler": partial(uniform_single_effect, causal_effect=-2.0),
        "intercept": -1.0,
    },
}

F1_KWARGS = {
    **{
        f"loc_{format_float(loc)}": {"f1": fixed_normal(loc=loc, scale=0.1)}
        for loc in LOC_GRID
    },
    **{
        f"scale_{format_float(scale)}": {"f1": fixed_normal(loc=0.0, scale=scale)}
        for scale in SCALE_GRID
    },
}

# take the product of simulations (design, regime, f1)
RAW_SIMULATION_SPECS = make_product_simulation_specs(
    design_names=tuple(DESIGN_KWARGS),
    regime_names=tuple(REGIME_KWARGS),
    f1_names=tuple(F1_KWARGS),
    separator=SEPARATOR,
    f0=F0,
    base_seed=BASE_SEED,
    design_kwargs=DESIGN_KWARGS,
    regime_kwargs=REGIME_KWARGS,
    f1_kwargs=F1_KWARGS,
)
SIMULATION_BY_NAME = {spec.name: spec for spec in RAW_SIMULATION_SPECS}


def _named_simulations(*names: str) -> tuple[SimulationSpec, ...]:
    missing = [name for name in names if name not in SIMULATION_BY_NAME]
    if missing:
        raise KeyError(f"Unknown simulation names: {', '.join(missing)}")
    return tuple(SIMULATION_BY_NAME[name] for name in names)


def _simulation_names_for(
    *,
    design: str,
    regime: str,
    f1_prefix: str | None = None,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in SIMULATION_BY_NAME
        if name.startswith(f"{design}{SEPARATOR}{regime}{SEPARATOR}")
        and (f1_prefix is None or name.split(SEPARATOR)[2].startswith(f1_prefix))
    )


def _simulation_names_for_design_prefix(
    *,
    design_prefix: str,
    regime: str,
    f1_prefix: str | None = None,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in SIMULATION_BY_NAME
        if name.startswith(design_prefix)
        and f"{SEPARATOR}{regime}{SEPARATOR}" in name
        and (f1_prefix is None or name.split(SEPARATOR)[2].startswith(f1_prefix))
    )


def make_collection(
    *,
    name: str,
    simulations: Iterable[SimulationSpec],
    methods: Iterable[MethodSpec] = THRESHOLD_SWEEP_SER_SPECS,
    n_batches: int,
    replicates_per_batch: int,
) -> CollectionSpec:
    return REGISTRY.register_collection(
        name=name,
        simulations=tuple(simulations),
        methods=tuple(methods),
        n_batches=n_batches,
        replicates_per_batch=replicates_per_batch,
        batch_builder=batch_specs_for_simulation,
    )


def make_collection_union(
    *,
    name: str,
    collections: Iterable[str],
) -> CollectionSpec:
    return REGISTRY.register_collection_union(name=name, collections=tuple(collections))


def _register_single_batch_collections(batches: Iterable[BatchSpec]) -> None:
    for batch in batches:
        REGISTRY.register_collection(
            name=batch.name,
            batches=(batch,),
            methods=THRESHOLD_SWEEP_SER_SPECS,
        )


def _register_fit_collections(
    *,
    collection_prefix: str,
    simulations: tuple[SimulationSpec, ...],
    ser_methods: tuple[MethodSpec, ...],
    susie_methods: tuple[MethodSpec, ...],
    n_batches: int,
    replicates_per_batch: int,
) -> None:
    fit_collections: list[str] = []
    if ser_methods is not None:
        make_collection(
            name=f"{collection_prefix}__ser_fits",
            simulations=simulations,
            methods=ser_methods,
            n_batches=n_batches,
            replicates_per_batch=replicates_per_batch,
        )
        fit_collections.append(f"{collection_prefix}__ser_fits")
    if susie_methods is not None:
        make_collection(
            name=f"{collection_prefix}__susie_fits",
            simulations=simulations,
            methods=susie_methods,
            n_batches=n_batches,
            replicates_per_batch=replicates_per_batch,
        )
        fit_collections.append(f"{collection_prefix}__susie_fits")
    make_collection_union(
        name=collection_prefix,
        collections=tuple(fit_collections),
    )


def _register_fit_collection_unions(
    *,
    collection_prefix: str,
    collections: tuple[str, ...],
) -> None:
    make_collection_union(
        name=f"{collection_prefix}__ser_fits",
        collections=tuple(f"{name}__ser_fits" for name in collections),
    )
    make_collection_union(
        name=f"{collection_prefix}__susie_fits",
        collections=tuple(f"{name}__susie_fits" for name in collections),
    )


HALLMARK_SIMULATION_SPECS = tuple(
    spec for spec in RAW_SIMULATION_SPECS if spec.name.startswith("hallmark__")
)
C4_SIMULATION_SPECS = tuple(
    spec for spec in RAW_SIMULATION_SPECS if spec.name.startswith("c4__")
)

HALLMARK_BATCH_SPECS = tuple(
    batch
    for simulation_spec in HALLMARK_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        simulation_spec,
        replicates_per_batch=HALLMARK_REPLICATES_PER_BATCH,
        n_batches=HALLMARK_N_BATCHES,
    )
)
C4_BATCH_SPECS = tuple(
    batch
    for simulation_spec in C4_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        simulation_spec,
        replicates_per_batch=C4_REPLICATES_PER_BATCH,
        n_batches=C4_N_BATCHES,
    )
)

GAUSSIAN_BATCH_SPECS = tuple(
    batch
    for spec in RAW_SIMULATION_SPECS
    if spec.name.startswith("gaussian_")
    for batch in batch_specs_for_simulation(spec, replicates_per_batch=50, n_batches=1)
)

UNIFORM_BATCH_SPECS = tuple(
    batch
    for spec in RAW_SIMULATION_SPECS
    if spec.name.startswith("uniform_")
    for batch in batch_specs_for_simulation(spec, replicates_per_batch=50, n_batches=1)
)


_register_single_batch_collections(HALLMARK_BATCH_SPECS + C4_BATCH_SPECS)

TINY_TEST_BATCH = BatchSpec(
    name="tiny_test",
    simulation_spec=SIMULATION_BY_NAME["hallmark__ser_enrich__scale_0.50"],
    replicates=(0,),
)
REGISTRY.register_collection(
    name="tiny_test",
    batches=(TINY_TEST_BATCH,),
    methods=(_logistic_oracle_method_spec(L=1),),
)

###
# HALLMARK
###

_register_fit_collections(
    collection_prefix="hallmark__ser_enrich__loc",
    simulations=_named_simulations(
        *_simulation_names_for(
            design="hallmark",
            regime="ser_enrich",
            f1_prefix="loc_",
        )
    ),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=HALLMARK_N_BATCHES,
    replicates_per_batch=HALLMARK_REPLICATES_PER_BATCH,
)
_register_fit_collections(
    collection_prefix="hallmark__ser_enrich__scale",
    simulations=_named_simulations(
        *_simulation_names_for(
            design="hallmark",
            regime="ser_enrich",
            f1_prefix="scale_",
        )
    ),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=HALLMARK_N_BATCHES,
    replicates_per_batch=HALLMARK_REPLICATES_PER_BATCH,
)
_register_fit_collections(
    collection_prefix="hallmark__ser_enrich__scale_2.0",
    simulations=_named_simulations("hallmark__ser_enrich__scale_2.00"),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=HALLMARK_N_BATCHES,
    replicates_per_batch=HALLMARK_REPLICATES_PER_BATCH,
)
_register_fit_collection_unions(
    collection_prefix="hallmark__ser__all",
    collections=(
        "hallmark__ser_enrich__loc",
        "hallmark__ser_enrich__scale",
    ),
)


###
#  C4
###
_register_fit_collections(
    collection_prefix="c4__ser_enrich__loc",
    simulations=_named_simulations(
        *_simulation_names_for(
            design="c4",
            regime="ser_enrich",
            f1_prefix="loc_",
        )
    ),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=C4_N_BATCHES,
    replicates_per_batch=C4_REPLICATES_PER_BATCH,
)
_register_fit_collections(
    collection_prefix="c4__ser_enrich__scale",
    simulations=_named_simulations(
        *_simulation_names_for(
            design="c4",
            regime="ser_enrich",
            f1_prefix="scale_",
        )
    ),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=C4_N_BATCHES,
    replicates_per_batch=C4_REPLICATES_PER_BATCH,
)
_register_fit_collection_unions(
    collection_prefix="c4__ser__all",
    collections=(
        "c4__ser_enrich__loc",
        "c4__ser_enrich__scale",
    ),
)

###
#  Gaussian SER
###
GAUSSIAN_N_BATCHES = 1
GAUSSIAN_REPLICATES_PER_BATCH = 100
_register_fit_collections(
    collection_prefix="gaussian__ser_enrich__loc",
    simulations=_named_simulations(
        *_simulation_names_for_design_prefix(
            design_prefix="gaussian_",
            regime="ser_enrich",
            f1_prefix="loc_",
        )
    ),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=GAUSSIAN_N_BATCHES,
    replicates_per_batch=GAUSSIAN_REPLICATES_PER_BATCH,
)
_register_fit_collections(
    collection_prefix="gaussian__ser_enrich__scale",
    simulations=_named_simulations(
        *_simulation_names_for_design_prefix(
            design_prefix="gaussian_",
            regime="ser_enrich",
            f1_prefix="scale_",
        )
    ),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=GAUSSIAN_N_BATCHES,
    replicates_per_batch=GAUSSIAN_REPLICATES_PER_BATCH,
)
_register_fit_collection_unions(
    collection_prefix="gaussian__ser__all",
    collections=(
        "gaussian__ser_enrich__loc",
        "gaussian__ser_enrich__scale",
    ),
)

###
#  UNIFORM SER
###
UNIFORM_N_BATCHES = 1
UNIFORM_REPLICATES_PER_BATCH = 100
_register_fit_collections(
    collection_prefix="uniform__ser_enrich__loc",
    simulations=_named_simulations(
        *_simulation_names_for_design_prefix(
            design_prefix="uniform_",
            regime="ser_enrich",
            f1_prefix="loc_",
        )
    ),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=UNIFORM_N_BATCHES,
    replicates_per_batch=UNIFORM_REPLICATES_PER_BATCH,
)
_register_fit_collections(
    collection_prefix="uniform__ser_enrich__scale",
    simulations=_named_simulations(
        *_simulation_names_for_design_prefix(
            design_prefix="uniform_",
            regime="ser_enrich",
            f1_prefix="scale_",
        )
    ),
    ser_methods=DEFAULT_SER_SPECS,
    susie_methods=DEFAULT_SUSIE_SPECS,
    n_batches=UNIFORM_N_BATCHES,
    replicates_per_batch=UNIFORM_REPLICATES_PER_BATCH,
)
_register_fit_collection_unions(
    collection_prefix="uniform__ser__all",
    collections=(
        "uniform__ser_enrich__loc",
        "uniform__ser_enrich__scale",
    ),
)

make_collection_union(
    name="all__ser_fits",
    collections=(
        "hallmark__ser__all__ser_fits",
        "c4__ser__all__ser_fits",
        "gaussian__ser__all__ser_fits",
        "uniform__ser__all__ser_fits",
    ),
)
make_collection_union(
    name="all__susie_fits",
    collections=(
        "hallmark__ser__all__susie_fits",
        "c4__ser__all__susie_fits",
        "gaussian__ser__all__susie_fits",
        "uniform__ser__all__susie_fits",
    ),
)
make_collection_union(
    name="all",
    collections=("all__ser_fits", "all__susie_fits"),
)


def manifest_dict() -> dict[str, object]:
    manifest: dict[str, object] = {
        "simulation_specs": {},
        "method_specs": {},
        "batches": {},
        "collections": {},
    }
    simulation_specs = manifest["simulation_specs"]
    method_specs = manifest["method_specs"]
    batches = manifest["batches"]
    collections = manifest["collections"]

    assert isinstance(simulation_specs, dict)
    assert isinstance(method_specs, dict)
    assert isinstance(batches, dict)
    assert isinstance(collections, dict)

    for simulation_spec in REGISTRY.simulations:
        simulation_node = dehydrate_hashed(simulation_spec)
        simulation_specs[simulation_node[HASH_KEY]] = simulation_node

    for method_spec in REGISTRY.methods:
        method_node = dehydrate_hashed(method_spec)
        method_specs[method_node[HASH_KEY]] = method_node

    batch_records_by_name: dict[str, dict[str, object]] = {}
    for batch in REGISTRY.batches:
        simulation_node = dehydrate_hashed(batch.simulation_spec)
        batch_node = {
            "name": batch.name,
            "simulation_spec": simulation_node,
            "replicates": list(batch.replicates),
        }
        batch_record = {**batch_node, HASH_KEY: dehydrate_hashed(batch)[HASH_KEY]}
        batches[batch_record[HASH_KEY]] = batch_record
        batch_records_by_name[batch.name] = batch_record

    for collection in REGISTRY.collections:
        collection_batches = [
            batch_records_by_name[batch.name] for batch in collection.batches
        ]
        collection_method_specs = [
            dehydrate_hashed(method_spec) for method_spec in collection.method_specs
        ]
        collection_node = {
            "name": collection.name,
            "batches": collection_batches,
            "method_specs": collection_method_specs,
        }
        collections[collection.name] = {
            **collection_node,
            HASH_KEY: dehydrate_hashed(collection)[HASH_KEY],
        }

    return manifest


def write_manifest(path: str | Path | None = None) -> Path:
    if path is not None:
        destination = Path(path)
    else:
        from datetime import date

        today = date.today().strftime("%Y_%m_%d")
        destination = (
            Path(__file__).resolve().parent / "results" / f"manifest_{today}.json"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest_dict(), indent=2), encoding="utf-8")
    return destination


SIMULATION_SPECS = REGISTRY.simulations
BATCH_SPECS = REGISTRY.batches
COLLECTION_SPECS = REGISTRY.collections


if __name__ == "__main__":
    manifest_path = write_manifest()
    print(f"Manifest written to {manifest_path}")
