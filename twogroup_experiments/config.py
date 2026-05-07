from __future__ import annotations

from functools import partial

from gibss.distributions import Normal, PointMass

from twogroup_experiments.core import (
    COX_HEAVY,
    LOGISTIC_ORACLE,
    TWOGROUP_ORACLE,
    MethodSpec,
    SimulationSpec,
    c4_gene_sets_X,
    fit_cox_method,
    fit_logistic_method,
    fit_twogroup_method,
    hallmark_gene_sets_X,
    summarize_cox_method,
    summarize_logistic_method,
    summarize_twogroup_method,
    uniform_single_effect,
)
from twogroup_experiments.utils import BatchSpec, CollectionSpec

__all__ = [
    "THRESHOLDS",
    "HALLMARK_REPLICATES_PER_BATCH",
    "HALLMARK_N_BATCHES",
    "C4_REPLICATES_PER_BATCH",
    "C4_N_BATCHES",
    "SIMULATION_SPECS",
    "DEFAULT_METHOD_SPECS",
    "HALLMARK_BATCH_SPECS",
    "C4_BATCH_SPECS",
    "BATCH_SPECS",
    "COLLECTION_SPECS",
]

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

HALLMARK_REPLICATES_PER_BATCH = 50
HALLMARK_N_BATCHES = 2
C4_REPLICATES_PER_BATCH = 10
C4_N_BATCHES = 10
BASE_SEED = 20260501

F0 = PointMass(0.0)


F1Alpha = Normal(loc=0.5, scale=0.1, estimate_loc=False, estimate_scale=False)
F1A = Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False)
F1B = Normal(loc=1.5, scale=0.1, estimate_loc=False, estimate_scale=False)
F1C = Normal(loc=2.0, scale=0.1, estimate_loc=False, estimate_scale=False)
F1D = Normal(loc=2.5, scale=0.1, estimate_loc=False, estimate_scale=False)
F1E = Normal(loc=3.0, scale=0.1, estimate_loc=False, estimate_scale=False)

F2A = Normal(loc=0.0, scale=0.5, estimate_loc=False, estimate_scale=False)
F2B = Normal(loc=0.0, scale=1.0, estimate_loc=False, estimate_scale=False)
F2C = Normal(loc=0.0, scale=2.0, estimate_loc=False, estimate_scale=False)
F2C2 = Normal(loc=0.0, scale=3.0, estimate_loc=False, estimate_scale=False)
F2D = Normal(loc=0.0, scale=4.0, estimate_loc=False, estimate_scale=False)
F2E = Normal(loc=0.0, scale=5.0, estimate_loc=False, estimate_scale=False)
F2F = Normal(loc=0.0, scale=6.0, estimate_loc=False, estimate_scale=False)

F1INIT = Normal(loc=0.0, scale=1.0, estimate_loc=True, estimate_scale=True)


def _logistic_threshold_method_spec(threshold: float) -> MethodSpec:
    return MethodSpec(
        name="logistic_threshold",
        fit_function=fit_logistic_method,
        summarize_function=summarize_logistic_method,
        kwargs={
            "response_source": "score_threshold",
            "threshold": float(threshold),
        },
    )


def _cox_light_threshold_method_spec(threshold: float) -> MethodSpec:
    return MethodSpec(
        name="cox_light_threshold",
        fit_function=fit_cox_method,
        summarize_function=summarize_cox_method,
        kwargs={
            "threshold": float(threshold),
            "time_sign": -1.0,
        },
    )


TWOGROUP = MethodSpec(
    name="twogroup",
    fit_function=fit_twogroup_method,
    summarize_function=summarize_twogroup_method,
    kwargs={"f1": F1INIT},
)

DEFAULT_METHOD_SPECS = (
    COX_HEAVY,
    LOGISTIC_ORACLE,
    TWOGROUP_ORACLE,
    TWOGROUP,
    *tuple(_cox_light_threshold_method_spec(threshold) for threshold in THRESHOLDS),
    *tuple(_logistic_threshold_method_spec(threshold) for threshold in THRESHOLDS),
)


def _simulation_spec(
    *,
    name: str,
    design_sampler,
    causal_effect: float,
    intercept: float,
    f1,
) -> SimulationSpec:
    return SimulationSpec(
        name=name,
        design_sampler=design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=causal_effect),
        intercept=intercept,
        f0=F0,
        f1=f1,
        base_seed=BASE_SEED,
    )


SIMULATION_SPECS = (
    _simulation_spec(
        name="hallmark_ser_nonlocal_alpha",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1Alpha,
    ),
    _simulation_spec(
        name="hallmark_ser_nonlocal_a",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1A,
    ),
    _simulation_spec(
        name="hallmark_ser_nonlocal_b",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1B,
    ),
    _simulation_spec(
        name="hallmark_ser_nonlocal_c",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1C,
    ),
    _simulation_spec(
        name="hallmark_ser_nonlocal_d",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1D,
    ),
    _simulation_spec(
        name="hallmark_ser_local_a",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2A,
    ),
    _simulation_spec(
        name="hallmark_ser_local_b",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2B,
    ),
    _simulation_spec(
        name="hallmark_ser_local_c",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2C,
    ),
    _simulation_spec(
        name="hallmark_ser_local_d",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2D,
    ),
    _simulation_spec(
        name="hallmark_ser_nonlocal_a_dep",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F1A,
    ),
    _simulation_spec(
        name="hallmark_ser_nonlocal_b_dep",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F1B,
    ),
    _simulation_spec(
        name="hallmark_ser_nonlocal_c_dep",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F1C,
    ),
    _simulation_spec(
        name="hallmark_ser_nonlocal_d_dep",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F1D,
    ),
    _simulation_spec(
        name="hallmark_ser_local_a_dep",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F2A,
    ),
    _simulation_spec(
        name="hallmark_ser_local_b_dep",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F2B,
    ),
    _simulation_spec(
        name="hallmark_ser_local_c_dep",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F2C,
    ),
    _simulation_spec(
        name="hallmark_ser_local_d_dep",
        design_sampler=hallmark_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F2D,
    ),
    _simulation_spec(
        name="c4_ser_nonlocal_a",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1A,
    ),
    _simulation_spec(
        name="c4_ser_nonlocal_b",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1B,
    ),
    _simulation_spec(
        name="c4_ser_nonlocal_c",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1C,
    ),
    _simulation_spec(
        name="c4_ser_nonlocal_d",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F1D,
    ),
    _simulation_spec(
        name="c4_ser_local_a",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2A,
    ),
    _simulation_spec(
        name="c4_ser_local_b",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2B,
    ),
    _simulation_spec(
        name="c4_ser_local_c",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2C,
    ),
    _simulation_spec(
        name="c4_ser_local_c2",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2C2,
    ),
    _simulation_spec(
        name="c4_ser_local_d",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2D,
    ),
    _simulation_spec(
        name="c4_ser_local_e",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2E,
    ),
    _simulation_spec(
        name="c4_ser_local_f",
        design_sampler=c4_gene_sets_X,
        causal_effect=2.0,
        intercept=-2.0,
        f1=F2F,
    ),
    _simulation_spec(
        name="c4_ser_nonlocal_a_dep",
        design_sampler=c4_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F1A,
    ),
    _simulation_spec(
        name="c4_ser_nonlocal_b_dep",
        design_sampler=c4_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F1B,
    ),
    _simulation_spec(
        name="c4_ser_nonlocal_c_dep",
        design_sampler=c4_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F1C,
    ),
    _simulation_spec(
        name="c4_ser_nonlocal_d_dep",
        design_sampler=c4_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F1D,
    ),
    _simulation_spec(
        name="c4_ser_local_a_dep",
        design_sampler=c4_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F2A,
    ),
    _simulation_spec(
        name="c4_ser_local_b_dep",
        design_sampler=c4_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F2B,
    ),
    _simulation_spec(
        name="c4_ser_local_c_dep",
        design_sampler=c4_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F2C,
    ),
    _simulation_spec(
        name="c4_ser_local_d_dep",
        design_sampler=c4_gene_sets_X,
        causal_effect=-2.0,
        intercept=-1.0,
        f1=F2D,
    ),
)

HALLMARK_SIMULATION_SPECS = tuple(
    simulation_spec
    for simulation_spec in SIMULATION_SPECS
    if simulation_spec.name.startswith("hallmark_")
)

C4_SIMULATION_SPECS = tuple(
    simulation_spec
    for simulation_spec in SIMULATION_SPECS
    if simulation_spec.name.startswith("c4_")
)


def _batch_specs_for_simulation(
    simulation_spec: SimulationSpec,
    *,
    replicates_per_batch: int,
    n_batches: int,
) -> tuple[BatchSpec, ...]:
    return tuple(
        BatchSpec(
            name=f"{simulation_spec.name}__batch{batch_index}",
            simulation_spec=simulation_spec,
            replicates=tuple(
                range(
                    batch_index * replicates_per_batch,
                    (batch_index + 1) * replicates_per_batch,
                )
            ),
        )
        for batch_index in range(n_batches)
    )


HALLMARK_BATCH_SPECS = tuple(
    batch_spec
    for simulation_spec in HALLMARK_SIMULATION_SPECS
    for batch_spec in _batch_specs_for_simulation(
        simulation_spec,
        replicates_per_batch=HALLMARK_REPLICATES_PER_BATCH,
        n_batches=HALLMARK_N_BATCHES,
    )
)

C4_BATCH_SPECS = tuple(
    batch_spec
    for simulation_spec in C4_SIMULATION_SPECS
    for batch_spec in _batch_specs_for_simulation(
        simulation_spec,
        replicates_per_batch=C4_REPLICATES_PER_BATCH,
        n_batches=C4_N_BATCHES,
    )
)

BATCH_SPECS = (
    HALLMARK_BATCH_SPECS
    + C4_BATCH_SPECS
    + (
        BatchSpec(
            name="tiny_test",
            simulation_spec=next(
                spec for spec in SIMULATION_SPECS if spec.name == "hallmark_ser_local_a"
            ),
            replicates=(0,),
        ),
    )
)


#
def _collection(
    name: str,
    *simulation_names: str,
    method_specs: tuple[MethodSpec, ...] = DEFAULT_METHOD_SPECS,
) -> CollectionSpec:
    return CollectionSpec(
        name=name,
        batches=tuple(
            batch
            for simulation_name in simulation_names
            for batch in BATCH_SPECS
            if batch.simulation_spec.name == simulation_name
        ),
        method_specs=method_specs,
    )


COLLECTION_SPECS = (
    *(
        CollectionSpec(
            name=batch.name, batches=(batch,), method_specs=DEFAULT_METHOD_SPECS
        )
        for batch in BATCH_SPECS
        if batch.name != "tiny_test"
    ),
    CollectionSpec(
        name="tiny_test",
        batches=(next(batch for batch in BATCH_SPECS if batch.name == "tiny_test"),),
        method_specs=(LOGISTIC_ORACLE,),
    ),
    _collection(
        "hallmark_ser_nonlocal",
        "hallmark_ser_nonlocal_alpha",
        "hallmark_ser_nonlocal_a",
        "hallmark_ser_nonlocal_b",
        "hallmark_ser_nonlocal_c",
        "hallmark_ser_nonlocal_d",
    ),
    _collection(
        "hallmark_ser_local",
        "hallmark_ser_local_a",
        "hallmark_ser_local_b",
        "hallmark_ser_local_c",
        "hallmark_ser_local_d",
    ),
    _collection(
        "hallmark_ser_nonlocal_dep",
        "hallmark_ser_nonlocal_a_dep",
        "hallmark_ser_nonlocal_b_dep",
        "hallmark_ser_nonlocal_c_dep",
        "hallmark_ser_nonlocal_d_dep",
    ),
    _collection(
        "hallmark_ser_local_dep",
        "hallmark_ser_local_a_dep",
        "hallmark_ser_local_b_dep",
        "hallmark_ser_local_c_dep",
        "hallmark_ser_local_d_dep",
    ),
    _collection(
        "hallmark_ser_all",
        "hallmark_ser_local_a",
        "hallmark_ser_local_b",
        "hallmark_ser_local_c",
        "hallmark_ser_local_d",
        "hallmark_ser_nonlocal_a",
        "hallmark_ser_nonlocal_b",
        "hallmark_ser_nonlocal_c",
        "hallmark_ser_nonlocal_d",
        "hallmark_ser_local_a_dep",
        "hallmark_ser_local_b_dep",
        "hallmark_ser_local_c_dep",
        "hallmark_ser_local_d_dep",
        "hallmark_ser_nonlocal_a_dep",
        "hallmark_ser_nonlocal_b_dep",
        "hallmark_ser_nonlocal_c_dep",
        "hallmark_ser_nonlocal_d_dep",
    ),
    _collection("hallmark_ser_local_c", "hallmark_ser_local_c"),
    _collection(
        "c4_ser_nonlocal",
        "c4_ser_nonlocal_alpha",
        "c4_ser_nonlocal_a",
        "c4_ser_nonlocal_b",
        "c4_ser_nonlocal_c",
        "c4_ser_nonlocal_d",
    ),
    _collection(
        "c4_ser_local",
        "c4_ser_local_a",
        "c4_ser_local_b",
        "c4_ser_local_c",
        "c4_ser_local_d",
    ),
    _collection(
        "c4_ser_nonlocal_dep",
        "c4_ser_nonlocal_a_dep",
        "c4_ser_nonlocal_b_dep",
        "c4_ser_nonlocal_c_dep",
        "c4_ser_nonlocal_d_dep",
    ),
    _collection(
        "c4_ser_local_dep",
        "c4_ser_local_a_dep",
        "c4_ser_local_b_dep",
        "c4_ser_local_c_dep",
        "c4_ser_local_d_dep",
    ),
    _collection(
        "c4_ser_all",
        "c4_ser_local_a",
        "c4_ser_local_b",
        "c4_ser_local_c",
        "c4_ser_local_d",
        "c4_ser_nonlocal_a",
        "c4_ser_nonlocal_b",
        "c4_ser_nonlocal_c",
        "c4_ser_nonlocal_d",
        "c4_ser_local_a_dep",
        "c4_ser_local_b_dep",
        "c4_ser_local_c_dep",
        "c4_ser_local_d_dep",
        "c4_ser_nonlocal_a_dep",
        "c4_ser_nonlocal_b_dep",
        "c4_ser_nonlocal_c_dep",
        "c4_ser_nonlocal_d_dep",
    ),
)
