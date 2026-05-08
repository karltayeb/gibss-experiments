from __future__ import annotations

from functools import partial
from typing import Iterable

from gibss.distributions import Normal

from core import SimulationSpec
from utils import BatchSpec


def format_float(value: float) -> str:
    return f"{float(value):.2f}"


def fixed_normal(*, loc: float, scale: float) -> Normal:
    return Normal(
        loc=float(loc),
        scale=float(scale),
        estimate_loc=False,
        estimate_scale=False,
    )


def make_simulation_spec(
    *,
    name: str,
    f0,
    base_seed: int,
    design_sampler,
    effect_sampler,
    intercept: float,
    f1,
) -> SimulationSpec:
    return SimulationSpec(
        name=name,
        design_sampler=design_sampler,
        effect_sampler=effect_sampler,
        intercept=float(intercept),
        f0=f0,
        f1=f1,
        base_seed=base_seed,
    )


def build_simulation_spec(
    *,
    design_name: str,
    regime_name: str,
    f1_name: str,
    separator: str,
    f0,
    base_seed: int,
    design_kwargs: dict[str, dict[str, object]],
    regime_kwargs: dict[str, dict[str, object]],
    f1_kwargs: dict[str, dict[str, object]],
) -> SimulationSpec:
    name = separator.join((design_name, regime_name, f1_name))
    kwargs = {
        **design_kwargs[design_name],
        **regime_kwargs[regime_name],
        **f1_kwargs[f1_name],
    }
    return make_simulation_spec(name=name, f0=f0, base_seed=base_seed, **kwargs)


def make_product_simulation_specs(
    *,
    design_names: Iterable[str],
    regime_names: Iterable[str],
    f1_names: Iterable[str],
    separator: str,
    f0,
    base_seed: int,
    design_kwargs: dict[str, dict[str, object]],
    regime_kwargs: dict[str, dict[str, object]],
    f1_kwargs: dict[str, dict[str, object]],
) -> tuple[SimulationSpec, ...]:
    return tuple(
        build_simulation_spec(
            design_name=design_name,
            regime_name=regime_name,
            f1_name=f1_name,
            separator=separator,
            f0=f0,
            base_seed=base_seed,
            design_kwargs=design_kwargs,
            regime_kwargs=regime_kwargs,
            f1_kwargs=f1_kwargs,
        )
        for design_name in design_names
        for regime_name in regime_names
        for f1_name in f1_names
    )


def batch_specs_for_simulation(
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
