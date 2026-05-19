from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from core import (
    HASH_KEY,
    MethodSpec,
    SimulationSpec,
    TwoGroupSimulation,
    dehydrate_hashed,
    dehydrate_node,
    run_method_spec,
    simulate,
    spec_hash,
    summarize_method_spec,
)


@dataclass(frozen=True)
class BatchSpec:
    name: str
    simulation_spec: SimulationSpec
    replicates: tuple[int, ...]


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    batches: tuple[BatchSpec, ...]
    method_specs: tuple[MethodSpec, ...]


CS_BETA_GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)


def manifest_dict() -> dict[str, object]:
    from config import manifest_dict as config_manifest_dict

    return config_manifest_dict()


def simulation_struct_without_x(simulation: TwoGroupSimulation) -> dict[str, Any]:
    return {
        "causal_indices": [int(idx) for idx in simulation.causal_indices],
        "causal_effects": [float(value) for value in simulation.causal_effects],
        "intercept": float(simulation.intercept),
        "b": simulation.b.tolist(),
        "z": simulation.z.tolist(),
        "theta": simulation.theta.tolist(),
        "thetahat": simulation.thetahat.tolist(),
        "se": simulation.se.tolist(),
        "f0": dehydrate_node(simulation.f0),
        "f1": dehydrate_node(simulation.f1),
    }


def simulate_batch(
    simulation_spec: SimulationSpec,
    *,
    replicates: Iterable[int],
) -> pl.DataFrame:
    replicate_ids = tuple(int(replicate) for replicate in replicates)
    if not replicate_ids:
        raise ValueError("simulate_batch requires at least one replicate.")

    rows: list[dict[str, Any]] = []
    total = len(replicate_ids)
    for index, replicate in enumerate(replicate_ids, start=1):
        if index == 1 or index == total or index % 5 == 0:
            print(f"[twogroup-experiments] simulate replicate {index}/{total}", flush=True)
        simulation = simulate(simulation_spec, replicate)
        rows.append(
            {
                "replicate": replicate,
                "simulation": simulation_struct_without_x(simulation),
            }
        )
    return pl.from_dicts(rows)


def fit_batch_method(
    simulation_spec: SimulationSpec,
    *,
    method_spec: MethodSpec,
    replicates: Iterable[int],
) -> pl.DataFrame:
    replicate_ids = tuple(int(replicate) for replicate in replicates)
    if not replicate_ids:
        raise ValueError("fit_batch_method requires at least one replicate.")

    rows: list[dict[str, Any]] = []
    total = len(replicate_ids)
    for index, replicate in enumerate(replicate_ids, start=1):
        if index == 1 or index == total or index % 5 == 0:
            print(
                f"[twogroup-experiments] fit {method_spec.name} replicate {index}/{total}",
                flush=True,
            )
        simulation = simulate(simulation_spec, replicate)
        fit_obj = run_method_spec(method_spec, simulation)
        rows.append(
            {
                "replicate": replicate,
                **summarize_method_spec(method_spec, fit_obj, simulation),
            }
        )
    return pl.from_dicts(rows)


def attach_spec_metadata(
    fits_df: pl.DataFrame,
    *,
    method_spec_node: dict[str, Any],
    simulation_spec_node: dict[str, Any],
) -> pl.DataFrame:
    return fits_df.with_columns(
        pl.lit(json.dumps(method_spec_node, sort_keys=True)).alias("method_spec"),
        pl.lit(json.dumps(simulation_spec_node, sort_keys=True)).alias("simulation_spec"),
    )


def write_parquet(df: pl.DataFrame, path_text: str) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _plain_python(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_python(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_plain_python(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_python(item) for item in value]
    return value


def write_yaml(data: dict[str, object], path_text: str) -> None:
    import yaml

    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_plain_python(data), handle, sort_keys=True)


def write_text(text: str, path_text: str) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def symlink_output(target_text: str, link_text: str) -> None:
    target = Path(target_text)
    link = Path(link_text)
    if not target.exists():
        raise FileNotFoundError(f"Cannot create alias symlink to missing target: {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    relative_target = os.path.relpath(target, start=link.parent)
    link.symlink_to(relative_target)


