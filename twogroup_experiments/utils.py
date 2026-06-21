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
    SimulationSpec,
    TwoGroupSimulation,
    _distribution_struct,
    simulate,
    spec_hash,
)


@dataclass(frozen=True)
class BatchSpec:
    name: str
    simulation_spec: SimulationSpec
    replicates: tuple[int, ...]


CS_BETA_GRID = np.append(np.round(np.arange(0.01, 1.00, 0.01), 2), 1.0)


def manifest_dict() -> dict[str, object]:
    from experiments import loader
    cfg = loader.load_config()
    return loader.manifest_dict(cfg["library"], cfg)


def _as_dense_array(X: Any) -> np.ndarray:
    """Densify to a numpy float array, handling sparse designs (e.g. jax BCOO).

    ``np.asarray`` on a jax ``BCOO`` falls back to element-wise object iteration
    and effectively hangs, so densify via ``.todense()`` first when available.
    """
    if hasattr(X, "todense"):
        X = X.todense()
    return np.asarray(X, dtype=float)


def correlation_with_causal(X: np.ndarray, causal_indices: Iterable[int]) -> list[list[float]]:
    """Pearson correlations from each causal feature to every feature."""
    X_arr = _as_dense_array(X)
    centered = X_arr - X_arr.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    rows: list[list[float]] = []
    for causal_idx in causal_indices:
        ci = int(causal_idx)
        if norms[ci] == 0:
            corr = np.zeros(X_arr.shape[1], dtype=float)
        else:
            denom = norms[ci] * norms
            corr = np.divide(
                centered[:, ci] @ centered,
                denom,
                out=np.zeros(X_arr.shape[1], dtype=float),
                where=denom > 0,
            )
        corr[ci] = 1.0
        rows.append([float(v) for v in corr])
    return rows


def simulation_struct_without_x(simulation: TwoGroupSimulation) -> dict[str, Any]:
    return {
        "causal_indices": [int(idx) for idx in simulation.causal_indices],
        "causal_effects": [float(value) for value in simulation.causal_effects],
        "correlation_with_causal": correlation_with_causal(
            simulation.X,
            simulation.causal_indices,
        ),
        "intercept": float(simulation.intercept),
        "b": simulation.b.tolist(),
        "z": simulation.z.tolist(),
        "theta": simulation.theta.tolist(),
        "thetahat": simulation.thetahat.tolist(),
        "se": simulation.se.tolist(),
        "f0": _distribution_struct(simulation.f0),
        "f1": _distribution_struct(simulation.f1),
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
    method_coord: dict,
    replicates: Iterable[int],
) -> pl.DataFrame:
    from experiments.loader import run_method
    rows: list[dict[str, Any]] = []
    for replicate in (int(r) for r in replicates):
        sim = simulate(simulation_spec, replicate)
        rows.append({"replicate": replicate, **run_method(method_coord, sim)})
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
