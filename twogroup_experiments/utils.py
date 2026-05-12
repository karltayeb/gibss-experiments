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


PIP_THRESHOLD_PLOT_DATA = "pip_threshold_plot_data.parquet"
CAUSAL_PIP_PLOT_DATA = "causal_pip_plot_data.parquet"
CS_COMPONENT_PLOT_DATA = "cs_component_plot_data.parquet"
CS_TRUTH_PLOT_DATA = "cs_truth_plot_data.parquet"
PLOT_DATA_FILENAMES = (
    PIP_THRESHOLD_PLOT_DATA,
    CAUSAL_PIP_PLOT_DATA,
    CS_COMPONENT_PLOT_DATA,
    CS_TRUTH_PLOT_DATA,
)
PIP_THRESHOLD_GRID = np.round(np.arange(0.001, 1.0, 0.001), 3)
CS_BETA_GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)
PIP_THRESHOLD_PLOT_BASE_SCHEMA: dict[str, pl.DataType] = {
    "replicate": pl.Int64,
    "method": pl.String,
    "threshold": pl.Float64,
    "method_spec": pl.String,
    "simulation_spec": pl.String,
    "pip_threshold": pl.Float64,
    "selected_total": pl.Int64,
    "selected_causal": pl.Int64,
    "power": pl.Float64,
    "fdp": pl.Float64,
    "n_exact": pl.Int64,
    "n_causal_exact": pl.Int64,
}
CAUSAL_PIP_PLOT_BASE_SCHEMA: dict[str, pl.DataType] = {
    "replicate": pl.Int64,
    "method": pl.String,
    "threshold": pl.Float64,
    "method_spec": pl.String,
    "simulation_spec": pl.String,
    "causal_variable": pl.Int64,
    "causal_pip": pl.Float64,
    "max_pip": pl.Float64,
}
CS_COMPONENT_PLOT_BASE_SCHEMA: dict[str, pl.DataType] = {
    "replicate": pl.Int64,
    "method": pl.String,
    "threshold": pl.Float64,
    "method_spec": pl.String,
    "simulation_spec": pl.String,
    "component": pl.Int64,
    "ordered_pips": pl.List(pl.Float64),
    "betas": pl.List(pl.Float64),
    "cs_sizes": pl.List(pl.Int64),
    "ser_log_bf": pl.Float64,
}
CS_TRUTH_PLOT_BASE_SCHEMA: dict[str, pl.DataType] = {
    "replicate": pl.Int64,
    "method": pl.String,
    "threshold": pl.Float64,
    "method_spec": pl.String,
    "simulation_spec": pl.String,
    "causal_variable": pl.Int64,
    "component": pl.Int64,
    "causal_rank": pl.Int64,
    "betas": pl.List(pl.Float64),
    "covered": pl.List(pl.Boolean),
}


def manifest_dict() -> dict[str, object]:
    from config import manifest_dict as config_manifest_dict

    return config_manifest_dict()


def materialize_manifest(path: str | Path) -> None:
    from config import write_manifest

    write_manifest(path)


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


def build_plot_data_frames(
    fits_df: pl.DataFrame,
    simulations_df: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    joined = fits_df.join(simulations_df, on="replicate", how="left")
    pip_threshold_rows: list[dict[str, Any]] = []
    causal_pip_rows: list[dict[str, Any]] = []
    cs_component_rows: list[dict[str, Any]] = []
    cs_truth_rows: list[dict[str, Any]] = []

    grid_ints = np.arange(1, 1000, dtype=int)
    threshold_grid = grid_ints / 1000.0

    for row in joined.iter_rows(named=True):
        alpha = np.asarray(row["ser_posterior"]["alpha"], dtype=float)
        causal_indices = np.asarray(row["simulation"]["causal_indices"], dtype=int)
        pip_int = np.clip(np.rint(alpha * 1000).astype(int), 1, 999)
        total_counts = np.bincount(pip_int, minlength=1000)
        causal_counts = np.bincount(pip_int[causal_indices], minlength=1000)
        total_tail = np.cumsum(total_counts[::-1])[::-1]
        causal_tail = np.cumsum(causal_counts[::-1])[::-1]
        n_causal = max(int(causal_indices.size), 1)

        for grid_int, pip_threshold in zip(grid_ints, threshold_grid, strict=True):
            selected_total = int(total_tail[grid_int])
            selected_causal = int(causal_tail[grid_int])
            pip_threshold_rows.append(
                {
                    "replicate": int(row["replicate"]),
                    "method": row["method"],
                    "threshold": row["threshold"],
                    "method_spec": row["method_spec"],
                    "simulation_spec": row["simulation_spec"],
                    "pip_threshold": float(pip_threshold),
                    "selected_total": selected_total,
                    "selected_causal": selected_causal,
                    "power": float(selected_causal / n_causal),
                    "fdp": float(
                        (selected_total - selected_causal) / selected_total
                    )
                    if selected_total > 0
                    else 0.0,
                    "n_exact": int(total_counts[grid_int]),
                    "n_causal_exact": int(causal_counts[grid_int]),
                }
            )

        for causal_variable in causal_indices.tolist():
            causal_pip_rows.append(
                {
                    "replicate": int(row["replicate"]),
                    "method": row["method"],
                    "threshold": row["threshold"],
                    "method_spec": row["method_spec"],
                    "simulation_spec": row["simulation_spec"],
                    "causal_variable": int(causal_variable),
                    "causal_pip": float(alpha[causal_variable]),
                    "max_pip": float(row["fit_summary"]["max_pip"]),
                }
            )

        order = np.argsort(-alpha)
        ordered_pips = alpha[order]
        cumulative_pips = np.cumsum(ordered_pips)
        cs_sizes = (
            np.searchsorted(cumulative_pips, CS_BETA_GRID, side="left") + 1
        ).astype(int)
        ser_log_bf = float(row["ser_posterior"]["ser_log_bf"])
        cs_component_rows.append(
            {
                "replicate": int(row["replicate"]),
                "method": row["method"],
                "threshold": row["threshold"],
                "method_spec": row["method_spec"],
                "simulation_spec": row["simulation_spec"],
                "component": 0,
                "ordered_pips": ordered_pips.tolist(),
                "betas": CS_BETA_GRID.tolist(),
                "cs_sizes": cs_sizes.tolist(),
                "ser_log_bf": ser_log_bf,
            }
        )

        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(order.size, dtype=int)
        for causal_variable in causal_indices.tolist():
            causal_rank = int(inverse_order[int(causal_variable)]) + 1
            cs_truth_rows.append(
                {
                    "replicate": int(row["replicate"]),
                    "method": row["method"],
                    "threshold": row["threshold"],
                    "method_spec": row["method_spec"],
                    "simulation_spec": row["simulation_spec"],
                    "causal_variable": int(causal_variable),
                    "component": 0,
                    "causal_rank": causal_rank,
                    "betas": CS_BETA_GRID.tolist(),
                    "covered": (causal_rank <= cs_sizes).tolist(),
                }
            )

    return {
        "pip_threshold_plot_data": _plot_frame(
            pip_threshold_rows, PIP_THRESHOLD_PLOT_BASE_SCHEMA
        ),
        "causal_pip_plot_data": _plot_frame(
            causal_pip_rows, CAUSAL_PIP_PLOT_BASE_SCHEMA
        ),
        "cs_component_plot_data": _plot_frame(
            cs_component_rows, CS_COMPONENT_PLOT_BASE_SCHEMA
        ),
        "cs_truth_plot_data": _plot_frame(
            cs_truth_rows, CS_TRUTH_PLOT_BASE_SCHEMA
        ),
    }


def _plot_frame(
    rows: list[dict[str, Any]], schema: dict[str, pl.DataType]
) -> pl.DataFrame:
    return pl.from_dicts(rows, schema=schema)


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


def symlink_plot_data_outputs(
    source_root: str | Path,
    target_root: str | Path,
) -> None:
    source = Path(source_root)
    target = Path(target_root)
    for filename in PLOT_DATA_FILENAMES:
        if (source / filename).exists():
            symlink_output(str(source / filename), str(target / filename))
