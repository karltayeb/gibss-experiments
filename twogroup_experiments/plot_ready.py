from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

_HASH_KEY = "__spec_hash__"  # legacy key for old dehydrated nodes (P3.3 will remove)
from utils import attach_spec_metadata
from viz_utils import method_metadata_from_method_spec_json, make_method_display_label


def _batch_name(batch: dict) -> str:
    """Handle flat (name) and legacy nested (fields.name) batch nodes."""
    return batch.get("name") or batch["fields"]["name"]


def _batch_sim_node(batch: dict) -> dict:
    """Simulation spec node, handling both flat and legacy nested formats."""
    return batch.get("simulation_spec") or batch["fields"]["simulation_spec"]


def _batch_sim_name(batch: dict) -> str:
    return _batch_sim_node(batch)["fields"]["name"]


def load_collection_fits_with_specs(
    collection: dict[str, Any],
    results_root: str = "results",
) -> pl.DataFrame:
    """Load all fits for a collection, concatenated across batches x method_specs."""
    frames = []
    for batch in collection["batches"]:
        batch_hash = batch["__spec_hash__"]
        for method_spec in collection["method_specs"]:
            method_hash = method_spec["__spec_hash__"]
            fits_path = f"{results_root}/by_batch/{batch_hash}/fits/{method_hash}/fits.parquet"
            fits_df = attach_spec_metadata(
                pl.read_parquet(fits_path),
                method_spec_node=method_spec,
                simulation_spec_node=_batch_sim_node(batch),
            )
            fits_df = fits_df.with_columns(pl.lit(batch_hash).alias("batch_hash"))
            frames.append(fits_df.select(
                "method", "threshold", "method_spec", "batch_hash", "replicate",
                "single_effects", "fit_summary", "credible_sets",
            ))
    return pl.concat(frames, how="diagonal_relaxed")


def load_collection_simulations(
    collection: dict[str, Any],
    results_root: str = "results",
) -> dict[str, pl.DataFrame]:
    """Load simulations for each batch in a collection."""
    result = {}
    for batch in collection["batches"]:
        batch_hash = batch["__spec_hash__"]
        result[batch_hash] = pl.read_parquet(
            f"{results_root}/by_batch/{batch_hash}/simulations.parquet"
        )
    return result


def build_method_metadata(fits_df: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in (
        fits_df.select("method", "threshold", "method_spec")
        .unique()
        .sort("method", "threshold", nulls_last=True)
        .iter_rows(named=True)
    ):
        metadata = method_metadata_from_method_spec_json(row["method_spec"])
        rows.append(
            {
                "method": row["method"],
                "threshold": row["threshold"],
                "method_spec": row["method_spec"],
                **metadata,
                "method_display": make_method_display_label(
                    method_label_base=str(metadata["method_label_base"]),
                    threshold=row["threshold"],
                    is_thresholded=bool(metadata["is_thresholded"]),
                    is_oracle=bool(metadata["is_oracle"]),
                    oracle_label=str(metadata["oracle_label"]),
                ),
                "method_display_base": make_method_display_label(
                    method_label_base=str(metadata["method_label_base"]),
                    threshold=None,
                    is_thresholded=False,
                    is_oracle=bool(metadata["is_oracle"]),
                    oracle_label=str(metadata["oracle_label"]),
                ),
            }
        )
    return pl.from_dicts(rows)


def build_simulation_metadata(collection: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for batch in collection["batches"]:
        rows.append(
            {
                "batch_hash": batch[_HASH_KEY],
                "batch_name": _batch_name(batch),
                "simulation_spec": json.dumps(_batch_sim_node(batch), sort_keys=True),
                "simulation_name": _batch_sim_name(batch),
            }
        )
    return pl.from_dicts(rows)


def build_sample_metadata(batch_hash: str, simulations_df: pl.DataFrame) -> pl.DataFrame:
    rows = [
        {"sample_id": f"{batch_hash}::{int(rep)}", "batch_hash": batch_hash, "replicate": int(rep)}
        for rep in simulations_df["replicate"].to_list()
    ]
    return pl.from_dicts(rows)


def build_collection_yaml_node(
    name: str,
    batch_nodes: list[dict],
    method_nodes: list[dict],
) -> dict:
    return {
        "name": name,
        "batches": batch_nodes,
        "method_specs": method_nodes,
    }


def union_collection_yaml_nodes(
    name: str,
    collection_nodes: list[dict],
) -> dict:
    """Union multiple dehydrated collection specs, deduplicating by __spec_hash__."""
    seen_batches: dict[str, dict] = {}
    seen_methods: dict[str, dict] = {}
    for node in collection_nodes:
        for batch in node["batches"]:
            seen_batches.setdefault(batch[_HASH_KEY], batch)
        for method in node["method_specs"]:
            seen_methods.setdefault(method[_HASH_KEY], method)
    return build_collection_yaml_node(
        name=name,
        batch_nodes=list(seen_batches.values()),
        method_nodes=list(seen_methods.values()),
    )


def available_plot_ready_collections(alias_root: Path) -> list[str]:
    """Return collection aliases that have a plot_ready/ subdirectory."""
    return sorted(
        p.name
        for p in alias_root.iterdir()
        if p.is_dir() and (p / "plot_ready").is_dir() and any((p / "plot_ready").glob("*.parquet"))
    )


def load_plot_ready_collection(collection_root: Path) -> dict[str, pl.DataFrame]:
    """Load all plot_ready parquets for a collection into a dict."""
    plot_ready_dir = collection_root / "plot_ready"
    names = [
        "method_metadata",
        "simulation_metadata",
        "sample_metadata",
        "pip_plot_data",
        "cs_plot_data",
        "f1_plot_data",
        "enrich_plot_data",
    ]
    result = {}
    for name in names:
        path = plot_ready_dir / f"{name}.parquet"
        if path.exists():
            result[name] = pl.read_parquet(path)
    return result
