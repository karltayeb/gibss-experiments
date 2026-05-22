from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from core import HASH_KEY
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
                "batch_hash": batch[HASH_KEY],
                "batch_name": _batch_name(batch),
                "simulation_spec": json.dumps(_batch_sim_node(batch), sort_keys=True),
                "simulation_name": _batch_sim_name(batch),
            }
        )
    return pl.from_dicts(rows)


def build_sample_metadata(
    collection_batches: list[dict[str, Any]],
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    rows = []
    for batch in collection_batches:
        batch_hash = batch[HASH_KEY]
        batch_name = _batch_name(batch)
        for replicate in simulations_by_batch[batch_hash]["replicate"].to_list():
            rows.append(
                {
                    "sample_id": f"{batch_hash}::{int(replicate)}",
                    "batch_hash": batch_hash,
                    "batch_name": batch_name,
                    "replicate": int(replicate),
                }
            )
    return pl.from_dicts(rows)


def _build_sample_metadata_from_manifest(
    collection: dict[str, Any],
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Build sample metadata using __spec_hash__ key (for use with real manifest)."""
    rows = []
    for batch in collection["batches"]:
        batch_hash = batch[HASH_KEY]
        batch_name = _batch_name(batch)
        for replicate in simulations_by_batch[batch_hash]["replicate"].to_list():
            rows.append(
                {
                    "sample_id": f"{batch_hash}::{int(replicate)}",
                    "batch_hash": batch_hash,
                    "batch_name": batch_name,
                    "replicate": int(replicate),
                }
            )
    return pl.from_dicts(rows)


_PIP_THRESHOLD_GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]


def build_pip_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold). Arrays pre-aggregated for plot time."""
    empty_schema = {
        "sample_id": pl.String, "method": pl.String, "threshold": pl.Float64,
        "causal_indices": pl.List(pl.Int64), "causal_pips": pl.List(pl.Float64),
        "pip_bin_counts": pl.List(pl.Int64), "pip_bin_causal_counts": pl.List(pl.Int64),
        "power_at_threshold": pl.List(pl.Float64), "fdp_at_threshold": pl.List(pl.Float64),
    }
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        alphas = np.stack([np.asarray(e["alpha"], dtype=float) for e in row["single_effects"]])
        marginal_pip = 1.0 - np.prod(1.0 - alphas, axis=0)

        sim_df = simulations_by_batch[row["batch_hash"]]
        sim_row = sim_df.filter(pl.col("replicate") == row["replicate"]).row(0, named=True)
        causal_indices = sorted(set(int(i) for i in sim_row["simulation"]["causal_indices"]))

        causal_pips = [float(marginal_pip[ci]) for ci in causal_indices]

        bin_idx = np.clip((marginal_pip * 20).astype(int), 0, 19)
        is_causal = np.zeros(len(marginal_pip), dtype=bool)
        is_causal[causal_indices] = True
        pip_bin_counts = [int((bin_idx == b).sum()) for b in range(20)]
        pip_bin_causal_counts = [int(((bin_idx == b) & is_causal).sum()) for b in range(20)]

        n_causal = max(len(causal_indices), 1)
        power_at_threshold = []
        fdp_at_threshold = []
        for t in _PIP_THRESHOLD_GRID:
            selected = marginal_pip >= t
            sel_causal = int(selected[causal_indices].sum())
            sel_total = int(selected.sum())
            power_at_threshold.append(float(sel_causal / n_causal))
            fdp_at_threshold.append(float((sel_total - sel_causal) / max(sel_total, 1)))

        rows.append({
            "sample_id": row["sample_id"],
            "method": row["method"],
            "threshold": row["threshold"],
            "causal_indices": causal_indices,
            "causal_pips": causal_pips,
            "pip_bin_counts": pip_bin_counts,
            "pip_bin_causal_counts": pip_bin_causal_counts,
            "power_at_threshold": power_at_threshold,
            "fdp_at_threshold": fdp_at_threshold,
        })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)


def build_cs_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold, l). Arrays for CS sweep at each beta."""
    from utils import CS_BETA_GRID

    empty_schema = {
        "sample_id": pl.String, "method": pl.String, "threshold": pl.Float64,
        "l": pl.Int64, "ser_log_bf": pl.Float64,
        "causal_indices": pl.List(pl.Int64), "causal_alpha": pl.List(pl.Float64),
        "rank_of_causal": pl.List(pl.Int64),
        "mass_above_causal": pl.List(pl.Float64), "cs_sizes": pl.List(pl.Int64),
    }
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        for l, effect in enumerate(row["single_effects"]):
            alpha = np.asarray(effect["alpha"], dtype=float)
            cs_struct = row["credible_sets"][l]
            causal_indices = [int(ci) for ci in cs_struct["causal_indices"]]

            order = np.argsort(-alpha)
            cumulative = np.cumsum(alpha[order])
            rank_of = {int(feat): rk for rk, feat in enumerate(order.tolist())}

            causal_alpha = [float(alpha[ci]) for ci in causal_indices]
            rank_of_causal = [rank_of[ci] for ci in causal_indices]
            mass_above_causal = [
                float(cumulative[rk - 1]) if rk > 0 else 0.0
                for rk in rank_of_causal
            ]
            cs_sizes = [
                int(np.searchsorted(cumulative, float(beta), side="left") + 1)
                for beta in CS_BETA_GRID
            ]
            rows.append({
                "sample_id": row["sample_id"],
                "method": row["method"],
                "threshold": row["threshold"],
                "l": l,
                "ser_log_bf": float(effect["ser_log_bf"]),
                "causal_indices": causal_indices,
                "causal_alpha": causal_alpha,
                "rank_of_causal": rank_of_causal,
                "mass_above_causal": mass_above_causal,
                "cs_sizes": cs_sizes,
            })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)


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
            seen_batches.setdefault(batch[HASH_KEY], batch)
        for method in node["method_specs"]:
            seen_methods.setdefault(method[HASH_KEY], method)
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
    ]
    result = {}
    for name in names:
        path = plot_ready_dir / f"{name}.parquet"
        if path.exists():
            result[name] = pl.read_parquet(path)
    return result
