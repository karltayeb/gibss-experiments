from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from utils import attach_spec_metadata
from viz3_utils import method_metadata_from_method_spec_json, make_method_display_label


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
                simulation_spec_node=batch["simulation_spec"],
            )
            fits_df = fits_df.with_columns(pl.lit(batch_hash).alias("batch_hash"))
            frames.append(fits_df)
    return pl.concat(frames)


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
            }
        )
    return pl.from_dicts(rows)


def build_simulation_metadata(collection: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for batch in collection["batches"]:
        rows.append(
            {
                "batch_hash": batch["hash"],
                "batch_name": batch["name"],
                "simulation_spec": json.dumps(batch["simulation_spec"], sort_keys=True),
                "simulation_name": batch["simulation_spec"]["fields"]["name"],
            }
        )
    return pl.from_dicts(rows)


def build_sample_metadata(
    collection_batches: list[dict[str, Any]],
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    rows = []
    for batch in collection_batches:
        batch_hash = batch["hash"]
        batch_name = batch["name"]
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
        batch_hash = batch["__spec_hash__"]
        batch_name = batch["name"]
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


def summarize_pip_calibration_per_sample(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Build one row per sample_id x method x threshold x pip_bin_index."""
    fits_with_sample_id = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for row in fits_with_sample_id.iter_rows(named=True):
        batch_hash = row["batch_hash"]
        replicate = row["replicate"]
        alpha = np.asarray(row["ser_posterior"]["alpha"], dtype=float)

        sim_df = simulations_by_batch[batch_hash]
        sim_row = sim_df.filter(pl.col("replicate") == replicate).row(0, named=True)
        causal_indices = np.asarray(sim_row["simulation"]["causal_indices"], dtype=int)

        bin_indices = np.clip((alpha * 20).astype(int), 0, 19)
        is_causal = np.zeros(len(alpha), dtype=bool)
        is_causal[causal_indices] = True

        for bin_idx in range(20):
            mask = bin_indices == bin_idx
            n_exact = int(mask.sum())
            n_causal_exact = int((mask & is_causal).sum())
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "method": row["method"],
                    "threshold": row["threshold"],
                    "pip_bin_index": bin_idx,
                    "n_exact": n_exact,
                    "n_causal_exact": n_causal_exact,
                }
            )
    return pl.from_dicts(
        rows,
        schema={
            "sample_id": pl.String,
            "method": pl.String,
            "threshold": pl.Float64,
            "pip_bin_index": pl.Int64,
            "n_exact": pl.Int64,
            "n_causal_exact": pl.Int64,
        },
    )


def aggregate_pip_calibration(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold", "pip_bin_index")
        .agg(
            pl.col("n_exact").sum().alias("n_total"),
            pl.col("n_causal_exact").sum().alias("n_causal"),
        )
        .with_columns(
            (pl.col("pip_bin_index") * 0.05).alias("pip_left"),
            ((pl.col("pip_bin_index") + 1) * 0.05).alias("pip_right"),
            ((pl.col("pip_bin_index") + 0.5) * 0.05).alias("pip_mid"),
            pl.when(pl.col("n_total") > 0)
                .then(pl.col("n_causal") / pl.col("n_total"))
                .otherwise(None)
                .alias("empirical_rate"),
        )
        .select(
            "method",
            "threshold",
            "pip_bin_index",
            "pip_left",
            "pip_right",
            "pip_mid",
            "n_total",
            "n_causal",
            "empirical_rate",
        )
        .sort("method", "threshold", "pip_bin_index")
    )


_PIP_THRESHOLD_GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]


def summarize_power_fdp_per_sample(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
    simulations_by_batch: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Build one row per sample_id x method x threshold x pip_threshold."""
    fits_with_sample_id = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for row in fits_with_sample_id.iter_rows(named=True):
        batch_hash = row["batch_hash"]
        replicate = row["replicate"]
        alpha = np.asarray(row["ser_posterior"]["alpha"], dtype=float)

        sim_df = simulations_by_batch[batch_hash]
        sim_row = sim_df.filter(pl.col("replicate") == replicate).row(0, named=True)
        causal_indices = np.asarray(sim_row["simulation"]["causal_indices"], dtype=int)
        n_causal = max(len(causal_indices), 1)

        for pip_threshold in _PIP_THRESHOLD_GRID:
            selected = alpha >= pip_threshold
            selected_causal = selected[causal_indices].sum()
            selected_total = selected.sum()
            power = float(selected_causal / n_causal)
            fdp = float(
                (selected_total - selected_causal) / max(selected_total, 1)
            )
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "method": row["method"],
                    "threshold": row["threshold"],
                    "pip_threshold": float(pip_threshold),
                    "power": power,
                    "fdp": fdp,
                }
            )
    return pl.from_dicts(
        rows,
        schema={
            "sample_id": pl.String,
            "method": pl.String,
            "threshold": pl.Float64,
            "pip_threshold": pl.Float64,
            "power": pl.Float64,
            "fdp": pl.Float64,
        },
    )


def aggregate_power_fdp(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold", "pip_threshold")
        .agg(
            pl.col("power").mean().alias("power"),
            pl.col("fdp").mean().alias("fdp"),
        )
        .sort("method", "threshold", "pip_threshold")
    )


def summarize_causal_pip_per_sample(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Build one row per sample_id x method x threshold with causal PIP."""
    fits_with_sample_id = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    return fits_with_sample_id.select(
        "sample_id",
        "method",
        "threshold",
        pl.col("fit_summary").struct.field("causal_pip").alias("mean_causal_pip"),
    )


def aggregate_causal_pip(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold")
        .agg(pl.col("mean_causal_pip").mean().alias("mean_causal_pip"))
        .sort("method", "threshold")
    )


def summarize_cs_metrics_per_sample(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Build one row per sample_id x method x threshold x metric."""
    fits_with_sample_id = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for row in fits_with_sample_id.iter_rows(named=True):
        causal_in_cs = bool(row["credible_set"]["causal_in_cs"])
        cs_size = int(row["credible_set"]["cs_size"])
        base = {
            "sample_id": row["sample_id"],
            "method": row["method"],
            "threshold": row["threshold"],
        }
        rows.append({**base, "metric": "Power", "value": float(causal_in_cs)})
        rows.append({**base, "metric": "Coverage", "value": float(causal_in_cs)})
        rows.append({**base, "metric": "CS Size", "value": float(cs_size)})
    return pl.from_dicts(
        rows,
        schema={
            "sample_id": pl.String,
            "method": pl.String,
            "threshold": pl.Float64,
            "metric": pl.String,
            "value": pl.Float64,
        },
    )


def aggregate_cs_summary(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold", "metric")
        .agg(pl.col("value").mean().alias("value"))
        .sort("method", "threshold", "metric")
    )


def summarize_cs_size_histogram_observations(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Build raw method x threshold x cs_size observations."""
    fits_with_sample_id = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    return fits_with_sample_id.select(
        "method",
        "threshold",
        pl.col("credible_set").struct.field("cs_size").alias("cs_size"),
    )


def finalize_cs_size_histogram(observations: pl.DataFrame) -> pl.DataFrame:
    return observations.select("method", "threshold", "cs_size").sort(
        "method", "threshold", "cs_size"
    )


def summarize_ser_log_bf_histogram_observations(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Build raw method x threshold x ser_log_bf observations."""
    fits_with_sample_id = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    return fits_with_sample_id.select(
        "method",
        "threshold",
        pl.col("ser_posterior").struct.field("ser_log_bf").alias("ser_log_bf"),
    )


def finalize_ser_log_bf_histogram(observations: pl.DataFrame) -> pl.DataFrame:
    return observations.select("method", "threshold", "ser_log_bf").sort(
        "method", "threshold", "ser_log_bf"
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
        "pip_calibration",
        "power_fdp",
        "causal_pip",
        "cs_summary",
        "cs_size_histogram",
        "ser_log_bf_histogram",
    ]
    return {name: pl.read_parquet(plot_ready_dir / f"{name}.parquet") for name in names}
