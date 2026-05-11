from __future__ import annotations

import json
from typing import Any

import polars as pl

from viz3_utils import method_metadata_from_method_spec_json, make_method_display_label


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


def summarize_pip_calibration_per_sample(fits_df: pl.DataFrame, sample_metadata: pl.DataFrame) -> pl.DataFrame:
    # Build one row per sample_id x method x threshold x pip_bin_index.
    # stub — full implementation in per-sample summarizer (leave as NotImplementedError for now)
    raise NotImplementedError


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


def summarize_power_fdp_per_sample(fits_df: pl.DataFrame, sample_metadata: pl.DataFrame) -> pl.DataFrame:
    # Build one row per sample_id x method x threshold x pip_threshold.
    raise NotImplementedError


def aggregate_power_fdp(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold", "pip_threshold")
        .agg(
            pl.col("power").mean().alias("power"),
            pl.col("fdp").mean().alias("fdp"),
        )
        .sort("method", "threshold", "pip_threshold")
    )


def summarize_causal_pip_per_sample(fits_df: pl.DataFrame, sample_metadata: pl.DataFrame) -> pl.DataFrame:
    # Build one row per sample_id x method x threshold with sample-level mean causal PIP.
    raise NotImplementedError


def aggregate_causal_pip(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold")
        .agg(pl.col("mean_causal_pip").mean().alias("mean_causal_pip"))
        .sort("method", "threshold")
    )


def summarize_cs_metrics_per_sample(fits_df: pl.DataFrame, sample_metadata: pl.DataFrame) -> pl.DataFrame:
    # Build one row per sample_id x method x threshold x metric.
    raise NotImplementedError


def aggregate_cs_summary(per_sample: pl.DataFrame) -> pl.DataFrame:
    return (
        per_sample.group_by("method", "threshold", "metric")
        .agg(pl.col("value").mean().alias("value"))
        .sort("method", "threshold", "metric")
    )
