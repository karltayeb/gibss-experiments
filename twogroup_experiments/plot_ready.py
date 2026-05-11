from __future__ import annotations

import json
from pathlib import Path
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
