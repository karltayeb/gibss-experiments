from __future__ import annotations

from typing import Any

import polars as pl

from reductions import ReductionContext


def build_f1_plot_data(
    fits_df: pl.DataFrame,
    true_loc: float | None,
    true_scale: float | None,
) -> pl.DataFrame:
    """One row per (sample_id, method). f1 loc/scale + intercept estimates + true f1 values.

    fits_df must be filtered to one twogroup method's fits and carry a batch_hash column.
    true_loc and true_scale come from the simulation coordinate's signal.f1 distribution params.
    """
    schema = {
        "sample_id": pl.String,
        "batch_hash": pl.String,
        "method": pl.String,
        "f1_loc": pl.Float64,
        "f1_scale": pl.Float64,
        "est_intercept": pl.Float64,
        "true_f1_loc": pl.Float64,
        "true_f1_scale": pl.Float64,
    }
    rows = []
    batch_hash = fits_df["batch_hash"][0]

    for row in fits_df.select(["replicate", "method", "two_group_state", "family_state"]).iter_rows(named=True):
        f1 = row["two_group_state"]["f1"]
        rows.append({
            "sample_id": f"{batch_hash}::{int(row['replicate'])}",
            "batch_hash": batch_hash,
            "method": row["method"],
            "f1_loc": float(f1["loc"]) if f1["loc"] is not None else None,
            "f1_scale": float(f1["scale"]) if f1["scale"] is not None else None,
            "est_intercept": float(row["family_state"]["intercept"]),
            "true_f1_loc": true_loc,
            "true_f1_scale": true_scale,
        })
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(rows, schema=schema)


def build(ctx: ReductionContext) -> pl.DataFrame:
    f1_params = next(iter(ctx.sim_coordinate["signal"]["f1"].values()))  # inner value dict of {Normal: {loc, scale, ...}}
    true_loc = float(f1_params["loc"]) if f1_params.get("loc") is not None else None
    true_scale = float(f1_params["scale"]) if f1_params.get("scale") is not None else None
    return build_f1_plot_data(ctx.fits, true_loc=true_loc, true_scale=true_scale)
