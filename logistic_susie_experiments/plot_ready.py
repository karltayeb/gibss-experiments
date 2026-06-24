from __future__ import annotations

import polars as pl


def build_sample_metadata(batch_hash: str, simulations_df: pl.DataFrame) -> pl.DataFrame:
    rows = [
        {"sample_id": f"{batch_hash}::{int(rep)}", "batch_hash": batch_hash, "replicate": int(rep)}
        for rep in simulations_df["replicate"].to_list()
    ]
    return pl.from_dicts(rows)
