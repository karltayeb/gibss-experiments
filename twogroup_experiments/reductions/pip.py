from __future__ import annotations

import numpy as np
import polars as pl

from reductions import ReductionContext

_N_PIP_BINS = 200
_PIP_BIN_WIDTH = 1.0 / _N_PIP_BINS  # 0.005


def build_pip_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
    simulations_df: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold). Bin arrays used to derive plots."""
    empty_schema = {
        "sample_id": pl.String, "batch_hash": pl.String, "method": pl.String, "threshold": pl.Float64,
        "causal_indices": pl.List(pl.Int64), "causal_pips": pl.List(pl.Float64),
        "pip_bin_counts": pl.List(pl.Int64), "pip_bin_causal_counts": pl.List(pl.Int64),
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

        sim_df = simulations_df
        sim_row = sim_df.filter(pl.col("replicate") == row["replicate"]).row(0, named=True)
        causal_indices = sorted(set(int(i) for i in sim_row["simulation"]["causal_indices"]))

        causal_pips = [float(marginal_pip[ci]) for ci in causal_indices]

        bin_idx = np.clip((marginal_pip * _N_PIP_BINS).astype(int), 0, _N_PIP_BINS - 1)
        is_causal = np.zeros(len(marginal_pip), dtype=bool)
        is_causal[causal_indices] = True
        pip_bin_counts = [int((bin_idx == b).sum()) for b in range(_N_PIP_BINS)]
        pip_bin_causal_counts = [int(((bin_idx == b) & is_causal).sum()) for b in range(_N_PIP_BINS)]

        rows.append({
            "sample_id": row["sample_id"],
            "batch_hash": row["batch_hash"],
            "method": row["method"],
            "threshold": row["threshold"],
            "causal_indices": causal_indices,
            "causal_pips": causal_pips,
            "pip_bin_counts": pip_bin_counts,
            "pip_bin_causal_counts": pip_bin_causal_counts,
        })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)


def build(ctx: ReductionContext) -> pl.DataFrame:
    return build_pip_plot_data(ctx.fits, ctx.sample_metadata, ctx.sims)


if "snakemake" in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).parent.parent)
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    import polars as pl
    from reductions import ReductionContext
    bh, mh = snakemake.wildcards.batch_hash, snakemake.wildcards.method_hash
    fits = pl.read_parquet(snakemake.input.fits).with_columns(pl.lit(bh).alias("batch_hash"))
    ctx = ReductionContext(
        fits=fits,
        sims=pl.read_parquet(snakemake.input.sims),
        sample_metadata=pl.read_parquet(snakemake.input.sample_md),
        sim_coordinate=snakemake.params.coordinate,
    )
    build(ctx).write_parquet(snakemake.output[0])
