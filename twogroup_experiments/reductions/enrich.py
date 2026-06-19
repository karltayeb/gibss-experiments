from __future__ import annotations

import polars as pl

from reductions import ReductionContext


def build_enrich_plot_data(
    fits_df: pl.DataFrame,
    simulations_df: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (sample_id, method). Intercept + mu_at_causal estimates + true values.

    fits_df must carry a batch_hash column and be filtered to one twogroup method's fits.
    simulations_df is the single batch's simulations DataFrame.
    """
    schema = {
        "sample_id": pl.String,
        "batch_hash": pl.String,
        "method": pl.String,
        "est_intercept": pl.Float64,
        "mu_at_causal": pl.Float64,
        "true_intercept": pl.Float64,
        "true_effect": pl.Float64,
    }
    rows = []
    batch_hash = fits_df["batch_hash"][0]
    sims_df = (
        simulations_df
        .select("replicate", pl.col("simulation").struct.unnest())
        .select(
            "replicate",
            pl.col("causal_indices").list.get(0, null_on_oob=True).alias("causal_idx"),
            pl.col("causal_effects").list.get(0, null_on_oob=True).alias("true_effect"),
            pl.col("intercept").alias("true_intercept"),
        )
    )
    rep_to_sim = {r["replicate"]: r for r in sims_df.iter_rows(named=True)}

    for row in fits_df.select(["replicate", "method", "family_state", "single_effects"]).iter_rows(named=True):
        sim = rep_to_sim.get(row["replicate"])
        if sim is None:
            continue
        if sim["causal_idx"] is None:
            continue
        causal_idx = int(sim["causal_idx"])
        mu_list = row["single_effects"][0]["mu"]
        rows.append({
            "sample_id": f"{batch_hash}::{int(row['replicate'])}",
            "batch_hash": batch_hash,
            "method": row["method"],
            "est_intercept": float(row["family_state"]["intercept"]),
            "mu_at_causal": float(mu_list[causal_idx]) if causal_idx < len(mu_list) else None,
            "true_intercept": float(sim["true_intercept"]),
            "true_effect": float(sim["true_effect"]),
        })
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(rows, schema=schema)


def build(ctx: ReductionContext) -> pl.DataFrame:
    return build_enrich_plot_data(ctx.fits, ctx.sims)


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
