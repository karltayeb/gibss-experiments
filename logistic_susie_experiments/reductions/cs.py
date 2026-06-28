import numpy as np
import polars as pl

from reductions import ReductionContext


def build_cs_plot_data(
    fits_df: pl.DataFrame,
    sample_metadata: pl.DataFrame,
    simulations_df: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold, l). Arrays for CS sweep at each beta."""
    from utils import CS_BETA_GRID

    empty_schema = {
        "sample_id": pl.String, "batch_hash": pl.String, "method": pl.String, "threshold": pl.Float64,
        "l": pl.Int64, "ser_log_bf": pl.Float64,
        "n_features": pl.Int64,
        "causal_indices": pl.List(pl.Int64), "causal_alpha": pl.List(pl.Float64),
        "rank_of_causal": pl.List(pl.Int64),
        "mass_above_causal": pl.List(pl.Float64), "cs_sizes": pl.List(pl.Int64),
        "cs_causal_radius": pl.List(pl.List(pl.Float64)),
    }
    fits_with_sid = fits_df.join(
        sample_metadata.select("sample_id", "batch_hash", "replicate"),
        on=["batch_hash", "replicate"],
        how="left",
    )
    rows: list[dict] = []
    for row in fits_with_sid.iter_rows(named=True):
        sim_df = simulations_df
        sim_row = sim_df.filter(pl.col("replicate") == row["replicate"]).row(0, named=True)
        sim_struct = sim_row["simulation"]
        corr_by_causal = {
            int(causal_idx): [float(v) for v in corr]
            for causal_idx, corr in zip(
                sim_struct["causal_indices"],
                sim_struct["correlation_with_causal"],
            )
        }
        for l, effect in enumerate(row["single_effects"]):
            alpha = np.asarray(effect["alpha"], dtype=float)
            alpha = alpha / alpha.sum()
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
            n_feat = len(alpha)
            cs_sizes = [
                min(int(np.searchsorted(cumulative, float(beta), side="left") + 1), n_feat)
                for beta in CS_BETA_GRID
            ]
            # Causal radius = correlation-distance from the causal (the "center")
            # to the CS's least-correlated member (its furthest edge):
            #   r_min = min_{m in CS} |corr(m, causal)|  ->  radius = sqrt(1 - r_min^2)
            # 0 = every member tightly correlated with the truth (tight CS); larger
            # = the CS reaches a feature nearly uncorrelated with the causal.
            # Only defined when the CS covers the causal (causal_rank < cs_size).
            cs_causal_radius: list[list[float | None]] = []
            for ci, causal_rank in zip(causal_indices, rank_of_causal):
                causal_corr = np.abs(np.asarray(corr_by_causal[int(ci)], dtype=float))
                radius_by_beta: list[float | None] = []
                for cs_size in cs_sizes:
                    if causal_rank < cs_size:
                        prefix = order[:cs_size]
                        r_min = float(np.min(causal_corr[prefix]))
                        radius_by_beta.append(float(np.sqrt(max(0.0, 1.0 - r_min * r_min))))
                    else:
                        radius_by_beta.append(None)
                cs_causal_radius.append(radius_by_beta)
            rows.append({
                "sample_id": row["sample_id"],
                "batch_hash": row["batch_hash"],
                "method": row["method"],
                "threshold": row["threshold"],
                "l": l,
                "ser_log_bf": float(effect["ser_log_bf"]),
                "n_features": len(alpha),
                "causal_indices": causal_indices,
                "causal_alpha": causal_alpha,
                "rank_of_causal": rank_of_causal,
                "mass_above_causal": mass_above_causal,
                "cs_sizes": cs_sizes,
                "cs_causal_radius": cs_causal_radius,
            })
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema)


def build(ctx: ReductionContext) -> pl.DataFrame:
    return build_cs_plot_data(ctx.fits, ctx.sample_metadata, ctx.sims)


if "snakemake" in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).parent.parent)
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    import polars as pl
    from reductions import ReductionContext
    bh = snakemake.wildcards.batch_hash
    fits = pl.read_parquet(snakemake.input.fits).with_columns(pl.lit(bh).alias("batch_hash"))
    ctx = ReductionContext(
        fits=fits,
        sims=pl.read_parquet(snakemake.input.sims),
        sample_metadata=pl.read_parquet(snakemake.input.sample_md),
        sim_coordinate=snakemake.params.coordinate,
    )
    build(ctx).write_parquet(snakemake.output[0])
