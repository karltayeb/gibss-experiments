from __future__ import annotations

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
            cs_causal_radius: list[list[float | None]] = []
            for ci, causal_rank in zip(causal_indices, rank_of_causal):
                causal_corr = np.abs(np.asarray(corr_by_causal[int(ci)], dtype=float))
                radius_by_beta: list[float | None] = []
                for cs_size in cs_sizes:
                    if causal_rank < cs_size:
                        prefix = order[:cs_size]
                        radius_by_beta.append(float(np.min(causal_corr[prefix])))
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
