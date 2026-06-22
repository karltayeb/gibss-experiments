from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
import viz_utils
from utils import CS_BETA_GRID

NB = len(CS_BETA_GRID)


def _cs_plot_data() -> pl.DataFrame:
    # Two samples, one credible set each, causal at rank 1, CS size 3 at every
    # beta -> always covered. Sample B has ser_log_bf below the BF filter, so it
    # counts toward (unfiltered) coverage but not toward (filtered) power.
    def row(sample_id: str, ser_log_bf: float) -> dict:
        return {
            "collection_name": "c0",
            "sample_id": sample_id,
            "method": "cox",
            "threshold": None,
            "l": 0,
            "rank_of_causal": [1],
            "causal_indices": [7],
            "n_features": 10,
            "cs_sizes": [3] * NB,
            "cs_causal_radius": [[0.5] * NB],
            "ser_log_bf": ser_log_bf,
        }
    schema = {
        "collection_name": pl.String, "sample_id": pl.String, "method": pl.String,
        "threshold": pl.Float64, "l": pl.Int64, "rank_of_causal": pl.List(pl.Int64),
        "causal_indices": pl.List(pl.Int64), "n_features": pl.Int64,
        "cs_sizes": pl.List(pl.Int64), "cs_causal_radius": pl.List(pl.List(pl.Float64)),
        "ser_log_bf": pl.Float64,
    }
    return pl.from_dicts([row("A", 5.0), row("B", 0.0)], schema=schema)


def _method_meta() -> pl.DataFrame:
    return pl.DataFrame(
        {"method": ["cox"], "threshold": [None], "method_display": ["Cox"],
         "is_thresholded": [False]},
        schema={"method": pl.String, "threshold": pl.Float64,
                "method_display": pl.String, "is_thresholded": pl.Boolean},
    )


def test_radius_coverage_matches_size_coverage():
    cs = _cs_plot_data()
    meta = _method_meta()
    size = viz_utils.make_cs_coverage_size_curves(cs, meta, selected_methods={"cox"})
    radius = viz_utils.make_cs_radius_power_summary(
        cs, meta, selected_methods={"cox"}, max_cs_size=100, min_ser_log_bf=2.0,
    )
    joined = (
        radius.select("collection_name", "method", "threshold", "beta", "coverage")
        .rename({"coverage": "coverage_radius"})
        .join(
            size.select("collection_name", "method", "threshold", "beta", "coverage")
            .rename({"coverage": "coverage_size"}),
            on=["collection_name", "method", "threshold", "beta"],
            how="inner", nulls_equal=True,
        )
    )
    assert not joined.is_empty()
    # the two calibration panels must plot identical empirical coverage
    assert (joined["coverage_radius"] - joined["coverage_size"]).abs().max() < 1e-12
    # and coverage must not be stuck equal to power (the bug)
    pwr = radius["power"].max()
    cov = radius["coverage"].max()
    assert abs(cov - 1.0) < 1e-12  # both samples covered -> coverage 1.0
    assert abs(pwr - 0.5) < 1e-12  # only sample A passes BF filter -> power 0.5
