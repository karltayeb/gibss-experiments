"""Shared helpers for all family renderer modules."""
from __future__ import annotations

import polars as pl


def foreground_methods(method_metadata: pl.DataFrame, settings: dict) -> set[str]:
    requested = set(settings.get("method_filter", []))
    present = set(method_metadata["method"].to_list())
    return requested & present


def method_order(method_metadata: pl.DataFrame, foreground: set[str]) -> list[str]:
    return (
        method_metadata.filter(pl.col("method").is_in(foreground))
        .select("method", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method"])["method"]
        .to_list()
    )


def set_agg_facecolor(fig) -> None:
    for ax in fig.axes:
        ax.set_facecolor("#ddeeff")
