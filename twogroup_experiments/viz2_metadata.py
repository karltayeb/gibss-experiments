from __future__ import annotations

import json
from typing import Any

import polars as pl


def method_family_label_map() -> dict[str, str]:
    return {
        "logistic_threshold": "Logistic",
        "cox_light_threshold": "Cox Light",
        "twogroup": "Twogroup",
        "twogroup_oracle": "Twogroup",
        "logistic_oracle": "Logistic",
        "cox_heavy": "Cox Heavy",
    }


def method_family_display_order() -> list[str]:
    return [
        "logistic_oracle",
        "twogroup_oracle",
        "twogroup",
        "cox_heavy",
        "cox_light_threshold",
        "logistic_threshold",
    ]


def method_metadata_from_method_spec_json(method_spec_json: str) -> dict[str, object]:
    method_spec = json.loads(method_spec_json)
    name = str(method_spec["fields"]["name"])
    kwargs = dict(method_spec["fields"].get("kwargs", {}))
    L = int(kwargs.get("L", 1))
    method_family = name.rsplit("_L", 1)[0]
    is_thresholded = "threshold" in method_family
    is_oracle = "oracle" in method_family
    family_label = method_family_label_map().get(method_family, method_family)
    suffix = "SER" if L == 1 else f"SuSiE [L={L}]"
    oracle_label = "Oracle"
    if method_family == "twogroup_oracle":
        oracle_label = "oracle f1"
    return {
        "method_family": method_family,
        "L": L,
        "is_thresholded": is_thresholded,
        "is_oracle": is_oracle,
        "oracle_label": oracle_label,
        "method_label_base": f"{family_label} {suffix}",
    }


def make_method_display_label(
    method_label_base: str,
    threshold: float | None,
    is_thresholded: bool,
    is_oracle: bool,
    oracle_label: str = "Oracle",
) -> str:
    if is_oracle:
        return f"{method_label_base} ({oracle_label})"
    if is_thresholded and threshold is not None:
        return f"{method_label_base} (@{threshold:g})"
    return method_label_base


def add_plot_metadata_columns(plot_data: pl.DataFrame) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    metadata_rows = [
        {
            "method_spec": method_spec,
            **method_metadata_from_method_spec_json(method_spec),
        }
        for method_spec in plot_data.get_column("method_spec").unique().to_list()
    ]
    metadata_df = pl.from_dicts(metadata_rows)
    return plot_data.join(metadata_df, on="method_spec", how="left").with_columns(
        pl.when(pl.col("is_oracle"))
        .then(pl.format("{} ({})", pl.col("method_label_base"), pl.col("oracle_label")))
        .when(pl.col("is_thresholded") & pl.col("threshold").is_not_null())
        .then(pl.format("{} (@{})", pl.col("method_label_base"), pl.col("threshold")))
        .otherwise(pl.col("method_label_base"))
        .alias("method_display")
    )


def method_display_order() -> list[str]:
    families = method_family_display_order()
    return [f"{family}_L{L}" for family in families for L in (1, 5)]


def method_label_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for family in method_family_display_order():
        family_label = method_family_label_map().get(family, family)
        is_oracle = "oracle" in family
        oracle_label = "oracle f1" if family == "twogroup_oracle" else "Oracle"
        for L in (1, 5):
            suffix = "SER" if L == 1 else f"SuSiE [L={L}]"
            base = f"{family_label} {suffix}"
            key = f"{family}_L{L}"
            if is_oracle:
                result[key] = f"{base} ({oracle_label})"
            else:
                result[key] = base
    return result


def available_method_families(df: pl.DataFrame) -> list[str]:
    return sorted(df.get_column("method_family").unique().to_list())


def available_L_values(df: pl.DataFrame) -> list[int]:
    return sorted(int(value) for value in df.get_column("L").unique().to_list())


def selected_method_names(
    plot_data: pl.DataFrame, *, selected_method_families: list[str], selected_L: int
) -> set[str]:
    return set(
        plot_data.filter(
            pl.col("method_family").is_in(selected_method_families)
            & (pl.col("L") == selected_L)
        )
        .get_column("method")
        .unique()
        .to_list()
    )


def method_selection_proxy(
    plot_data: pl.DataFrame,
    *,
    selected_method_families: list[str],
    selected_L: int,
) -> set[str]:
    return selected_method_names(
        plot_data,
        selected_method_families=selected_method_families,
        selected_L=selected_L,
    )
