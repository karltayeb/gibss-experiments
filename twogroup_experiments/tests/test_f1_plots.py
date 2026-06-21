from __future__ import annotations

import polars as pl

import viz_utils


def _f1_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "collection_name": ["coll-a", "coll-a"],
            "method": ["twogroup_oracle_L1", "twogroup_L1"],
            "method_display": ["Twogroup SER (Oracle)", "Twogroup SER"],
            "method_family": ["twogroup_oracle", "twogroup"],
            "f1_loc": [2.0, 1.8],
            "f1_scale": [0.1, 0.2],
            "true_f1_loc": [2.0, 2.0],
            "true_f1_scale": [0.1, 0.1],
            "est_intercept": [-2.0, -1.8],
            "mu_at_causal": [2.0, 1.7],
            "true_intercept": [-2.0, -2.0],
            "true_effect": [2.0, 2.0],
        }
    )


def test_f1_boxplot_includes_twogroup_oracle_legend_label():
    fig = viz_utils.render_f1_boxplot(
        _f1_rows(),
        collection_names=["coll-a"],
        method_order=["Twogroup SER (Oracle)", "Twogroup SER"],
    )

    labels = [text.get_text() for text in fig.axes[3].get_legend().get_texts()]

    assert "Twogroup SER (Oracle)" in labels


def test_f1_scatter_titles_use_oracle_display_label():
    fig = viz_utils.render_f1_scatter_chart(
        _f1_rows(),
        collection_names=["coll-a"],
        method_order=["Twogroup SER (Oracle)", "Twogroup SER"],
    )

    titles = [ax.get_title() for ax in fig.axes]

    assert "Twogroup SER (Oracle)" in titles


def test_f1_enrich_scatter_titles_use_oracle_display_label():
    fig = viz_utils.render_f1_enrich_scatter_chart(
        _f1_rows(),
        collection_names=["coll-a"],
        method_order=["Twogroup SER (Oracle)", "Twogroup SER"],
    )

    titles = [ax.get_title() for ax in fig.axes]

    assert "Twogroup SER (Oracle)" in titles
