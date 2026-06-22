from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import polars as pl
import viz_utils

_SCHEMA = {
    "simulation_name": pl.String, "method": pl.String, "method_display": pl.String,
    "method_display_base": pl.String, "threshold": pl.Float64, "mean_causal_pip": pl.Float64,
}


def _summary() -> pl.DataFrame:
    rows = []
    for t, v in [(1.0, 0.3), (2.0, 0.5), (3.0, 0.7)]:
        rows.append({
            "simulation_name": "s",
            "method": f"cox__threshold={t:.2f}__L=1",
            "method_display": f"Cox SER (@{t:g})",
            "method_display_base": "Cox SER",
            "threshold": t,
            "mean_causal_pip": v,
        })
    rows.append({
        "simulation_name": "s", "method": "twogroup__L=1",
        "method_display": "Twogroup SER", "method_display_base": "Twogroup SER",
        "threshold": None, "mean_causal_pip": 0.6,
    })
    return pl.from_dicts(rows, schema=_SCHEMA)


def test_causal_pip_groups_thresholds_into_one_line():
    fig = viz_utils.render_causal_pip_chart(_summary(), facet=False)
    ax = fig.axes[0]
    _, labels = ax.get_legend_handles_labels()
    # one legend entry per family, not one per threshold
    assert labels.count("Cox SER") == 1
    # the cox thresholds are connected into a single 3-point line
    cox_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == 3]
    assert len(cox_lines) >= 1
