"""Per-analysis geometry hooks + registry.

Each analysis provides: a `requires` reduction key, an `aggregate(rows)->stats`
that pools a series' rows into the plotted statistic (must be grouping-invariant),
and a `draw(ax, stats, *, color, linestyle, label)` that renders one series.
The generic driver (generate_plots.render_plot) owns filtering and faceting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AnalysisHook:
    requires: str
    aggregate: Callable
    draw: Callable


HOOKS: dict[str, AnalysisHook] = {}


def add_hook(name: str, requires: str, aggregate, draw):
    HOOKS[name] = AnalysisHook(requires=requires, aggregate=aggregate, draw=draw)
