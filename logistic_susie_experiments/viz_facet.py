"""Pure faceting/aesthetic engine for the plot spec.

apply_filter: equality (scalar) / membership (list) over dimension columns.
assign_groups: split rows into a facet grid (facet_row x facet_col) of series
(color x linestyle); everything not on a channel is pooled into its series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


def apply_filter(rows: list[dict], filt: dict | None) -> list[dict]:
    if not filt:
        return list(rows)
    def keep(r: dict) -> bool:
        for key, want in filt.items():
            val = r.get(key)
            if isinstance(want, (list, tuple, set)):
                if val not in want:
                    return False
            elif val != want:
                return False
        return True
    return [r for r in rows if keep(r)]


@dataclass(frozen=True)
class Series:
    color_key: object
    linestyle_key: object
    rows: list[dict]


@dataclass
class FacetGrid:
    row_keys: list
    col_keys: list
    cells: dict = field(default_factory=dict)  # (row_key, col_key) -> list[Series]

    def cell(self, row_key, col_key) -> list:
        return self.cells.get((row_key, col_key), [])


def _distinct(rows, key):
    if key is None:
        return [None]
    seen = []
    for r in rows:
        v = r.get(key)
        if v not in seen:
            seen.append(v)
    return seen


def assign_groups(rows, *, facet_row=None, facet_col=None, color=None, linestyle=None) -> FacetGrid:
    row_keys = _distinct(rows, facet_row)
    col_keys = _distinct(rows, facet_col)
    color_keys = _distinct(rows, color)
    ls_keys = _distinct(rows, linestyle)

    def matches(r, key, val):
        return key is None or r.get(key) == val

    cells: dict = {}
    for rk, ck in product(row_keys, col_keys):
        series_list = []
        for colk, lsk in product(color_keys, ls_keys):
            sub = [r for r in rows
                   if matches(r, facet_row, rk) and matches(r, facet_col, ck)
                   and matches(r, color, colk) and matches(r, linestyle, lsk)]
            if sub:
                series_list.append(Series(colk, lsk, sub))
        cells[(rk, ck)] = series_list
    return FacetGrid(row_keys, col_keys, cells)
