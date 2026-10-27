"""Experiment grid: config.yaml -> content-addressed cells.

A cell is (PopulationSpec, ExpressionSpec); its hash names `results/cells/<hash>/`.
Collections are named Cartesian grids over ExpressionSpec fields with a replicate count,
the SuSiE L values and the arms to fit. `grid` is one dict (a Cartesian product) or a list
of dicts (the union of their products, e.g. a signal grid plus null cells). A collection may
override fields of the global `population` (window, sample size). 
Replicates run in batches: a collection's `reps` are split into contiguous ranges of
`batch_size` (default: all reps in one batch), and each range is one simulate job and one
fit job per (arm, L), stored under `reps_<start>_<stop>`. A cell shared by two collections
carries the union of their batch ranges; a range that both ask for is simulated and fitted
once. Replicate `rep` is drawn from `rep_seed(cell, rep)` whatever batch it sits in.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, fields

import yaml

from .genotypes import PopulationSpec
from .simulate import ExpressionSpec

__all__ = ["Cell", "load_config", "expand", "cell_hash", "batches", "collection_members"]


@dataclass(frozen=True)
class Cell:
    population: PopulationSpec
    expression: ExpressionSpec
    reps: int
    batches: tuple[tuple[int, int], ...]  # (start, stop) replicate ranges, union over collections
    L_values: tuple[str, ...]
    arms: tuple[str, ...]
    collections: tuple[str, ...]

    @property
    def hash(self) -> str:
        return cell_hash(self.population, self.expression)

    def coordinate(self) -> dict:
        """Flat dict of every spec field (for joining onto result tables)."""
        d = {"cell": self.hash, "population": self.population.hash}
        d.update(asdict(self.expression))
        d["beta_abs"] = d.pop("beta")  # the cs table's `beta` is a posterior mean
        return d


def cell_hash(pop: PopulationSpec, expr: ExpressionSpec) -> str:
    payload = json.dumps({"population": asdict(pop), "expression": asdict(expr)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def batches(reps: int, batch_size: int | None = None) -> tuple[tuple[int, int], ...]:
    """Contiguous (start, stop) ranges covering range(reps); one batch when batch_size is None."""
    reps = int(reps)
    size = reps if batch_size is None else int(batch_size)
    if reps <= 0 or size <= 0:
        raise ValueError(f"reps and batch_size must be positive, got {reps} and {size}")
    return tuple((start, min(start + size, reps)) for start in range(0, reps, size))


def collection_batches(coll: dict) -> tuple[tuple[int, int], ...]:
    return batches(coll["reps"], coll.get("batch_size"))


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _grid(grid: dict) -> list[dict]:
    names = [f.name for f in fields(ExpressionSpec)]
    unknown = set(grid) - set(names)
    if unknown:
        raise ValueError(f"grid keys not ExpressionSpec fields: {sorted(unknown)}")
    keys = list(grid)
    vals = [v if isinstance(v, list) else [v] for v in (grid[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def _points(grid) -> list[dict]:
    grids = grid if isinstance(grid, list) else [grid]
    return [pt for g in grids for pt in _grid(g)]


def collection_population(cfg: dict, name: str) -> PopulationSpec:
    return PopulationSpec(**{**cfg["population"], **(cfg["collections"][name].get("population") or {})})


def expand(cfg: dict) -> dict[str, Cell]:
    """All cells across collections, keyed by hash. Merges reps (max), L values and arms
    (union) for a cell that appears in several collections."""
    merged: dict[str, dict] = {}
    for name, coll in cfg["collections"].items():
        pop = collection_population(cfg, name)
        for point in _points(coll["grid"]):
            expr = ExpressionSpec(**point)
            h = cell_hash(pop, expr)
            entry = merged.setdefault(h, {
                "population": pop, "expression": expr, "reps": 0, "batches": set(), "L": set(), "arms": set(),
                "collections": set(),
            })
            entry["reps"] = max(entry["reps"], int(coll["reps"]))
            entry["batches"].update(collection_batches(coll))
            Ls = coll["L"] if isinstance(coll["L"], list) else [coll["L"]]
            entry["L"].update(str(L) for L in Ls)
            entry["arms"].update(coll["arms"])
            entry["collections"].add(name)
    return {
        h: Cell(
            population=e["population"], expression=e["expression"], reps=e["reps"],
            batches=tuple(sorted(e["batches"])), L_values=tuple(sorted(e["L"])), arms=tuple(sorted(e["arms"])),
            collections=tuple(sorted(e["collections"])),
        )
        for h, e in merged.items()
    }


def collection_members(cfg: dict, cells: dict[str, Cell], name: str) -> list[tuple[str, str, str, int, int]]:
    """(cell_hash, arm, L, start, stop) fit batches a collection's reduce step gathers."""
    coll = cfg["collections"][name]
    Ls = coll["L"] if isinstance(coll["L"], list) else [coll["L"]]
    out = []
    for h, c in cells.items():
        if name in c.collections:
            for arm in coll["arms"]:
                for L in Ls:
                    for start, stop in collection_batches(coll):
                        out.append((h, arm, str(L), start, stop))
    return out
