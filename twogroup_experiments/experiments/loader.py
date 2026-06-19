from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import yaml

import core
from gibss import distributions as _distributions

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def format_float(value: float) -> str:
    return f"{float(value):.2f}"


def resolve_callable(name: str) -> Any:
    if not hasattr(core, name):
        raise KeyError(f"Unknown callable in core: {name!r}")
    return getattr(core, name)


def resolve_distribution(node: dict[str, Any]) -> Any:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(f"Distribution node must be a single-key map, got {node!r}")
    (type_name, ctor_kwargs), = node.items()
    if not hasattr(_distributions, type_name):
        raise KeyError(f"Unknown distribution type: {type_name!r}")
    return getattr(_distributions, type_name)(**(ctor_kwargs or {}))


def _partial_from_entry(entry: dict[str, Any]):
    fn = resolve_callable(entry["function"])
    return partial(fn, **(entry.get("arguments") or {}))


def resolve_simulation(library: dict[str, Any], design: str, enrichment: str,
                       signal: str, error: str) -> tuple[core.SimulationSpec, str]:
    design_entry = library["designs"][design]
    enrich_entry = library["enrichments"][enrichment]
    signal_entry = library["signals"][signal]
    error_entry = library["errors"][error]

    name = f"{design}__{enrichment}__{signal}"
    if error != "gaussian":
        name = f"{name}__{error}"

    error_sampler = None if error_entry is None else _partial_from_entry(error_entry)
    spec = core.SimulationSpec(
        name=name,
        design_sampler=_partial_from_entry(design_entry),
        effect_sampler=_partial_from_entry(enrich_entry),
        intercept=float(enrich_entry["intercept"]),
        f0=resolve_distribution(signal_entry["f0"]),
        f1=resolve_distribution(signal_entry["f1"]),
        base_seed=int(library["defaults"]["base_seed"]),
        error_sampler=error_sampler,
    )
    return spec, name


def load_library(experiments_dir: Path | None = None) -> dict[str, Any]:
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    data = yaml.safe_load((base / "library.yaml").read_text(encoding="utf-8")) or {}
    for section in ("defaults", "designs", "enrichments", "signals", "errors",
                    "methods", "reductions", "analyses", "analysis_groups"):
        data.setdefault(section, {})
    return data
