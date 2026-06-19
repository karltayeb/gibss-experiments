from __future__ import annotations

import itertools
from functools import partial
from pathlib import Path
from typing import Any

import yaml

import core
from core import HASH_KEY, dehydrate_hashed
from gibss import distributions as _distributions
from utils import BatchSpec

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


def format_over_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def _is_distribution_node(value: Any) -> bool:
    return (isinstance(value, dict) and len(value) == 1
            and next(iter(value)) in ("Normal", "PointMass", "NormalMixture"))


def resolve_distributions_in_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: (resolve_distribution(v) if _is_distribution_node(v) else v)
            for k, v in kwargs.items()}


def expand_method(base_name: str, entry: dict[str, Any]) -> list[core.MethodSpec]:
    if "__" in base_name:
        raise ValueError(f"Method base name must not contain '__': {base_name!r}")
    fn = resolve_callable(entry["function"])
    template = dict(entry.get("template") or {})
    over = entry.get("over") or {"_dummy": [None]}
    keys = list(over.keys())
    specs: list[core.MethodSpec] = []
    for combo in itertools.product(*(over[k] for k in keys)):
        over_kwargs = {k: v for k, v in zip(keys, combo) if k != "_dummy"}
        suffix = "".join(f"__{k}={format_over_value(v)}" for k, v in over_kwargs.items())
        kwargs = resolve_distributions_in_kwargs({**template, **over_kwargs})
        specs.append(core.MethodSpec(name=f"{base_name}{suffix}", function=fn, kwargs=kwargs))
    return specs


def load_library(experiments_dir: Path | None = None) -> dict[str, Any]:
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    data = yaml.safe_load((base / "library.yaml").read_text(encoding="utf-8")) or {}
    for section in ("defaults", "designs", "enrichments", "signals", "errors",
                    "methods", "reductions", "analyses", "analysis_groups"):
        data.setdefault(section, {})
    return data


def batch_specs_for_simulation(spec, *, replicates_per_batch: int, n_batches: int) -> list[BatchSpec]:
    return [
        BatchSpec(
            name=f"{spec.name}__batch{i}",
            simulation_spec=spec,
            replicates=tuple(range(i * replicates_per_batch, (i + 1) * replicates_per_batch)),
        )
        for i in range(n_batches)
    ]


def library_methods(library: dict[str, Any]) -> dict[str, core.MethodSpec]:
    out: dict[str, core.MethodSpec] = {}
    for base, entry in library["methods"].items():
        for spec in expand_method(base, entry):
            out[spec.name] = spec
    return out


def manifest_dict(library: dict[str, Any], simulations: dict[str, core.SimulationSpec],
                  methods: dict[str, core.MethodSpec]) -> dict[str, Any]:
    defaults = library["defaults"]
    batches: dict[str, Any] = {}
    for spec in simulations.values():
        for batch in batch_specs_for_simulation(
            spec,
            replicates_per_batch=int(defaults["replicates_per_batch"]),
            n_batches=int(defaults["n_batches"]),
        ):
            sim_node = dehydrate_hashed(batch.simulation_spec)
            node = {
                "name": batch.name,
                "simulation_spec": sim_node,
                "replicates": list(batch.replicates),
            }
            node[HASH_KEY] = dehydrate_hashed(batch)[HASH_KEY]
            batches[node[HASH_KEY]] = node
    method_specs: dict[str, Any] = {}
    for spec in methods.values():
        node = dehydrate_hashed(spec)
        method_specs[node[HASH_KEY]] = node
    return {"batches": batches, "method_specs": method_specs}
