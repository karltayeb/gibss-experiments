from __future__ import annotations

import importlib
import inspect
import itertools
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl
import yaml

import core
from core import spec_hash
from experiments import predicates as _predicates
from gibss import distributions as _distributions
from utils import BatchSpec
from viz_utils import (make_method_display_label, method_family_label_map,
                       method_family_oracle_label_map)

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def format_float(value: float) -> str:
    return f"{float(value):.2f}"


def resolve_callable(name: str) -> Any:
    if not hasattr(core, name):
        raise KeyError(f"Unknown callable in core: {name!r}")
    return getattr(core, name)


def resolve_predicate(name: str):
    """Resolve a predicate function from experiments.predicates by name."""
    if not hasattr(_predicates, name):
        raise KeyError(f"Unknown predicate in experiments.predicates: {name!r}")
    return getattr(_predicates, name)


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
                       signal: str, error: str) -> core.SimulationSpec:
    coord = simulation_coordinate(library, design, enrichment, signal, error)
    enrich = coord["enrichment"]
    sig = coord["signal"]
    name = f"{design}__{enrichment}__{signal}" + ("" if error == "gaussian" else f"__{error}")
    return core.SimulationSpec(
        design_sampler=_partial_from_entry(coord["design"]),
        effect_sampler=_partial_from_entry(enrich),
        intercept=float(enrich["intercept"]),
        f0=resolve_distribution(sig["f0"]),
        f1=resolve_distribution(sig["f1"]),
        error_sampler=None if coord["error"] is None else _partial_from_entry(coord["error"]),
        base_seed=coord["base_seed"],
        hash=sim_hash(coord),
        name=name,
    )


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


def expand_method(base_name: str, entry: dict[str, Any]) -> list[dict]:
    if "__" in base_name:
        raise ValueError(f"Method base name must not contain '__': {base_name!r}")
    template = dict(entry.get("template") or {})
    over = entry.get("over") or {"_dummy": [None]}
    keys = list(over.keys())
    out: list[dict] = []
    for combo in itertools.product(*(over[k] for k in keys)):
        ov = {k: v for k, v in zip(keys, combo) if k != "_dummy"}
        suffix = "".join(f"__{k}={format_over_value(v)}" for k, v in ov.items())
        out.append(method_coordinate(f"{base_name}{suffix}", entry["function"], {**template, **ov}))
    return out


def resolve_method(coord: dict) -> tuple:
    """Return (name, fn, resolved_kwargs) for a method coordinate."""
    return coord["name"], resolve_callable(coord["function"]), resolve_distributions_in_kwargs(coord["kwargs"])


def run_method(coord: dict, simulation) -> dict:
    """Execute a method coordinate against a simulation and return a result row."""
    name, fn, kwargs = resolve_method(coord)
    return {"method": name, **fn(simulation, **kwargs)}


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


def library_methods(library: dict[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for base, entry in library["methods"].items():
        for coord in expand_method(base, entry):
            out[coord["name"]] = coord
    return out


_SIM_FIELDS = ("design", "enrichment", "signal", "error")


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _expand_block(library: dict[str, Any], sc_name: str, block: dict[str, Any]) -> list[dict]:
    if "simulations" in block:  # explicit one-off
        sims = [resolve_simulation(library, s["design"], s["enrichment"],
                                   s["signal"], s.get("error", "gaussian"))
                for s in block["simulations"]]
        coords = [{"coordinate": simulation_coordinate(library, s["design"], s["enrichment"],
                                                       s["signal"], s.get("error", "gaussian")),
                   "name": spec.name}
                  for s, spec in zip(block["simulations"], sims)]
        return [{"name": block["name"], "alias": block.get("alias", block["name"]),
                 "simulations": sims, "coordinates": coords}]

    template = dict(block["template"])
    over = block.get("over") or {}
    over_keys = list(over.keys())
    results: list[dict] = []
    for combo in (itertools.product(*(over[k] for k in over_keys)) if over_keys else [()]):
        over_map = dict(zip(over_keys, combo))
        fields = {**template, **over_map}
        # within-collection product over any list-valued template field
        member_lists = {f: _as_list(fields.get(f, "gaussian" if f == "error" else None))
                        for f in _SIM_FIELDS}
        sims = []
        coords = []
        for d, e, s, err in itertools.product(member_lists["design"], member_lists["enrichment"],
                                              member_lists["signal"], member_lists["error"]):
            spec = resolve_simulation(library, d, e, s, err)
            sims.append(spec)
            coords.append({"coordinate": simulation_coordinate(library, d, e, s, err),
                           "name": spec.name})
        suffix = "".join(f"__{k}={over_map[k]}" for k in over_keys)
        name = f"{sc_name}{suffix}" if over_keys else sc_name
        alias = block.get("alias") or "__".join(str(over_map[k]) for k in over_keys) or sc_name
        results.append({"name": name, "alias": alias, "simulations": sims, "coordinates": coords})
    return results


def expand_collections(library: dict[str, Any], sc_name: str, collections_entry: Any) -> list[dict]:
    blocks = collections_entry if isinstance(collections_entry, list) else [collections_entry]
    out: list[dict] = []
    for block in blocks:
        out.extend(_expand_block(library, sc_name, block))
    return out


def load_config(experiments_dir: Path | None = None) -> dict[str, Any]:
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    library = load_library(base)
    supercollections: dict[str, Any] = {}
    for path in sorted(base.glob("*.yaml")):
        if path.name == "library.yaml":
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        supercollections.update(data.get("supercollections", {}) or {})
    return {"library": library, "supercollections": supercollections}


def resolve_methods_for_sc(library: dict[str, Any], sc: dict[str, Any]) -> dict[str, dict]:
    lib_methods = library_methods(library)
    out: dict[str, dict] = {}
    for item in sc.get("methods", []):
        if isinstance(item, str):
            out[item] = lib_methods[item]
        else:  # inline def: single-key map base -> entry
            (base, entry), = item.items()
            for coord in expand_method(base, entry):
                out[coord["name"]] = coord
    return out


def supercollection_collections(library: dict[str, Any], sc_name: str, sc: dict[str, Any]) -> list[dict]:
    return expand_collections(library, sc_name, sc["collections"])


def flatten_analyses(library: dict[str, Any], analyses_list: list[str]) -> list[str]:
    groups = library["analysis_groups"]
    analyses = library["analyses"]
    overlap = set(groups) & set(analyses)
    if overlap:
        raise ValueError(f"Analysis/group name collision: {sorted(overlap)}")
    out: list[str] = []
    for item in analyses_list:
        names = groups[item] if item in groups else [item]
        for n in names:
            if n not in analyses:
                raise KeyError(f"Unknown analysis: {n!r}")
            if n not in out:
                out.append(n)
    return out


def resolve_sc_analyses(config: dict[str, Any], sc_name: str) -> list[tuple[str, str]]:
    library = config["library"]
    sc = config["supercollections"][sc_name]
    seen: dict[tuple[str, str], None] = {}
    for output in sc.get("outputs", []):
        for analysis in flatten_analyses(library, output.get("analyses", [])):
            seen[(analysis, output["name"])] = None
    return list(seen.keys())


def all_simulations(config: dict[str, Any]) -> dict[str, core.SimulationSpec]:
    library = config["library"]
    out: dict[str, core.SimulationSpec] = {}
    for sc_name, sc in config["supercollections"].items():
        for coll in supercollection_collections(library, sc_name, sc):
            for spec in coll["simulations"]:
                out[spec.name] = spec
    return out


def all_methods(config: dict[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sc in config["supercollections"].values():
        out.update(resolve_methods_for_sc(config["library"], sc))
    return out


def _all_sim_coordinates(config: dict[str, Any]) -> list[dict]:
    """Yield deduplicated (coordinate, name) pairs across all supercollections."""
    library = config["library"]
    seen: dict[str, dict] = {}  # hash -> {"coordinate": ..., "name": ...}
    for sc_name, sc in config["supercollections"].items():
        for coll in supercollection_collections(library, sc_name, sc):
            for member in coll["coordinates"]:
                h = sim_hash(member["coordinate"])
                seen.setdefault(h, member)
    return list(seen.values())


def _all_method_coordinates(config: dict[str, Any]) -> list[dict]:
    """Yield deduplicated method coordinate dicts across all supercollections."""
    seen: dict[str, dict] = {}  # hash -> coord
    for sc in config["supercollections"].values():
        for coord in resolve_methods_for_sc(config["library"], sc).values():
            h = method_hash(coord)
            seen.setdefault(h, coord)
    return list(seen.values())


def manifest_dict(library: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Build manifest keyed by content hashes.

    Returns::

        {"batches": {hash: {"coordinate": ..., "replicates": [...], "hash": ..., "name": ...}},
         "methods":  {hash: {...coord_fields..., "hash": ...}}}

    n_batches=1 (one batch per simulation coordinate);
    replicates = range(replicates_per_batch).
    """
    rpb = int(library["defaults"]["replicates_per_batch"])
    batches: dict[str, Any] = {}
    for member in _all_sim_coordinates(config):
        coord = member["coordinate"]
        h = sim_hash(coord)
        batches[h] = {
            "coordinate": coord,
            "replicates": list(range(rpb)),
            "hash": h,
            "name": member["name"],
        }
    methods: dict[str, Any] = {}
    for mcoord in _all_method_coordinates(config):
        h = method_hash(mcoord)
        methods[h] = {**mcoord, "hash": h}
    return {"batches": batches, "methods": methods}


RESULTS_ROOT = "results"


def batch_hashes_for_simulation(library: dict[str, Any], spec: core.SimulationSpec) -> list[str]:
    """Return the batch hash for a simulation spec (n_batches=1; hash = sim_hash of coordinate)."""
    return [spec.hash]


def reduction_output(batch_hash: str, method_hash: str, reduction: str) -> str:
    return f"{RESULTS_ROOT}/by_batch/{batch_hash}/fits/{method_hash}/reductions/{reduction}.parquet"


def reduction_inputs(manifest: dict[str, Any], batch_hash: str, method_hash: str) -> list[str]:
    base = f"{RESULTS_ROOT}/by_batch/{batch_hash}"
    return [
        f"{base}/fits/{method_hash}/fits.parquet",
        f"{base}/simulations.parquet",
        f"{base}/sample_metadata.parquet",
    ]


def reduction_method_filter(library: dict[str, Any], reduction: str) -> str | None:
    return library["reductions"][reduction].get("method_filter")


def analysis_simulation_filter(library: dict[str, Any], analysis: str) -> str | None:
    return library["analyses"][analysis].get("simulation_filter")


def _method_hashes(methods: dict[str, dict]) -> dict[str, str]:
    return {c["name"]: method_hash(c) for c in methods.values()}


def collection_method_pairs(config: dict[str, Any], sc_name: str) -> dict[str, dict]:
    """Return per-collection dicts with 'alias' and 'pairs'.

    Each pair is (batch_hash, method_hash, method_name, method_coord, sim_coordinate).
    """
    library = config["library"]
    sc = config["supercollections"][sc_name]
    methods = resolve_methods_for_sc(library, sc)
    mhash = _method_hashes(methods)
    out: dict[str, dict] = {}
    for coll in supercollection_collections(library, sc_name, sc):
        pairs = []
        for member, spec in zip(coll["coordinates"], coll["simulations"]):
            sim_coord = member["coordinate"]
            for bh in batch_hashes_for_simulation(library, spec):
                for mname, mh in mhash.items():
                    mcoord = methods[mname]
                    pairs.append((bh, mh, mname, mcoord, sim_coord))
        out[coll["name"]] = {"alias": coll["alias"], "pairs": pairs}
    return out


def analysis_inputs(config: dict[str, Any], manifest: dict[str, Any],
                    sc_name: str, analysis: str) -> list[str]:
    library = config["library"]
    requires = library["analyses"][analysis].get("requires", [])
    sim_filter_name = analysis_simulation_filter(library, analysis)
    sim_pred = resolve_predicate(sim_filter_name) if sim_filter_name is not None else None
    cmp = collection_method_pairs(config, sc_name)
    paths: list[str] = []
    seen: set[str] = set()
    for reduction in requires:
        mfilter_name = reduction_method_filter(library, reduction)
        method_pred = resolve_predicate(mfilter_name) if mfilter_name is not None else None
        for coll in cmp.values():
            for bh, mh, mname, mcoord, sim_coord in coll["pairs"]:
                if method_pred is not None and not method_pred(mcoord):
                    continue
                if sim_pred is not None and not sim_pred(sim_coord):
                    continue
                p = reduction_output(bh, mh, reduction)
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
    return paths


def resolve_args(config: dict[str, Any], sc_name: str, args_name: str) -> dict[str, Any]:
    sc = config["supercollections"][sc_name]
    defaults = dict(sc.get("default_args", {}) or {})
    for output in sc.get("outputs", []):
        if output["name"] == args_name:
            return {**defaults, **(output.get("args") or {}),
                    "method_filter": output.get("method_filter", [])}
    raise KeyError(f"No output named {args_name!r} in supercollection {sc_name!r}")


def analysis_requires(config: dict[str, Any], analysis: str) -> list[str]:
    return list(config["library"]["analyses"][analysis].get("requires", []))


def analysis_function(config: dict[str, Any], analysis: str):
    import generate_plots
    return generate_plots.ANALYSIS_RENDERERS[analysis]


def load_sc_bundle(config: dict[str, Any], sc_name: str, requires: list[str],
                   results_root: str = RESULTS_ROOT,
                   simulation_filter: str | None = None) -> dict[str, Any]:
    library = config["library"]
    cmp = collection_method_pairs(config, sc_name)
    bundle: dict[str, Any] = {}
    collection_names = [info["alias"] for info in cmp.values()]
    sim_pred = resolve_predicate(simulation_filter) if simulation_filter is not None else None
    for reduction in requires:
        mfilter_name = reduction_method_filter(library, reduction)
        method_pred = resolve_predicate(mfilter_name) if mfilter_name is not None else None
        frames = []
        for info in cmp.values():
            sub = []
            for bh, mh, mname, mcoord, sim_coord in info["pairs"]:
                if method_pred is not None and not method_pred(mcoord):
                    continue
                if sim_pred is not None and not sim_pred(sim_coord):
                    continue
                path = f"{results_root}/{reduction_output(bh, mh, reduction).split('/', 1)[1]}"
                df = pl.read_parquet(path)
                sub.append(df)
            if sub:
                merged = pl.concat(sub, how="diagonal_relaxed").with_columns(
                    pl.lit(info["alias"]).alias("collection_name"))
                frames.append(merged)
        bundle[f"{reduction}_plot_data"] = (
            pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame())
    bundle["method_metadata"] = method_metadata(
        resolve_methods_for_sc(library, config["supercollections"][sc_name]))
    bundle["collection_names"] = collection_names
    return bundle


def simulation_coordinate(library, design, enrichment, signal, error) -> dict:
    return {
        "design": library["designs"][design],
        "enrichment": library["enrichments"][enrichment],
        "signal": library["signals"][signal],
        "error": library["errors"][error],          # None for "gaussian"
        "base_seed": int(library["defaults"]["base_seed"]),
    }


def method_coordinate(name, function, kwargs_raw) -> dict:
    return {"name": name, "function": function, "kwargs": kwargs_raw}


def sim_hash(coordinate) -> str: return spec_hash(coordinate)
def method_hash(coordinate) -> str: return spec_hash(coordinate)


# ---------------------------------------------------------------------------
# Code-tracking helpers: map coordinates → source file paths for Snakemake
# ---------------------------------------------------------------------------

def _file(fn) -> str:
    """Return the source file of fn, unwrapping functools.partial."""
    return inspect.getfile(getattr(fn, "func", fn))


def resolve_simulation_from_coord(coord: dict[str, Any]) -> core.SimulationSpec:
    """Build a SimulationSpec from a prebuilt coordinate dict (no library lookup)."""
    enrich = coord["enrichment"]
    sig = coord["signal"]
    return core.SimulationSpec(
        design_sampler=_partial_from_entry(coord["design"]),
        effect_sampler=_partial_from_entry(enrich),
        intercept=float(enrich["intercept"]),
        f0=resolve_distribution(sig["f0"]),
        f1=resolve_distribution(sig["f1"]),
        error_sampler=None if coord["error"] is None else _partial_from_entry(coord["error"]),
        base_seed=coord["base_seed"],
        hash=sim_hash(coord),
        name="",
    )


def simulation_code_files(coord: dict[str, Any], library: dict[str, Any]) -> list[str]:
    """Return sorted unique source files for core.simulate + the sim's samplers."""
    spec = resolve_simulation_from_coord(coord)
    fns = [core.simulate, spec.design_sampler, spec.effect_sampler]
    if spec.error_sampler is not None:
        fns.append(spec.error_sampler)
    return sorted({_file(f) for f in fns})


def method_code_files(method_coord: dict[str, Any], library: dict[str, Any]) -> list[str]:
    """Return the source file of the resolved fit function for a method coordinate."""
    return [_file(resolve_callable(method_coord["function"]))]


def reduction_code_files(reduction: str, library: dict[str, Any]) -> list[str]:
    """Return the source file of reductions.<reduction>.build."""
    mod = importlib.import_module(f"reductions.{reduction}")
    return [_file(getattr(mod, "build"))]


def analysis_code_files(analysis: str) -> list[str]:
    """Return the source file of the renderer for the given analysis."""
    import generate_plots
    return [_file(generate_plots.ANALYSIS_RENDERERS[analysis])]


def analysis_family(analysis: str) -> str:
    """Return the family module name (pip|cs|logbf|f1) for an analysis name."""
    from analyses import pip, cs, logbf, f1
    if analysis in pip.RENDERERS:
        return "pip"
    if analysis in cs.RENDERERS:
        return "cs"
    if analysis in logbf.RENDERERS:
        return "logbf"
    if analysis in f1.RENDERERS:
        return "f1"
    raise KeyError(f"Unknown analysis (not in any family RENDERERS): {analysis!r}")


def method_metadata(methods: dict[str, dict]) -> pl.DataFrame:
    label_map = method_family_label_map()
    oracle_map = method_family_oracle_label_map()
    rows = []
    for name, spec in methods.items():
        family = name.split("__")[0]
        L = int(spec["kwargs"].get("L", 1))
        threshold = spec["kwargs"].get("threshold")
        is_thresholded = threshold is not None
        is_oracle = "oracle" in family
        family_label = label_map.get(family, family)
        oracle_label = oracle_map.get(family, "Oracle")
        suffix = "SER" if L == 1 else f"SuSiE [L={L}]"
        base = f"{family_label} {suffix}"
        rows.append({
            "method": name,
            "method_family": family,
            "L": L,
            "threshold": float(threshold) if threshold is not None else None,
            "is_thresholded": is_thresholded,
            "is_oracle": is_oracle,
            "method_label_base": base,
            "method_display": make_method_display_label(
                method_label_base=base, threshold=threshold,
                is_thresholded=is_thresholded, is_oracle=is_oracle, oracle_label=oracle_label),
            "method_display_base": make_method_display_label(
                method_label_base=base, threshold=None, is_thresholded=False,
                is_oracle=is_oracle, oracle_label=oracle_label),
        })
    return pl.from_dicts(rows, schema_overrides={"threshold": pl.Float64})
