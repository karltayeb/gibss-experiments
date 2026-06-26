"""Config loader for the logistic SuSiE experiments pipeline.

Mirrors twogroup_experiments/experiments/loader.py but with a simplified
two-axis simulation model: ``design`` x ``enrichment`` (no signal / error).

Terminology:
- coordinate: plain-Python dict describing a simulation or method; hashed to
  derive content-addressed result paths.
- manifest: {batches: {hash: ...}, methods: {hash: ...}} consumed by Snakemake.
- supercollection / collection: declarative grouping in the *.yaml experiment
  files; a collection expands to a set of simulation coordinates.
"""
from __future__ import annotations

import importlib
import inspect
import itertools
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

import core
from core import spec_hash
from experiments import predicates as _predicates
from utils import BatchSpec

EXPERIMENTS_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = "results"


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def resolve_callable(name: str) -> Any:
    if not hasattr(core, name):
        raise KeyError(f"Unknown callable in core: {name!r}")
    return getattr(core, name)


def resolve_predicate(name: str):
    if not hasattr(_predicates, name):
        raise KeyError(f"Unknown predicate in experiments.predicates: {name!r}")
    return getattr(_predicates, name)


def _partial_from_entry(entry: dict[str, Any]):
    fn = resolve_callable(entry["function"])
    return partial(fn, **(entry.get("arguments") or {}))


def format_over_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Simulation coordinates / specs
# ---------------------------------------------------------------------------

def simulation_coordinate(library, design: str, enrichment: str) -> dict:
    return {
        "design": library["designs"][design],
        "enrichment": library["enrichments"][enrichment],
        "base_seed": int(library["defaults"]["base_seed"]),
    }


def _spec_from_coord(coord: dict[str, Any], name: str) -> core.SimulationSpec:
    enrich = coord["enrichment"]
    return core.SimulationSpec(
        design_sampler=_partial_from_entry(coord["design"]),
        effect_sampler=_partial_from_entry(enrich),
        intercept=float(enrich["intercept"]),
        base_seed=coord["base_seed"],
        hash=sim_hash(coord),
        name=name,
    )


def resolve_simulation(library, design: str, enrichment: str) -> core.SimulationSpec:
    coord = simulation_coordinate(library, design, enrichment)
    return _spec_from_coord(coord, f"{design}__{enrichment}")


def resolve_simulation_from_coord(coord: dict[str, Any]) -> core.SimulationSpec:
    """Build a SimulationSpec from a prebuilt coordinate (no library lookup)."""
    return _spec_from_coord(coord, "")


# ---------------------------------------------------------------------------
# Method coordinates
# ---------------------------------------------------------------------------

def method_coordinate(name, function, kwargs_raw) -> dict:
    return {"name": name, "function": function, "kwargs": kwargs_raw}


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


def library_methods(library: dict[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for base, entry in library["methods"].items():
        for coord in expand_method(base, entry):
            out[coord["name"]] = coord
    return out


def resolve_method(coord: dict) -> tuple:
    return coord["name"], resolve_callable(coord["function"]), dict(coord["kwargs"])


def run_method(coord: dict, simulation) -> dict:
    name, fn, kwargs = resolve_method(coord)
    return {"method": name, **fn(simulation, **kwargs)}


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


# ---------------------------------------------------------------------------
# Library / config loading
# ---------------------------------------------------------------------------

def _expand_enrichment_family(entry: dict[str, Any]) -> list[tuple[str, dict]]:
    """Expand one enrichment-family entry (has a ``name`` template) into concrete
    enrichments. Any list-valued field — ``intercept`` or an ``arguments`` value —
    becomes a grid axis; the cartesian product is taken and ``name`` is formatted
    with the per-combo scalar values.
    """
    fn = entry["function"]
    membership = entry.get("membership")
    args = dict(entry.get("arguments") or {})
    axes: dict[str, list] = {"intercept": _as_list(entry["intercept"])}
    for key, value in args.items():
        axes[key] = _as_list(value)
    keys = list(axes)
    out: list[tuple[str, dict]] = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        vals = dict(zip(keys, combo))
        name = entry["name"].format(**vals)
        concrete = {
            "function": fn,
            "arguments": {k: vals[k] for k in args},
            "intercept": vals["intercept"],
        }
        if membership is not None:
            concrete["membership"] = membership
        out.append((name, concrete))
    return out


def _expand_enrichments(raw: dict[str, Any]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Split the raw enrichments map into concrete entries + family groups.

    An entry with a ``name`` template is a family that expands into many concrete
    enrichments; the family key becomes a group listing the generated names.
    Entries without ``name`` are literal enrichments keyed by their map key.
    """
    flat: dict[str, dict] = {}
    groups: dict[str, list[str]] = {}
    for key, entry in raw.items():
        if "name" not in entry:
            flat[key] = entry
            continue
        members = _expand_enrichment_family(entry)
        groups[key] = [name for name, _ in members]
        for name, concrete in members:
            if name in flat:
                raise ValueError(f"Duplicate enrichment name from family {key!r}: {name!r}")
            flat[name] = concrete
    return flat, groups


def load_library(experiments_dir: Path | None = None) -> dict[str, Any]:
    import yaml
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    data = yaml.safe_load((base / "library.yaml").read_text(encoding="utf-8")) or {}
    for section in ("defaults", "designs", "enrichments", "methods", "reductions",
                    "analyses", "analysis_groups"):
        data.setdefault(section, {})
    data["enrichments"], data["enrichment_groups"] = _expand_enrichments(data["enrichments"])
    return data


def load_config(experiments_dir: Path | None = None) -> dict[str, Any]:
    import yaml
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    library = load_library(base)
    supercollections: dict[str, Any] = {}
    for path in sorted(base.glob("*.yaml")):
        if path.name == "library.yaml":
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        supercollections.update(data.get("supercollections", {}) or {})
    return {"library": library, "supercollections": supercollections}


# ---------------------------------------------------------------------------
# Collection expansion
# ---------------------------------------------------------------------------

_SIM_FIELDS = ("design", "enrichment")


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _resolve_enrichment_refs(library: dict[str, Any], names: list) -> list:
    """Expand any enrichment-family group name into its concrete member names.

    Non-string items (e.g. a nested ``[signal, null]`` list used to pair a null
    into an ``over`` collection) are passed through untouched.
    """
    groups = library.get("enrichment_groups", {})
    out: list = []
    for name in names:
        if isinstance(name, str) and name in groups:
            out.extend(groups[name])
        else:
            out.append(name)
    return out


def _expand_block(library: dict[str, Any], sc_name: str, block: dict[str, Any]) -> list[dict]:
    if "simulations" in block:  # explicit one-off list
        coords, sims = [], []
        for s in block["simulations"]:
            coord = simulation_coordinate(library, s["design"], s["enrichment"])
            spec = resolve_simulation(library, s["design"], s["enrichment"])
            coords.append({"coordinate": coord, "name": spec.name})
            sims.append(spec)
        return [{"name": block["name"], "alias": block.get("alias", block["name"]),
                 "simulations": sims, "coordinates": coords}]

    template = dict(block["template"])
    over = dict(block.get("over") or {})
    # A family group named as an `over` enrichment value expands into separate
    # over-values (one collection each). At the `template` level it instead unions
    # into a single collection (handled below via member_lists).
    if "enrichment" in over:
        over["enrichment"] = _resolve_enrichment_refs(library, _as_list(over["enrichment"]))
    over_keys = list(over.keys())
    combos = list(itertools.product(*(over[k] for k in over_keys))) if over_keys else [()]
    aliases = block.get("aliases")
    if aliases is not None and len(aliases) != len(combos):
        raise ValueError(
            f"aliases length {len(aliases)} != number of over-combos {len(combos)} for {sc_name!r}"
        )
    results: list[dict] = []
    for idx, combo in enumerate(combos):
        over_map = dict(zip(over_keys, combo))
        fields = {**template, **over_map}
        member_lists = {f: _as_list(fields.get(f)) for f in _SIM_FIELDS}
        member_lists["enrichment"] = _resolve_enrichment_refs(library, member_lists["enrichment"])
        sims, coords = [], []
        for d, e in itertools.product(member_lists["design"], member_lists["enrichment"]):
            coord = simulation_coordinate(library, d, e)
            spec = resolve_simulation(library, d, e)
            sims.append(spec)
            coords.append({"coordinate": coord, "name": spec.name})
        suffix = "".join(f"__{k}={over_map[k]}" for k in over_keys)
        if over_keys:
            name = f"{sc_name}{suffix}"
        else:
            # template-form block with no `over`: disambiguate sibling blocks via
            # an explicit per-block `name` (else they collide on sc_name).
            block_name = block.get("name")
            name = f"{sc_name}__{block_name}" if block_name else sc_name
        if aliases is not None:
            alias = aliases[idx]
        else:
            alias = block.get("alias") or "__".join(str(over_map[k]) for k in over_keys) or sc_name
        results.append({"name": name, "alias": alias, "simulations": sims, "coordinates": coords})
    return results


def expand_collections(library: dict[str, Any], sc_name: str, collections_entry: Any) -> list[dict]:
    blocks = collections_entry if isinstance(collections_entry, list) else [collections_entry]
    out: list[dict] = []
    for block in blocks:
        out.extend(_expand_block(library, sc_name, block))
    return out


def supercollection_collections(library: dict[str, Any], sc_name: str, sc: dict[str, Any]) -> list[dict]:
    return expand_collections(library, sc_name, sc["collections"])


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sim_hash(coordinate) -> str: return spec_hash(coordinate)
def method_hash(coordinate) -> str: return spec_hash(coordinate)


def _batch_hash(sim_hash_value: str, batch_index: int) -> str:
    """Batch ``batch_index`` of a simulation coordinate.

    Batch 0 keeps the bare ``sim_hash`` so single-batch results stay valid;
    later batches derive a distinct stable hash.
    """
    if batch_index == 0:
        return sim_hash_value
    return spec_hash({"simulation": sim_hash_value, "batch": int(batch_index)})


def batch_hashes_for_simulation(library, spec, n_batches: int | None = None) -> list[str]:
    nb = int(library["defaults"]["n_batches"] if n_batches is None else n_batches)
    return [_batch_hash(spec.hash, i) for i in range(nb)]


def batch_specs_for_simulation(spec, *, replicates_per_batch: int, n_batches: int) -> list[BatchSpec]:
    return [
        BatchSpec(
            name=f"{spec.name}__batch{i}",
            simulation_spec=spec,
            replicates=tuple(range(i * replicates_per_batch, (i + 1) * replicates_per_batch)),
        )
        for i in range(n_batches)
    ]


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def _all_sim_coordinates(config: dict[str, Any]) -> list[dict]:
    library = config["library"]
    seen: dict[str, dict] = {}
    for sc_name, sc in config["supercollections"].items():
        sc_nb = sc.get("n_batches")
        for coll in supercollection_collections(library, sc_name, sc):
            for member in coll["coordinates"]:
                h = sim_hash(member["coordinate"])
                existing = seen.get(h)
                if existing is None:
                    existing = dict(member)
                    seen[h] = existing
                if sc_nb is not None:
                    existing["n_batches"] = max(int(sc_nb), int(existing.get("n_batches", 0)))
    return list(seen.values())


def _all_method_coordinates(config: dict[str, Any]) -> list[dict]:
    seen: dict[str, dict] = {}
    for sc in config["supercollections"].values():
        for coord in resolve_methods_for_sc(config["library"], sc).values():
            seen.setdefault(method_hash(coord), coord)
    return list(seen.values())


def manifest_dict(library: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rpb = int(library["defaults"]["replicates_per_batch"])
    default_nb = int(library["defaults"]["n_batches"])
    batches: dict[str, Any] = {}
    for member in _all_sim_coordinates(config):
        coord = member["coordinate"]
        sh = sim_hash(coord)
        nb = int(member.get("n_batches", default_nb))
        for i in range(nb):
            bh = _batch_hash(sh, i)
            batches[bh] = {
                "coordinate": coord,
                "replicates": list(range(i * rpb, (i + 1) * rpb)),
                "hash": bh,
                "name": member["name"] if nb == 1 else f"{member['name']}__batch{i}",
            }
    methods: dict[str, Any] = {}
    for mcoord in _all_method_coordinates(config):
        h = method_hash(mcoord)
        methods[h] = {**mcoord, "hash": h}
    return {"batches": batches, "methods": methods}


# ---------------------------------------------------------------------------
# Result paths / batch-method pairs / reduction targets
# ---------------------------------------------------------------------------

def reduction_output(batch_hash: str, method_hash: str, reduction: str) -> str:
    return f"{RESULTS_ROOT}/by_batch/{batch_hash}/fits/{method_hash}/reductions/{reduction}.parquet"


def reduction_inputs(batch_hash: str, method_hash: str) -> list[str]:
    base = f"{RESULTS_ROOT}/by_batch/{batch_hash}"
    return [
        f"{base}/fits/{method_hash}/fits.parquet",
        f"{base}/simulations.parquet",
        f"{base}/sample_metadata.parquet",
    ]


def reduction_method_filter(library: dict[str, Any], reduction: str) -> str | None:
    return library["reductions"][reduction].get("method_filter")


def collection_method_pairs(config: dict[str, Any], sc_name: str) -> dict[str, dict]:
    """Per-collection dicts with 'alias' and 'pairs'.

    Each pair is (batch_hash, method_hash, method_name, method_coord, sim_coordinate).
    """
    library = config["library"]
    sc = config["supercollections"][sc_name]
    methods = resolve_methods_for_sc(library, sc)
    mhash = {c["name"]: method_hash(c) for c in methods.values()}
    sc_nb = sc.get("n_batches", library["defaults"]["n_batches"])
    out: dict[str, dict] = {}
    for coll in supercollection_collections(library, sc_name, sc):
        pairs = []
        for member, spec in zip(coll["coordinates"], coll["simulations"]):
            sim_coord = member["coordinate"]
            for bh in batch_hashes_for_simulation(library, spec, sc_nb):
                for mname, mh in mhash.items():
                    pairs.append((bh, mh, mname, methods[mname], sim_coord))
        out[coll["name"]] = {"alias": coll["alias"], "pairs": pairs}
    return out


def all_reduction_targets(config: dict[str, Any]) -> list[str]:
    """Every reduction parquet across all supercollections (method-filtered)."""
    library = config["library"]
    paths: list[str] = []
    seen: set[str] = set()
    for reduction in library["reductions"]:
        mfilter_name = reduction_method_filter(library, reduction)
        method_pred = resolve_predicate(mfilter_name) if mfilter_name else None
        for sc_name in config["supercollections"]:
            for coll in collection_method_pairs(config, sc_name).values():
                for bh, mh, _mname, mcoord, _sim in coll["pairs"]:
                    if method_pred is not None and not method_pred(mcoord):
                        continue
                    p = reduction_output(bh, mh, reduction)
                    if p not in seen:
                        seen.add(p)
                        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Code-tracking helpers: map coordinates -> source file paths for Snakemake
# ---------------------------------------------------------------------------

def _file(fn) -> str:
    return inspect.getfile(getattr(fn, "func", fn))


def simulation_code_files(coord: dict[str, Any], library: dict[str, Any]) -> list[str]:
    spec = resolve_simulation_from_coord(coord)
    fns = [core.simulate, spec.design_sampler, spec.effect_sampler]
    return sorted({_file(f) for f in fns})


def method_code_files(method_coord: dict[str, Any], library: dict[str, Any]) -> list[str]:
    return [_file(resolve_callable(method_coord["function"]))]


def reduction_code_files(reduction: str, library: dict[str, Any]) -> list[str]:
    mod = importlib.import_module(f"reductions.{reduction}")
    return [_file(getattr(mod, "build"))]


# ---------------------------------------------------------------------------
# Analyses (plotting layer)
# ---------------------------------------------------------------------------

def _base_analysis(analysis: str) -> str:
    return analysis


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
    """Return (analysis, output_name) pairs for a supercollection."""
    library = config["library"]
    sc = config["supercollections"][sc_name]
    seen: dict[tuple[str, str], None] = {}
    for output in sc.get("outputs", []):
        for analysis in flatten_analyses(library, output.get("analyses", [])):
            seen[(analysis, output["name"])] = None
    return list(seen.keys())


def analysis_requires(config: dict[str, Any], analysis: str) -> list[str]:
    return list(config["library"]["analyses"][analysis].get("requires", []))


def analysis_simulation_filter(library: dict[str, Any], analysis: str) -> str | None:
    return library["analyses"][analysis].get("simulation_filter")


def analysis_family(analysis: str) -> str:
    """Family module name (pip|cs) for an analysis."""
    from analyses import pip, cs
    if analysis in pip.RENDERERS:
        return "pip"
    if analysis in cs.RENDERERS:
        return "cs"
    raise KeyError(f"Unknown analysis (not in any family RENDERERS): {analysis!r}")


def analysis_function(config: dict[str, Any], analysis: str):
    import generate_plots
    return generate_plots.ANALYSIS_RENDERERS[analysis]


def analysis_code_files(analysis: str) -> list[str]:
    import generate_plots
    return [_file(generate_plots.ANALYSIS_RENDERERS[analysis])]


def resolve_args(config: dict[str, Any], sc_name: str, args_name: str) -> dict[str, Any]:
    sc = config["supercollections"][sc_name]
    defaults = dict(sc.get("default_args", {}) or {})
    for output in sc.get("outputs", []):
        if output["name"] == args_name:
            return {**defaults, **(output.get("args") or {}),
                    "method_filter": output.get("method_filter", [])}
    raise KeyError(f"No output named {args_name!r} in supercollection {sc_name!r}")


def analysis_inputs(config: dict[str, Any], manifest: dict[str, Any],
                    sc_name: str, analysis: str) -> list[str]:
    library = config["library"]
    requires = analysis_requires(config, analysis)
    sim_filter_name = analysis_simulation_filter(library, analysis)
    sim_pred = resolve_predicate(sim_filter_name) if sim_filter_name is not None else None
    cmp = collection_method_pairs(config, sc_name)
    paths: list[str] = []
    seen: set[str] = set()
    for reduction in requires:
        mfilter_name = reduction_method_filter(library, reduction)
        method_pred = resolve_predicate(mfilter_name) if mfilter_name else None
        for coll in cmp.values():
            for bh, mh, _mname, mcoord, sim_coord in coll["pairs"]:
                if method_pred is not None and not method_pred(mcoord):
                    continue
                if sim_pred is not None and not sim_pred(sim_coord):
                    continue
                p = reduction_output(bh, mh, reduction)
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Plot-ready bundle assembly
# ---------------------------------------------------------------------------

def method_metadata(methods: dict[str, dict]) -> pl.DataFrame:
    """Per-method display metadata. Logistic methods have no threshold/oracle/f1
    semantics, so those columns are constant (None / False)."""
    rows = []
    for name, spec in methods.items():
        family = name.split("__")[0]  # strip over-suffix (e.g. __L=5) for label lookup
        L = int(spec["kwargs"].get("L", 1))
        suffix = "SER" if L == 1 else f"SuSiE [L={L}]"
        from viz_utils import method_family_label_map
        family_label = method_family_label_map().get(family, family)
        base = f"{family_label} {suffix}"
        rows.append({
            "method": name,
            "method_family": family,
            "L": L,
            "threshold": None,
            "is_thresholded": False,
            "is_oracle": False,
            "oracle_label": "Oracle",
            "method_label_base": base,
            "method_display": base,
            "method_display_base": base,
        })
    return pl.from_dicts(rows, schema_overrides={"threshold": pl.Float64})


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
        method_pred = resolve_predicate(mfilter_name) if mfilter_name else None
        frames = []
        for info in cmp.values():
            sub = []
            for bh, mh, _mname, mcoord, sim_coord in info["pairs"]:
                if method_pred is not None and not method_pred(mcoord):
                    continue
                if sim_pred is not None and not sim_pred(sim_coord):
                    continue
                path = f"{results_root}/{reduction_output(bh, mh, reduction).split('/', 1)[1]}"
                sub.append(pl.read_parquet(path))
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
