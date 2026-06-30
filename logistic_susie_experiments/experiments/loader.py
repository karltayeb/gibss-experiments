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
from viz_dims import method_dims, sim_dims

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

def load_library(experiments_dir: Path | None = None) -> dict[str, Any]:
    """Load library.yaml. Enrichments are literal {function, arguments, intercept}
    entries (sweep blocks generate their own inline); no family expansion."""
    import yaml
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    data = yaml.safe_load((base / "library.yaml").read_text(encoding="utf-8")) or {}
    for section in ("defaults", "designs", "enrichments", "methods", "reductions", "analyses"):
        data.setdefault(section, {})
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

def _expand_sweep(library: dict[str, Any], sc_name: str, block: dict[str, Any]) -> list[dict]:
    """Expand a sweep block into one collection per rho value.

    Each collection is the full ``b0 × target_logbf`` grid (cartesian). Include
    ``target_logbf: 0`` in the list to get null simulations (no causal) — there is
    no implicit/auto-appended null.
    """
    from simulations.effect import logbf_sizing
    sw = block["sweep"]
    dspec = dict(sw["design"])
    fn = dspec.pop("function")
    freq = dspec.get("freq")
    design_key = logbf_sizing.DESIGN_KEY[(fn, freq if fn == "binary_markov_X" else None)]
    rhos = list(sw["rho"])
    b0s = list(sw["b0"])
    lbfs = list(sw["target_logbf"])
    effect_fn = sw.get("effect", "logbf_single_effect")     # e.g. logbf_k_effects
    effect_args = dict(sw.get("effect_args", {}))           # e.g. {k: 3}
    base_seed = int(library["defaults"]["base_seed"])
    results: list[dict] = []
    for rho in rhos:
        design_coord = {"function": fn, "arguments": {**dspec, "rho": rho}}
        sims, coords = [], []
        cells = [(b0, L) for b0 in b0s for L in lbfs]   # target_logbf=0 -> null
        for b0, L in cells:
            enr = {
                "function": effect_fn,
                "arguments": {"design": design_key, "b0": float(b0), "target_logbf": int(L), **effect_args},
                "intercept": float(b0),
            }
            coord = {"design": design_coord, "enrichment": enr, "base_seed": base_seed}
            name = f"{design_key}__b0={b0:g}__" + (f"lbf={L:g}" if L else "null")
            spec = _spec_from_coord(coord, f"{sc_name}__{name}")
            sims.append(spec)
            coords.append({"coordinate": coord, "name": spec.name})
        results.append({
            "name": f"{sc_name}__{block.get('name', design_key)}__rho={rho:g}",
            "alias": f"{block.get('name', design_key)} rho={rho:g}",
            "simulations": sims,
            "coordinates": coords,
        })
    return results


def _expand_block(library: dict[str, Any], sc_name: str, block: dict[str, Any]) -> list[dict]:
    if "sweep" in block:
        return _expand_sweep(library, sc_name, block)

    if "simulations" in block:  # explicit one-off list
        coords, sims = [], []
        for s in block["simulations"]:
            coord = simulation_coordinate(library, s["design"], s["enrichment"])
            spec = resolve_simulation(library, s["design"], s["enrichment"])
            coords.append({"coordinate": coord, "name": spec.name})
            sims.append(spec)
        return [{"name": block["name"], "alias": block.get("alias", block["name"]),
                 "simulations": sims, "coordinates": coords}]

    raise ValueError(
        f"collection block in {sc_name!r} must use 'sweep' or 'simulations'; the "
        f"'template'/'over' forms were removed (faceting replaced over; sweep is "
        f"the fit enumerator). Got keys: {sorted(block)}"
    )


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

_CHANNEL_KEYS = ("color", "linestyle", "facet_row", "facet_col", "facet_wrap", "ncol")


def resolve_plot_specs(config: dict, sc_name: str) -> dict:
    sc = config["supercollections"][sc_name]
    out = {}
    for name, spec in (sc.get("plots") or {}).items():
        # a plot is either a single `analysis` or a `panels` list (dashboard)
        if not any(k in spec for k in ("analysis", "panels", "columns", "layout")):
            raise KeyError(f"plot {name!r} in {sc_name!r} needs 'analysis', 'panels', 'columns', or 'layout'")
        out[name] = {
            "name": name,
            "analysis": spec.get("analysis"),
            "panels": list(spec["panels"]) if spec.get("panels") else None,
            "columns": [list(c) for c in spec["columns"]] if spec.get("columns") else None,
            "rows": list(spec["rows"]) if spec.get("rows") else None,
            "designs": list(spec["designs"]) if spec.get("designs") else None,
            "layout": spec.get("layout"),
            "bottom_panels": list(spec["bottom_panels"]) if spec.get("bottom_panels") else None,
            "wrap_panel": spec.get("wrap_panel"),
            "inset": dict(spec.get("inset") or {}),
            "filter": dict(spec.get("filter") or {}),
            **{k: spec.get(k) for k in _CHANNEL_KEYS},
        }
    return out


def resolve_plot_spec(config: dict, sc_name: str, plot_name: str) -> dict:
    specs = resolve_plot_specs(config, sc_name)
    if plot_name not in specs:
        raise KeyError(f"No plot named {plot_name!r} in {sc_name!r}")
    return specs[plot_name]


def plot_bucket(spec: dict) -> str:
    """Output-dir bucket for a plot: the analysis name for single-analysis plots,
    'summary' for multi-analysis dashboards."""
    return "summary" if (spec.get("panels") or spec.get("columns") or spec.get("layout")) else spec["analysis"]


def plot_reductions(config: dict, sc_name: str, plot_name: str) -> list[str]:
    """Reduction keys a plot consumes (union over its panels, or its single analysis)."""
    from analyses.hooks import HOOKS
    spec = resolve_plot_spec(config, sc_name, plot_name)
    if spec.get("layout"):
        if spec.get("rows"):           # thematic layout: analyses listed in the spec
            analyses = list(spec["rows"])
        else:                          # bespoke layout: static analysis list
            import generate_plots
            analyses = list(generate_plots.LAYOUT_ANALYSES[spec["layout"]])
    elif spec.get("columns"):
        analyses = [a for col in spec["columns"] for a in col]
    elif spec.get("panels"):
        analyses = list(spec["panels"])
    else:
        analyses = [spec["analysis"]]
    if spec.get("bottom_panels"):
        analyses += spec["bottom_panels"]
    if spec.get("wrap_panel"):
        analyses.append(spec["wrap_panel"])
    return sorted({HOOKS[a].requires for a in analyses})


def plot_analyses(config: dict, sc_name: str) -> list[tuple]:
    """(bucket, plot_name) pairs — bucket is the output-dir component."""
    return [(plot_bucket(s), s["name"]) for s in resolve_plot_specs(config, sc_name).values()]


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
    from manifest_cache import load_manifest_cached
    manifest = load_manifest_cached()
    method_dim_by_hash = {h: method_dims(m) for h, m in manifest["methods"].items()}
    sim_dim_by_batch = {h: sim_dims(b["coordinate"]) for h, b in manifest["batches"].items()}
    # map method NAME -> dims (rows carry method name, not hash)
    method_dim_by_name = {m["name"]: method_dim_by_hash[h] for h, m in manifest["methods"].items()}
    for reduction in requires:
        mfilter_name = reduction_method_filter(library, reduction)
        method_pred = resolve_predicate(mfilter_name) if mfilter_name else None
        frames = []
        for info in cmp.values():
            sub = []
            for bh, mh, mname, mcoord, sim_coord in info["pairs"]:
                if method_pred is not None and not method_pred(mcoord):
                    continue
                if sim_pred is not None and not sim_pred(sim_coord):
                    continue
                path = f"{results_root}/{reduction_output(bh, mh, reduction).split('/', 1)[1]}"
                frame = pl.read_parquet(path)
                dims = {**method_dim_by_name.get(mname, {}), **sim_dim_by_batch.get(bh, {})}
                # only scalar dims become columns (skip nested); cast via pl.lit
                lit_cols = [pl.lit(v).alias(k) for k, v in dims.items()
                            if not isinstance(v, (list, dict, tuple))]
                sub.append(frame.with_columns(lit_cols))
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
