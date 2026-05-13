import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo

    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    import plot_ready
    from utils import write_yaml


@app.cell(hide_code=True)
def title_cell():
    _md = mo.md("# Collection Creator")
    _md
    return


@app.cell
def manifest_cell():
    _manifest_path = Path(__file__).parent.parent / "results" / "manifest.json"
    manifest = json.loads(_manifest_path.read_text())
    mo.md(f"Loaded manifest: **{len(manifest['batches'])} batches**, **{len(manifest['method_specs'])} methods**, **{len(manifest['simulation_specs'])} simulations**")
    return (manifest,)


@app.cell
def sim_index_cell(manifest):
    """Parse simulation dimensions from batch nodes for filter UI."""
    def _parse_sim_name(name: str) -> dict:
        parts = name.split("__")
        d = {"name": name, "design": parts[0] if parts else ""}
        if len(parts) > 1:
            d["regime"] = parts[1]
        if len(parts) > 2:
            param_part = parts[2]
            idx = param_part.rfind("_")
            if idx > 0:
                d["parameter"] = param_part[:idx]
                try:
                    d["value"] = float(param_part[idx + 1:])
                except ValueError:
                    d["parameter"] = param_part
            else:
                d["parameter"] = param_part
        return d

    def _parse_method_node(node: dict) -> dict:
        fields = node.get("fields", {})
        name = fields.get("name", "")
        kwargs = fields.get("kwargs", {})
        return {
            "hash": node["__spec_hash__"],
            "name": name,
            "L": kwargs.get("L"),
            "threshold": kwargs.get("threshold"),
            "family": name.rsplit("_L", 1)[0] if "_L" in name else name,
        }

    # batch hash → {batch_node, sim_dims}
    batch_index = {}
    for _bh, _bn in manifest["batches"].items():
        _sim_name = _bn["simulation_spec"]["fields"]["name"]
        batch_index[_bh] = {**_bn, "sim_dims": _parse_sim_name(_sim_name)}

    method_index = {
        mh: _parse_method_node(mn)
        for mh, mn in manifest["method_specs"].items()
    }

    return batch_index, method_index


@app.cell
def mode_cell():
    mode_tabs = mo.ui.tabs({
        "Build": mo.md(""),
        "Batch-generate": mo.md(""),
        "Compose": mo.md(""),
    })
    mode_tabs
    return (mode_tabs,)


@app.cell
def build_sim_selector_cell(batch_index, mode_tabs):
    mo.stop(mode_tabs.value != "Build")

    # sim spec name → list of batch hashes
    sim_to_batches: dict[str, list[str]] = {}
    for _bh, _bv in batch_index.items():
        _sname = _bv["simulation_spec"]["fields"]["name"]
        sim_to_batches.setdefault(_sname, []).append(_bh)

    _sim_names = sorted(sim_to_batches.keys())

    sim_spec_select = mo.ui.multiselect(
        options=_sim_names,
        value=_sim_names,
        label="Simulation specs",
    )

    _all_btn  = mo.ui.button(label="All",  on_click=lambda _: sim_spec_select.set_value(_sim_names))
    _none_btn = mo.ui.button(label="None", on_click=lambda _: sim_spec_select.set_value([]))

    _n_batches = sum(len(sim_to_batches[s]) for s in sim_spec_select.value)
    mo.vstack([
        mo.hstack([_all_btn, _none_btn]),
        sim_spec_select,
        mo.md(f"**{len(sim_spec_select.value)} sim specs → {_n_batches} batches**"),
    ])
    return sim_spec_select, sim_to_batches


@app.cell
def build_method_selector_cell(method_index, mode_tabs):
    mo.stop(mode_tabs.value != "Build")

    def _method_label(mv: dict) -> str:
        if mv["threshold"] is not None:
            return f"{mv['name']} (L={mv['L']}, @{mv['threshold']})"
        return f"{mv['name']} (L={mv['L']})"

    _all_hashes = sorted(method_index.keys(), key=lambda h: _method_label(method_index[h]))
    _method_options = {h: _method_label(method_index[h]) for h in _all_hashes}
    _nothresh_hashes = [h for h in _all_hashes if method_index[h]["threshold"] is None]

    method_spec_select = mo.ui.multiselect(
        options=_method_options,
        value=_all_hashes,
        label="Method specs",
    )

    _all_btn      = mo.ui.button(label="All",          on_click=lambda _: method_spec_select.set_value(_all_hashes))
    _nothresh_btn = mo.ui.button(label="No threshold", on_click=lambda _: method_spec_select.set_value(_nothresh_hashes))
    _none_btn     = mo.ui.button(label="None",         on_click=lambda _: method_spec_select.set_value([]))

    mo.vstack([
        mo.hstack([_all_btn, _nothresh_btn, _none_btn]),
        method_spec_select,
        mo.md(f"**{len(method_spec_select.value)} method specs selected**"),
    ])
    return (method_spec_select,)


@app.cell
def build_write_cell(sim_spec_select, sim_to_batches, method_spec_select, manifest, mode_tabs):
    mo.stop(mode_tabs.value != "Build")

    collection_name_input = mo.ui.text(placeholder="my_collection_name", label="Collection name")

    _selected_batch_hashes = [
        bh for sname in sim_spec_select.value for bh in sim_to_batches[sname]
    ]
    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_write(_btn):
        mo.stop(not collection_name_input.value.strip())
        _batch_nodes = [manifest["batches"][h] for h in _selected_batch_hashes]
        _method_nodes = [manifest["method_specs"][h] for h in method_spec_select.value]
        _node = plot_ready.build_collection_yaml_node(
            name=collection_name_input.value.strip(),
            batch_nodes=_batch_nodes,
            method_nodes=_method_nodes,
        )
        _collections_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(_node, str(_collections_dir / f"{collection_name_input.value.strip()}.yaml"))

    write_btn = mo.ui.button(label="Write collection", on_click=_do_write)
    mo.vstack([
        collection_name_input,
        mo.md(f"**{len(_selected_batch_hashes)} batches × {len(method_spec_select.value)} method specs**"),
        write_btn,
    ])
    return (write_btn,)


@app.cell
def batchgen_cell(batch_index, manifest, method_index, mode_tabs):
    mo.stop(mode_tabs.value != "Batch-generate")

    _all_method_options = {
        mh: f"{mv['name']} (L={mv['L']})"
        for mh, mv in method_index.items()
    }

    batchgen_method_select = mo.ui.multiselect(
        options=_all_method_options,
        value=list(_all_method_options.keys()),
        label="Methods to include in every generated collection",
    )

    # one collection per unique simulation name (grouping all its batches)
    sim_to_batches: dict[str, list[str]] = {}
    for _bh, _bv in batch_index.items():
        _sname = _bv["simulation_spec"]["fields"]["name"]
        sim_to_batches.setdefault(_sname, []).append(_bh)

    _preview_rows = [
        {"collection": sim_name, "n_batches": len(batch_hashes)}
        for sim_name, batch_hashes in sorted(sim_to_batches.items())
    ]

    import polars as _pl
    _preview_df = _pl.from_dicts(_preview_rows)

    mo.vstack([
        batchgen_method_select,
        mo.md(f"Will generate **{len(sim_to_batches)} collection yamls**, one per simulation."),
        mo.ui.table(_preview_df, selection=None),
    ])
    return batchgen_method_select, sim_to_batches


@app.cell
def batchgen_write_cell(batchgen_method_select, manifest, mode_tabs, sim_to_batches):
    mo.stop(mode_tabs.value != "Batch-generate")

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_batchgen(_btn):
        _collections_dir.mkdir(parents=True, exist_ok=True)
        _method_nodes = [manifest["method_specs"][h] for h in batchgen_method_select.value]
        for _sim_name, _batch_hashes in sim_to_batches.items():
            _batch_nodes = [manifest["batches"][h] for h in _batch_hashes]
            _node = plot_ready.build_collection_yaml_node(
                name=_sim_name,
                batch_nodes=_batch_nodes,
                method_nodes=_method_nodes,
            )
            write_yaml(_node, str(_collections_dir / f"{_sim_name}.yaml"))

    batchgen_write_btn = mo.ui.button(
        label=f"Generate {len(sim_to_batches)} collection yamls",
        on_click=_do_batchgen,
    )
    batchgen_write_btn
    return (batchgen_write_btn,)


@app.cell
def compose_cell(mode_tabs):
    mo.stop(mode_tabs.value != "Compose")

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"
    _existing = {
        p.stem: str(p)
        for p in sorted(_collections_dir.glob("*.yaml"))
    }

    compose_select = mo.ui.multiselect(
        options=_existing,
        label="Collections to union",
    )
    compose_name_input = mo.ui.text(placeholder="my_union_collection", label="Union collection name")

    if compose_select.value:
        import yaml as _yaml
        _nodes = [_yaml.safe_load(open(p)) for p in compose_select.value]
        _all_batches = {b["__spec_hash__"] for n in _nodes for b in n["batches"]}
        _all_methods = {m["__spec_hash__"] for n in _nodes for m in n["method_specs"]}
        _preview_md = mo.md(
            f"Union: **{len(_all_batches)} batches**, **{len(_all_methods)} methods** (after dedup)"
        )
    else:
        _preview_md = mo.md("Select collections above to preview the union.")

    mo.vstack([compose_select, compose_name_input, _preview_md])
    return compose_name_input, compose_select


@app.cell
def compose_write_cell(compose_name_input, compose_select, mode_tabs):
    mo.stop(mode_tabs.value != "Compose")
    mo.stop(not compose_name_input.value.strip())
    mo.stop(not compose_select.value)

    import yaml as _yaml
    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_compose(_btn):
        _nodes = [_yaml.safe_load(open(p)) for p in compose_select.value]
        _union_node = plot_ready.union_collection_yaml_nodes(
            name=compose_name_input.value.strip(),
            collection_nodes=_nodes,
        )
        _out = _collections_dir / f"{compose_name_input.value.strip()}.yaml"
        write_yaml(_union_node, str(_out))

    compose_write_btn = mo.ui.button(
        label=f"Write union to {compose_name_input.value.strip()}.yaml",
        on_click=_do_compose,
    )
    compose_write_btn
    return (compose_write_btn,)
