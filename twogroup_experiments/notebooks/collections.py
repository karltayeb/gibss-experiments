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

    # sim spec name → list of batch hashes (shared by Build and Batch-generate)
    sim_to_batches: dict[str, list[str]] = {}
    for _bh, _bv in batch_index.items():
        sim_to_batches.setdefault(_bv["simulation_spec"]["fields"]["name"], []).append(_bh)

    return batch_index, method_index, sim_to_batches


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
def build_sim_table_cell(batch_index, mode_tabs, sim_to_batches):
    mo.stop(mode_tabs.value != "Build")
    import polars as _pl

    _rows = [
        {
            "sim_name": sname,
            "design":    batch_index[bhashes[0]]["sim_dims"].get("design", ""),
            "regime":    batch_index[bhashes[0]]["sim_dims"].get("regime", ""),
            "parameter": batch_index[bhashes[0]]["sim_dims"].get("parameter", ""),
            "value":     batch_index[bhashes[0]]["sim_dims"].get("value", ""),
            "n_batches": len(bhashes),
        }
        for sname, bhashes in sorted(sim_to_batches.items())
    ]
    sim_table = mo.ui.table(_pl.DataFrame(_rows), selection="multi")
    sim_table
    return (sim_table,)


@app.cell
def build_method_table_cell(method_index, mode_tabs):
    mo.stop(mode_tabs.value != "Build")
    import polars as _pl

    _rows = [
        {
            "hash":      h,
            "name":      mv["name"],
            "family":    mv["family"],
            "L":         mv["L"],
            "threshold": mv["threshold"],
        }
        for h, mv in sorted(
            method_index.items(),
            key=lambda x: (x[1]["family"], x[1]["L"] or 0, x[1]["threshold"] or -1),
        )
    ]
    method_table = mo.ui.table(_pl.DataFrame(_rows), selection="multi")
    method_table
    return (method_table,)


@app.cell
def build_write_cell(manifest, method_table, mode_tabs, sim_table, sim_to_batches):
    mo.stop(mode_tabs.value != "Build")

    collection_name_input = mo.ui.text(placeholder="my_collection_name", label="Collection name")

    _selected_sims = sim_table.value["sim_name"].to_list() if len(sim_table.value) else []
    _selected_method_hashes = method_table.value["hash"].to_list() if len(method_table.value) else []
    _selected_batch_hashes = [bh for s in _selected_sims for bh in sim_to_batches[s]]

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_write(_btn):
        mo.stop(not collection_name_input.value.strip())
        _node = plot_ready.build_collection_yaml_node(
            name=collection_name_input.value.strip(),
            batch_nodes=[manifest["batches"][h] for h in _selected_batch_hashes],
            method_nodes=[manifest["method_specs"][h] for h in _selected_method_hashes],
        )
        _coll_dir = _collections_dir / collection_name_input.value.strip()
        _coll_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(_node, str(_coll_dir / "collection_spec.yaml"))

    write_btn = mo.ui.button(label="Write collection", on_click=_do_write)
    mo.vstack([
        collection_name_input,
        mo.md(f"**{len(_selected_batch_hashes)} batches × {len(_selected_method_hashes)} method specs**"),
        write_btn,
    ])
    return collection_name_input, write_btn


@app.cell
def build_snakemake_cmd_cell(collection_name_input, mode_tabs):
    mo.stop(mode_tabs.value != "Build")
    mo.stop(not collection_name_input.value.strip())
    _name = collection_name_input.value.strip()
    _cmd = f"uv run snakemake --snakefile twogroup_experiments.snk results/collections/{_name}/plot_ready/out.txt"
    mo.md(f"```bash\n{_cmd}\n```")


@app.cell
def batchgen_method_table_cell(method_index, mode_tabs):
    mo.stop(mode_tabs.value != "Batch-generate")
    import polars as _pl

    _rows = [
        {
            "hash":      h,
            "name":      mv["name"],
            "family":    mv["family"],
            "L":         mv["L"],
            "threshold": mv["threshold"],
        }
        for h, mv in sorted(
            method_index.items(),
            key=lambda x: (x[1]["family"], x[1]["L"] or 0, x[1]["threshold"] or -1),
        )
    ]
    batchgen_method_table = mo.ui.table(_pl.DataFrame(_rows), selection="multi")
    batchgen_method_table
    return (batchgen_method_table,)


@app.cell
def batchgen_cell(batchgen_method_table, mode_tabs):
    mo.stop(mode_tabs.value != "Batch-generate")
    batchgen_suffix_input = mo.ui.text(placeholder="e.g. _ser or _L5  (optional)", label="Collection name suffix")
    batchgen_suffix_input
    return (batchgen_suffix_input,)


@app.cell
def batchgen_preview_cell(batchgen_method_table, batchgen_suffix_input, mode_tabs, sim_to_batches):
    mo.stop(mode_tabs.value != "Batch-generate")
    import polars as _pl

    _preview_rows = [
        {"collection_name": sim_name + batchgen_suffix_input.value, "n_batches": len(batch_hashes)}
        for sim_name, batch_hashes in sorted(sim_to_batches.items())
    ]
    mo.vstack([
        mo.md(f"Will generate **{len(sim_to_batches)} collection yamls**, **{len(batchgen_method_table.value)} method specs** each."),
        mo.ui.table(_pl.DataFrame(_preview_rows), selection=None),
    ])


@app.cell
def batchgen_write_cell(batchgen_method_table, batchgen_suffix_input, manifest, mode_tabs, sim_to_batches):
    mo.stop(mode_tabs.value != "Batch-generate")

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"
    _suffix = batchgen_suffix_input.value
    _n_methods = len(batchgen_method_table.value)

    def _do_batchgen(_btn):
        _method_nodes = [manifest["method_specs"][h] for h in batchgen_method_table.value["hash"].to_list()]
        for _sim_name, _batch_hashes in sim_to_batches.items():
            _coll_name = _sim_name + _suffix
            _coll_dir = _collections_dir / _coll_name
            _coll_dir.mkdir(parents=True, exist_ok=True)
            _batch_nodes = [manifest["batches"][h] for h in _batch_hashes]
            _node = plot_ready.build_collection_yaml_node(
                name=_coll_name,
                batch_nodes=_batch_nodes,
                method_nodes=_method_nodes,
            )
            write_yaml(_node, str(_coll_dir / "collection_spec.yaml"))

    batchgen_write_btn = mo.ui.button(
        label=f"Generate {len(sim_to_batches)} yamls ({_n_methods} methods each)",
        on_click=_do_batchgen,
    )
    batchgen_write_btn
    return (batchgen_write_btn,)


@app.cell
def compose_cell(mode_tabs):
    mo.stop(mode_tabs.value != "Compose")
    import yaml as _yaml
    import polars as _pl

    def _sim_dims(name: str) -> dict:
        parts = name.split("__")
        d = {"design": parts[0] if parts else ""}
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
        return d

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"
    _rows = []
    for _p in sorted(_collections_dir.glob("*/collection_spec.yaml")):
        _c = _yaml.safe_load(_p.read_text())
        _batches = _c.get("batches", [])
        _sim_names = list({b["simulation_spec"]["fields"]["name"] for b in _batches})
        if len(_sim_names) == 1:
            _sd = _sim_dims(_sim_names[0])
            _sim_col   = _sim_names[0]
            _design    = _sd.get("design", "")
            _regime    = _sd.get("regime", "")
            _parameter = _sd.get("parameter", "")
            _value     = _sd.get("value", "")
        else:
            _sim_col   = f"mixed ({len(_sim_names)})"
            _design = _regime = _parameter = _value = ""
        _rows.append({
            "name":      _c.get("name", _p.parent.name),
            "sim_spec":  _sim_col,
            "design":    _design,
            "regime":    _regime,
            "parameter": _parameter,
            "value":     _value,
            "n_batches": len(_batches),
            "n_methods": len(_c.get("method_specs", [])),
            "path":      str(_p),
        })

    _empty = _pl.DataFrame({"name": [], "sim_spec": [], "design": [], "regime": [], "parameter": [], "value": [], "n_batches": [], "n_methods": [], "path": []})
    compose_table = mo.ui.table(_pl.DataFrame(_rows) if _rows else _empty, selection="multi")
    compose_name_input = mo.ui.text(placeholder="my_union_collection", label="Union collection name")

    mo.vstack([compose_table, compose_name_input])
    return compose_name_input, compose_table


@app.cell
def compose_preview_cell(compose_table, mode_tabs):
    mo.stop(mode_tabs.value != "Compose")
    import yaml as _yaml

    if len(compose_table.value):
        _nodes = [_yaml.safe_load(open(p)) for p in compose_table.value["path"].to_list()]
        _all_batches = {b["__spec_hash__"] for n in _nodes for b in n["batches"]}
        _all_methods = {m["__spec_hash__"] for n in _nodes for m in n["method_specs"]}
        mo.md(f"Union: **{len(_all_batches)} batches**, **{len(_all_methods)} methods** (after dedup)")
    else:
        mo.md("Select collections above to preview the union.")


@app.cell
def compose_write_cell(compose_name_input, compose_table, mode_tabs):
    mo.stop(mode_tabs.value != "Compose")
    mo.stop(not compose_name_input.value.strip())
    mo.stop(not len(compose_table.value))

    import yaml as _yaml
    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_compose(_btn):
        _nodes = [_yaml.safe_load(open(p)) for p in compose_table.value["path"].to_list()]
        _union_node = plot_ready.union_collection_yaml_nodes(
            name=compose_name_input.value.strip(),
            collection_nodes=_nodes,
        )
        _coll_dir = _collections_dir / compose_name_input.value.strip()
        _coll_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(_union_node, str(_coll_dir / "collection_spec.yaml"))

    compose_write_btn = mo.ui.button(
        label=f"Write union to {compose_name_input.value.strip()}/collection_spec.yaml",
        on_click=_do_compose,
    )
    compose_write_btn
    return (compose_write_btn,)
