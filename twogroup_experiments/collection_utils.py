from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from core import HASH_KEY
from plot_ready import build_collection_yaml_node
from utils import write_yaml


def _parse_sim_name(sim_name: str) -> dict[str, Any]:
    parts = sim_name.split("__")
    design = parts[0] if parts else ""
    enrichment = parts[1] if len(parts) > 1 else None
    signal: float | None = None
    if len(parts) > 2:
        param_part = parts[2]
        idx = param_part.rfind("_")
        if idx > 0:
            try:
                signal = float(param_part[idx + 1:])
            except ValueError:
                pass
    return {"design": design, "enrichment": enrichment, "signal": signal}


def _parse_method_spec(method_spec_json: str) -> dict[str, Any]:
    spec = json.loads(method_spec_json)
    fields = spec.get("fields", {})
    name = str(fields.get("name", ""))
    kwargs = dict(fields.get("kwargs", {}))
    L = int(kwargs.get("L", 1))
    method_family = name.rsplit("_L", 1)[0]
    threshold_raw = kwargs.get("threshold")
    return {
        "method_name": name,
        "method_family": method_family,
        "L": L,
        "is_oracle": "oracle" in method_family,
        "is_thresholded": "threshold" in method_family,
        "threshold": float(threshold_raw) if threshold_raw is not None else None,
    }


def build_manifest_table(manifest: dict[str, Any]) -> pl.DataFrame:
    """One row per (sim_spec, method_spec); batch_hashes aggregated per sim_spec."""
    sim_hash_to_batches: dict[str, list[str]] = {}
    sim_hash_to_node: dict[str, dict] = {}

    for batch_hash, batch_node in manifest["batches"].items():
        sim_node = batch_node["simulation_spec"]
        sim_hash = sim_node[HASH_KEY]
        sim_hash_to_batches.setdefault(sim_hash, []).append(batch_hash)
        sim_hash_to_node.setdefault(sim_hash, sim_node)

    rows = []
    for sim_hash, sim_node in sim_hash_to_node.items():
        sim_name = sim_node["fields"]["name"]
        for method_hash, method_node in manifest["method_specs"].items():
            rows.append({
                "sim_hash": sim_hash,
                "method_hash": method_hash,
                "sim_spec": json.dumps(sim_node, sort_keys=True),
                "method_spec": json.dumps(method_node, sort_keys=True),
                "batch_hashes": sim_hash_to_batches[sim_hash],
                "sim_name": sim_name,
            })

    if not rows:
        return pl.DataFrame(schema={
            "sim_hash": pl.String,
            "method_hash": pl.String,
            "sim_spec": pl.String,
            "method_spec": pl.String,
            "batch_hashes": pl.List(pl.String),
            "sim_name": pl.String,
        })

    return pl.from_dicts(rows)


def add_sim_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Add design, enrichment, signal columns parsed from sim_name."""
    parsed = [_parse_sim_name(n) for n in df["sim_name"].to_list()]
    return df.with_columns(
        pl.Series("design", [p["design"] for p in parsed], dtype=pl.String),
        pl.Series("enrichment", [p["enrichment"] for p in parsed], dtype=pl.String),
        pl.Series("signal", [p["signal"] for p in parsed], dtype=pl.Float64),
    )


def add_method_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Add method_name, method_family, L, is_oracle, is_thresholded, threshold."""
    parsed = [_parse_method_spec(ms) for ms in df["method_spec"].to_list()]
    return df.with_columns(
        pl.Series("method_name", [p["method_name"] for p in parsed], dtype=pl.String),
        pl.Series("method_family", [p["method_family"] for p in parsed], dtype=pl.String),
        pl.Series("L", [p["L"] for p in parsed], dtype=pl.Int64),
        pl.Series("is_oracle", [p["is_oracle"] for p in parsed], dtype=pl.Boolean),
        pl.Series("is_thresholded", [p["is_thresholded"] for p in parsed], dtype=pl.Boolean),
        pl.Series("threshold", [p["threshold"] for p in parsed], dtype=pl.Float64),
    )


def collection_name(
    *,
    design: str = "all",
    enrichment: str = "all",
    signal: str = "all",
    method: str = "all",
) -> str:
    return f"design_{design}__enrichment_{enrichment}__signal_{signal}__method_{method}"


def write_collection(
    df: pl.DataFrame,
    name: str,
    collections_dir: Path | str,
    manifest: dict[str, Any],
) -> Path:
    """Write collection YAML to collections_dir/name/collection_spec.yaml."""
    all_batch_hashes: set[str] = set()
    for batch_list in df["batch_hashes"].to_list():
        all_batch_hashes.update(batch_list)

    batch_nodes = [manifest["batches"][h] for h in sorted(all_batch_hashes)]
    method_nodes = [
        manifest["method_specs"][h]
        for h in sorted(df["method_hash"].unique().to_list())
    ]

    node = build_collection_yaml_node(
        name=name,
        batch_nodes=batch_nodes,
        method_nodes=method_nodes,
    )

    coll_dir = Path(collections_dir) / name
    coll_dir.mkdir(parents=True, exist_ok=True)
    out_path = coll_dir / "collection_spec.yaml"
    write_yaml(node, str(out_path))
    print(f"wrote {out_path}  ({len(batch_nodes)} batches × {len(method_nodes)} methods)")
    return out_path
