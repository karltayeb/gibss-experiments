"""
Define collections from the manifest via polars filter/groupby pipelines.

Edit this file and run:
    uv run python make_collections.py

Then run snakemake for any collection you want to plot:
    uv run snakemake --snakefile twogroup_experiments.snk \
        results/collections/<name>/plot_ready/out.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from collection_utils import (
    add_method_metadata,
    add_sim_metadata,
    build_manifest_table,
    collection_name,
    write_collection,
)

RESULTS_ROOT = Path("results")
COLLECTIONS_DIR = RESULTS_ROOT / "collections"
MANIFEST_PATH = RESULTS_ROOT / "manifest.json"

manifest = json.loads(MANIFEST_PATH.read_text())

df = (
    build_manifest_table(manifest)
    .pipe(add_sim_metadata)
    .pipe(add_method_metadata)
)

# ── Inspect available dimensions ─────────────────────────────────────────────
print("designs:    ", sorted(df["design"].unique().to_list()))
print("enrichments:", sorted(df["enrichment"].drop_nulls().unique().to_list()))
print("signals:    ", sorted(df["signal"].drop_nulls().unique().to_list()))
print("families:   ", sorted(df["method_family"].unique().to_list()))
print("thresholds: ", sorted(df["threshold"].drop_nulls().unique().to_list()))
print()

# ── Example collections ───────────────────────────────────────────────────────

# 1. All oracle methods across everything
write_collection(
    df.filter(pl.col("is_oracle")),
    collection_name(method="oracle"),
    COLLECTIONS_DIR,
    manifest,
)

# 2. Threshold-free non-oracle methods (twogroup, cox_heavy) across everything
write_collection(
    df.filter(~pl.col("is_oracle") & ~pl.col("is_thresholded")),
    collection_name(method="threshold_free"),
    COLLECTIONS_DIR,
    manifest,
)

# 3. One collection per design × oracle
for row in (
    df.filter(pl.col("is_oracle"))
    .select("design").unique().sort("design")
    .iter_rows(named=True)
):
    write_collection(
        df.filter(pl.col("is_oracle") & (pl.col("design") == row["design"])),
        collection_name(design=row["design"], method="oracle"),
        COLLECTIONS_DIR,
        manifest,
    )

# 4. Logistic threshold sweep — one collection per (design, enrichment)
for row in (
    df.filter(pl.col("method_family") == "logistic_threshold")
    .select("design", "enrichment").unique().sort("design", "enrichment")
    .iter_rows(named=True)
):
    write_collection(
        df.filter(
            (pl.col("method_family") == "logistic_threshold")
            & (pl.col("design") == row["design"])
            & (pl.col("enrichment") == row["enrichment"])
        ),
        collection_name(
            design=row["design"],
            enrichment=row["enrichment"],
            method="logistic_threshold_sweep",
        ),
        COLLECTIONS_DIR,
        manifest,
    )

# 5. All methods, one collection per (design, enrichment, signal)
for row in (
    df.select("design", "enrichment", "signal").unique()
    .sort("design", "enrichment", "signal")
    .iter_rows(named=True)
):
    signal_str = f"{row['signal']:g}" if row["signal"] is not None else "all"
    write_collection(
        df.filter(
            (pl.col("design") == row["design"])
            & (pl.col("enrichment") == row["enrichment"])
            & (
                pl.col("signal").is_null()
                if row["signal"] is None
                else pl.col("signal") == row["signal"]
            )
        ),
        collection_name(
            design=row["design"],
            enrichment=row["enrichment"],
            signal=signal_str,
        ),
        COLLECTIONS_DIR,
        manifest,
    )
