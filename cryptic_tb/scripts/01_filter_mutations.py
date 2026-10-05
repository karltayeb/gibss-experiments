#!/usr/bin/env python
"""Stream MUTATIONS.csv.gz once, keep the candidate-gene rows, write a compact parquet.

The source table is 1.5 GB gzipped and covers 3,831 genes. Everything downstream needs
only the ~40 genes the WHO catalogue puts in tier 1 or tier 2 for the drugs under study,
so one filtering pass turns it into a file that fits comfortably in memory.

Run from cryptic_tb/:  uv run python scripts/01_filter_mutations.py
"""

from __future__ import annotations

import csv
import gzip
import sys
import time
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from who import DRUG_NAMES, load_catalogue  # noqa: E402

MUTATIONS = ROOT / "data/raw/data_tables/MUTATIONS.csv.gz"
MASTER = ROOT / "data/raw/who_catalogue_2023/Final Result Files/WHO-UCN-TB-2023.6-eng_catalogue_master_file.txt"
OUT = ROOT / "data/processed/mutations_candidate_genes.parquet"

# CRyPTIC names the fabG1-inhA operon promoter under fabG1; WHO books the same bases as
# inhA upstream variants, so fabG1 is in no WHO tier list yet carries the single most
# common isoniazid promoter variant. Add it by hand, and keep the compensatory RIF genes.
EXTRA_GENES = {"fabG1", "rpoA", "rpoC", "pncA", "panD", "rpsA", "ethA", "eis", "rrs"}

KEEP_COLUMNS = (
    "UNIQUEID", "GENE", "MUTATION", "POSITION", "AMINO_ACID_NUMBER", "GENOME_INDEX",
    "NUCLEOTIDE_NUMBER", "REF", "ALT", "IS_SNP", "IS_INDEL", "IN_CDS", "IN_PROMOTER",
    "IS_SYNONYMOUS", "IS_NONSYNONYMOUS", "IS_HET", "IS_NULL", "IS_FILTER_PASS",
    "ELEMENT_TYPE", "MUTATION_TYPE", "INDEL_LENGTH", "NUMBER_NUCLEOTIDE_CHANGES",
)


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    drugs = list(config["drugs"])
    cat = load_catalogue(MASTER)

    genes: set[str] = set(EXTRA_GENES)
    for drug in drugs:
        if drug in cat.tiers:
            genes |= set(cat.tiers[drug])
        # config gene lists are a fallback for drugs the catalogue does not cover
        genes |= set(config["drugs"][drug].get("genes", []))
        genes |= set(config["drugs"][drug].get("diagnostic_genes", []))
    print(f"keeping {len(genes)} genes for drugs {drugs}")
    print("  " + " ".join(sorted(genes)))

    rows: list[tuple] = []
    n_total = 0
    t0 = time.time()
    with gzip.open(MUTATIONS, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        idx = [reader.fieldnames.index(c) for c in KEEP_COLUMNS]
        del idx  # DictReader path below; kept for the column-order assertion
        missing = set(KEEP_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"MUTATIONS is missing columns: {sorted(missing)}")
        for row in reader:
            n_total += 1
            if n_total % 5_000_000 == 0:
                print(f"  {n_total:,} rows read, {len(rows):,} kept, "
                      f"{time.time() - t0:.0f}s", flush=True)
            if row["GENE"] not in genes:
                continue
            rows.append(tuple(row[c] for c in KEEP_COLUMNS))

    print(f"read {n_total:,} rows, kept {len(rows):,} in {time.time() - t0:.0f}s")

    frame = pl.DataFrame(
        {c: [r[i] for r in rows] for i, c in enumerate(KEEP_COLUMNS)},
        schema={c: pl.String for c in KEEP_COLUMNS},
    )
    # Normalise the types the design builder reasons about.
    frame = frame.with_columns(
        [pl.col(c).str.strip_chars() for c in ("GENE", "MUTATION", "ELEMENT_TYPE", "MUTATION_TYPE")]
    ).with_columns(
        [
            pl.col(c).eq("True").alias(c)
            for c in ("IS_SNP", "IS_INDEL", "IN_CDS", "IN_PROMOTER", "IS_SYNONYMOUS",
                      "IS_NONSYNONYMOUS", "IS_HET", "IS_NULL", "IS_FILTER_PASS")
        ]
        + [
            pl.col(c).cast(pl.Float64, strict=False)
            for c in ("POSITION", "AMINO_ACID_NUMBER", "GENOME_INDEX", "NUCLEOTIDE_NUMBER",
                      "INDEL_LENGTH", "NUMBER_NUCLEOTIDE_CHANGES")
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(frame.group_by("GENE").len().sort("len", descending=True).head(15))


if __name__ == "__main__":
    main()
