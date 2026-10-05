#!/usr/bin/env python
"""Build the per-drug design: y, X over candidate-gene variants, covariates, feature table.

Writes data/processed/{drug}/ with
    X.npz          scipy CSR, isolates x features, 0/1 carrier matrix
    features.csv   one row per column of X, with the WHO grade joined on
    y.csv          isolate id, phenotype, quality, and the covariate columns
    collapsed.csv  feature groups that share a carrier set exactly

Run from cryptic_tb/:  uv run python scripts/02_build_design.py
"""

from __future__ import annotations

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from who import cryptic_key, load_catalogue, load_coordinates  # noqa: E402

RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
MUTS = PROC / "mutations_candidate_genes.parquet"
MASTER = RAW / "who_catalogue_2023/Final Result Files/WHO-UCN-TB-2023.6-eng_catalogue_master_file.txt"
COORDS = RAW / "who_catalogue_2023/Final Result Files/WHO-UCN-TB-2023.7-eng_genomic_coordinates.txt"
REUSE = RAW / "reuse/CRyPTIC_reuse_table_20240917.csv"
GENOMES = RAW / "data_tables/GENOMES.csv.gz"
PHENO = RAW / "data_tables/UKMYC_PHENOTYPES.csv.gz"

# CRyPTIC's fabG1 promoter calls are WHO's inhA upstream variants; the gene belongs to the
# isoniazid predictor set even though no WHO tier names it.
GENE_EXTRAS = {"INH": {"fabG1"}, "ETH": {"fabG1"}}


def reuse_table() -> dict[str, dict[str, str]]:
    with open(REUSE, newline="") as fh:
        return {r["UNIQUEID"]: r for r in csv.DictReader(fh)}


def plate_and_lineage(ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Plate design from UKMYC_PHENOTYPES, Mykrobe lineage from GENOMES."""
    plate: dict[str, str] = {}
    with gzip.open(PHENO, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            u = r["UNIQUEID"]
            if u in ids and u not in plate:
                plate[u] = r["PLATEDESIGN"]
    lineage: dict[str, str] = {}
    with gzip.open(GENOMES, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            u = r["UNIQUEID"]
            if u in ids and u not in lineage:
                lineage[u] = r["MYKROBE_LINEAGE_NAME_1"] or "Unknown"
    return plate, lineage


def annotate_grade(cat, coords, drug, gene, mutation, row) -> tuple[int | None, str | None, str]:
    """WHO grade for one CRyPTIC feature. Returns (grade, who_name, how_it_was_found).

    Coordinate first: it is nomenclature-free, so it catches the variants the two
    sources file under different genes. Canonical key second. Gene-level LoF last, for
    the frameshifts WHO grades only as a collapsed `{gene}_LoF` feature.
    """
    gi = row["GENOME_INDEX"]
    ref, alt = (row["REF"] or "").upper(), (row["ALT"] or "").upper()
    if gi is not None and not np.isnan(gi) and ref and alt and alt not in ("X", "O"):
        for name in coords.get((int(gi), ref, alt), ()):  # usually one
            grade = cat.grade_by_name(drug, name)
            if grade is not None:
                return grade, name, "coordinate"

    key = cryptic_key(
        gene, mutation,
        in_promoter=bool(row["IN_PROMOTER"]),
        element_type=row["ELEMENT_TYPE"] or "GENE",
        aa_number=None if row["AMINO_ACID_NUMBER"] is None or np.isnan(row["AMINO_ACID_NUMBER"])
        else int(row["AMINO_ACID_NUMBER"]),
        indel_length=None if row["INDEL_LENGTH"] is None or np.isnan(row["INDEL_LENGTH"])
        else int(row["INDEL_LENGTH"]),
    )
    if key is not None:
        grade = cat.grade(drug, key)
        if grade is not None:
            return grade, cat.name(drug, key), "name"
        if key[0] == "fs":
            lof = cat.lof_grades.get(drug, {}).get(gene)
            if lof is not None:
                return lof, f"{gene}_LoF", "lof_fallback"
    return None, None, "unmatched"


def site_key(row) -> tuple | None:
    """The genomic site a CRyPTIC row speaks about, independent of the allele called.

    Needed because a no-call is not recorded against the variant it obscures. CRyPTIC
    writes an uncalled base as its own mutation (`rpoB` `c-61x`, alt `x`) or a
    filter-failed one with alt `o`, so the null for a site arrives under a different
    MUTATION string than the real variant at that site. Counting missingness per feature
    therefore has to go through the site, not the name. Nucleotide-level rows carry
    GENOME_INDEX; amino-acid rows carry only the codon number.
    """
    gi = row["GENOME_INDEX"]
    if gi is not None and not np.isnan(gi):
        return (row["GENE"], "pos", int(gi))
    aa = row["AMINO_ACID_NUMBER"]
    if aa is not None and not np.isnan(aa):
        return (row["GENE"], "codon", int(aa))
    return None


def is_lof(row) -> bool:
    """Does this CRyPTIC call knock the gene out?

    Frameshift (an indel whose length is not a multiple of three), a premature stop
    (CRyPTIC writes the stop codon as `!`), or a lost start codon. These are the variants
    WHO grades only as a collapsed `{gene}_LoF` feature, because individually they are
    too rare to grade -- which is exactly why a per-variant model struggles with them.
    """
    mutation = row["MUTATION"] or ""
    if row["IS_INDEL"]:
        length = row["INDEL_LENGTH"]
        return length is not None and not np.isnan(length) and int(length) % 3 != 0
    if mutation.endswith("!"):
        return True
    if mutation.startswith("M1") and len(mutation) > 2 and mutation[-1] not in "0123456789":
        return True
    return False


def build_drug(drug: str, cfg, cat, coords, muts: pl.DataFrame, reuse, plate, lineage) -> dict:
    genes = set(cat.tiers.get(drug, {})) | set(cfg["drugs"][drug].get("genes", []))
    genes |= set(cfg["drugs"][drug].get("diagnostic_genes", []))
    genes |= GENE_EXTRAS.get(drug, set())

    keep_quality = set(cfg["phenotype"]["quality_keep"])
    probe = next(iter(reuse.values()))
    if f"{drug}_BINARY_PHENOTYPE" not in probe:
        # The UKMYC plates do not carry every drug: pyrazinamide in particular is absent
        # from the CRyPTIC reuse table, so a PZA arm cannot be built from this cohort.
        print(f"[{drug}] no {drug}_BINARY_PHENOTYPE column in the reuse table; skipping")
        return {"drug": drug, "n": 0, "cases": 0, "genes": len(genes),
                "features_before_mac": 0, "features": 0, "collapsed_pairs": 0,
                "missing_calls": 0, "graded": 0, "grade_1_2": 0,
                "note": "drug not phenotyped in CRyPTIC"}
    isolates = [
        u for u, r in reuse.items()
        if r.get(f"{drug}_BINARY_PHENOTYPE") in ("R", "S")
        and r.get(f"{drug}_PHENOTYPE_QUALITY") in keep_quality
    ]
    isolates.sort()
    idx = {u: i for i, u in enumerate(isolates)}
    n = len(isolates)
    y = np.array([1.0 if reuse[u][f"{drug}_BINARY_PHENOTYPE"] == "R" else 0.0 for u in isolates])

    sub = muts.filter(pl.col("GENE").is_in(list(genes)))
    if cfg["genotype"]["drop_synonymous"]:
        sub = sub.filter(~pl.col("IS_SYNONYMOUS"))

    # carriers[(gene, mutation)] -> set of isolate row indices; missing calls tracked apart
    carriers: dict[tuple[str, str], set[int]] = defaultdict(set)
    missing: dict[tuple[str, str], set[int]] = defaultdict(set)
    meta: dict[tuple[str, str], dict] = {}
    lof_carriers: dict[str, set[int]] = defaultdict(set)
    # isolates whose genotype at a given site is unknown, keyed by site not by allele
    null_at_site: dict[tuple, set[int]] = defaultdict(set)
    feature_site: dict[tuple[str, str], tuple] = {}
    n_rows_used = n_missing_calls = 0
    for row in sub.iter_rows(named=True):
        i = idx.get(row["UNIQUEID"])
        if i is None:
            continue
        feat = (row["GENE"], row["MUTATION"])
        meta.setdefault(feat, row)
        # A failed filter or an explicit null is not evidence of the reference allele.
        if row["IS_NULL"] or not row["IS_FILTER_PASS"]:
            sk = site_key(row)
            if sk is not None:
                null_at_site[sk].add(i)
            missing[feat].add(i)
            n_missing_calls += 1
            continue
        carriers[feat].add(i)
        sk = site_key(row)
        if sk is not None:
            feature_site.setdefault(feat, sk)
        if is_lof(row):
            lof_carriers[row["GENE"]].add(i)
        n_rows_used += 1

    min_mac = cfg["genotype"]["min_mac"]
    feats = sorted(f for f, c in carriers.items() if len(c) >= min_mac)

    # Collapse columns with an identical carrier set: they are perfectly confounded and
    # SuSiE cannot tell them apart, so they must enter as one feature.
    by_pattern: dict[frozenset, list[tuple[str, str]]] = defaultdict(list)
    for f in feats:
        by_pattern[frozenset(carriers[f])].append(f)
    groups = sorted(by_pattern.values(), key=lambda g: (-len(carriers[g[0]]), g[0]))

    rows_i, cols_i = [], []
    records = []
    for j, group in enumerate(groups):
        rep = group[0]
        for i in sorted(carriers[rep]):
            rows_i.append(i)
            cols_i.append(j)
        gene, mutation = rep
        row = meta[rep]
        grade, who_name, how = annotate_grade(cat, coords, drug, gene, mutation, row)
        records.append({
            "column": j,
            "gene": gene,
            "mutation": mutation,
            "feature": f"{gene}_{mutation}",
            "n_carriers": len(carriers[rep]),
            "n_carriers_resistant": int(y[sorted(carriers[rep])].sum()),
            # isolates imputed as reference at this variant's site because the call failed
            "n_missing": len(null_at_site.get(feature_site.get(rep), ())),
            "genome_index": None if row["GENOME_INDEX"] is None or np.isnan(row["GENOME_INDEX"])
            else int(row["GENOME_INDEX"]),
            "amino_acid_number": None if row["AMINO_ACID_NUMBER"] is None
            or np.isnan(row["AMINO_ACID_NUMBER"]) else int(row["AMINO_ACID_NUMBER"]),
            "is_indel": bool(row["IS_INDEL"]),
            "in_promoter": bool(row["IN_PROMOTER"]),
            "element_type": row["ELEMENT_TYPE"],
            "tier": cat.tiers.get(drug, {}).get(gene),
            "who_grade": grade,
            "who_name": who_name,
            "who_match": how,
            "collapsed_with": ";".join(f"{g}_{m}" for g, m in group[1:]),
            "n_collapsed": len(group) - 1,
        })

    # Gene-level LoF burden columns, appended after the per-variant columns. They give
    # the model a way to express "this gene is broken" when no single broken-gene variant
    # is common enough to carry evidence on its own.
    n_variant_cols = len(groups)
    burden_genes = []
    if cfg["genotype"].get("gene_burden", True):
        burden_genes = sorted(g for g, c in lof_carriers.items() if len(c) >= min_mac)
        for k, gene in enumerate(burden_genes):
            j = n_variant_cols + k
            for i in sorted(lof_carriers[gene]):
                rows_i.append(i)
                cols_i.append(j)
            records.append({
                "column": j,
                "gene": gene,
                "mutation": "LoF_burden",
                "feature": f"{gene}_LoF_burden",
                "n_carriers": len(lof_carriers[gene]),
                "n_carriers_resistant": int(y[sorted(lof_carriers[gene])].sum()),
                "n_missing": 0,
                "genome_index": None,
                "amino_acid_number": None,
                "is_indel": False,
                "in_promoter": False,
                "element_type": "BURDEN",
                "tier": cat.tiers.get(drug, {}).get(gene),
                "who_grade": cat.lof_grades.get(drug, {}).get(gene),
                "who_name": f"{gene}_LoF",
                "who_match": "burden",
                "collapsed_with": "",
                "n_collapsed": 0,
            })

    X = sparse.csr_matrix(
        (np.ones(len(rows_i)), (rows_i, cols_i)),
        shape=(n, n_variant_cols + len(burden_genes)), dtype=np.float64
    )
    if not records:
        print(f"[{drug}] no variant cleared MAC >= {min_mac}; skipping")
        return {"drug": drug, "n": n, "cases": int(y.sum()), "genes": len(genes),
                "features_before_mac": len(carriers), "features": 0,
                "collapsed_pairs": 0, "missing_calls": n_missing_calls,
                "graded": 0, "grade_1_2": 0, "note": "no variant cleared MAC"}
    features = pl.DataFrame(records)

    out = PROC / drug
    out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out / "X.npz", X)
    features.write_csv(out / "features.csv")
    pl.DataFrame({
        "uniqueid": isolates,
        "y": y,
        "quality": [reuse[u][f"{drug}_PHENOTYPE_QUALITY"] for u in isolates],
        "mic": [reuse[u][f"{drug}_MIC"] for u in isolates],
        "site": [u.split(".")[1] for u in isolates],
        "plate": [plate.get(u, "NA") for u in isolates],
        "lineage": [lineage.get(u, "Unknown") for u in isolates],
    }).write_csv(out / "y.csv")
    features.filter(pl.col("n_collapsed") > 0).select(
        "feature", "n_carriers", "who_grade", "collapsed_with"
    ).write_csv(out / "collapsed.csv")

    return {
        "drug": drug,
        "n": n,
        "cases": int(y.sum()),
        "genes": len(genes),
        "features_before_mac": len(carriers),
        "features": n_variant_cols + len(burden_genes),
        "burden_cols": len(burden_genes),
        "collapsed_pairs": sum(len(g) - 1 for g in groups),
        "missing_calls": n_missing_calls,
        "graded": int(features["who_grade"].is_not_null().sum()),
        "grade_1_2": int(features.filter(pl.col("who_grade") <= 2).height),
        "note": "",
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cat = load_catalogue(MASTER)
    coords = load_coordinates(COORDS)
    muts = pl.read_parquet(MUTS)
    print(f"mutation rows {muts.height:,}  isolates {muts['UNIQUEID'].n_unique():,}")

    reuse = reuse_table()
    plate, lineage = plate_and_lineage(set(reuse))
    print(f"reuse isolates {len(reuse):,}  plate {len(plate):,}  lineage {len(lineage):,}")

    summary = [
        build_drug(d, cfg, cat, coords, muts, reuse, plate, lineage)
        for d in cfg["drugs"]
    ]
    frame = pl.DataFrame(summary)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    print(frame)
    frame.write_csv(PROC / "design_summary.csv")


if __name__ == "__main__":
    main()
