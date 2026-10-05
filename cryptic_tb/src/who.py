"""WHO 2023 mutation catalogue: tiers, grades, and the name bridge to CRyPTIC.

The catalogue and the CRyPTIC MUTATIONS table name the same variant differently
(`rpoB_p.Ser450Leu` vs `rpoB` + `S450L`), so neither name is a join key. Both are
parsed here into one canonical key:

    ("aa",    gene, codon, alt_aa1)      missense / nonsense / start-loss
    ("fs",    gene, codon)               frameshift
    ("nt",    gene, pos, ref, alt)       promoter (c.-n) and non-coding RNA (n.) SNVs
    ("cds",   gene, pos, ref, alt)       synonymous / in-frame coding SNVs named in c.
    ("lof",   gene)                      gene-level loss of function
    ("indel", gene, codon, kind, length) in-frame insertion / deletion

A frameshift key drops the reference amino acid because CRyPTIC does not record it
(`418_indel`, not `p.Asp140fs`); the codon pins the variant on its own.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*",
}

# Grade strings in the master file look like "1) Assoc w R"; keep the leading digit.
GRADE_RE = re.compile(r"^(\d)\)")

# WHO variant-name shapes, tried in order.
_W_MISSENSE = re.compile(r"^(?P<gene>.+?)_p\.(?P<ref>[A-Z][a-z]{2})(?P<codon>\d+)(?P<alt>[A-Z][a-z]{2}|\*)$")
_W_START_LOST = re.compile(r"^(?P<gene>.+?)_p\.Met1\?$")
_W_FS = re.compile(r"^(?P<gene>.+?)_p\.(?P<ref>[A-Z][a-z]{2})(?P<codon>\d+)fs$")
_W_LOF = re.compile(r"^(?P<gene>.+?)_(LoF|deletion)$")
_W_NT = re.compile(r"^(?P<gene>.+?)_(?P<kind>[cn])\.(?P<pos>-?\d+)(?P<ref>[ACGT]+)>(?P<alt>[ACGT]+)$")
_W_DEL = re.compile(r"^(?P<gene>.+?)_p\.(?P<ref>[A-Z][a-z]{2})(?P<codon>\d+)(_[A-Za-z0-9]+)?del(ins[A-Za-z]+)?$")
_W_DUP = re.compile(r"^(?P<gene>.+?)_p\.(?P<ref>[A-Z][a-z]{2})(?P<codon>\d+)dup$")
_W_INS = re.compile(r"^(?P<gene>.+?)_p\.(?P<ref>[A-Z][a-z]{2})(?P<codon>\d+)_[A-Za-z]{3}\d+ins(?P<ins>[A-Za-z]+)$")
_W_STOPEXT = re.compile(r"^(?P<gene>.+?)_p\.Ter(?P<codon>\d+)[A-Za-z]+ext")


def who_key(variant: str):
    """Canonical key for a WHO variant name, or None when it has no CRyPTIC analogue."""
    m = _W_START_LOST.match(variant)
    if m:
        return ("aa", m["gene"], 1, "?")
    m = _W_MISSENSE.match(variant)
    if m:
        alt = "*" if m["alt"] == "*" else AA3_TO_1.get(m["alt"])
        if alt is None:
            return None
        return ("aa", m["gene"], int(m["codon"]), alt)
    m = _W_FS.match(variant)
    if m:
        return ("fs", m["gene"], int(m["codon"]))
    m = _W_LOF.match(variant)
    if m:
        return ("lof", m["gene"])
    m = _W_NT.match(variant)
    if m:
        # c. on a coding gene is CDS numbering; c. with a negative position is the
        # promoter; n. is a non-coding transcript (rrs/rrl).
        pos = int(m["pos"])
        tag = "nt" if (m["kind"] == "n" or pos < 0) else "cds"
        return (tag, m["gene"], pos, m["ref"].upper(), m["alt"].upper())
    m = _W_INS.match(variant)
    if m:
        return ("indel", m["gene"], int(m["codon"]), "ins", 3 * len(m["ins"]) // 3)
    m = _W_DUP.match(variant)
    if m:
        return ("indel", m["gene"], int(m["codon"]), "ins", 1)
    m = _W_DEL.match(variant)
    if m:
        return ("indel", m["gene"], int(m["codon"]), "del", None)
    m = _W_STOPEXT.match(variant)
    if m:
        return ("aa", m["gene"], int(m["codon"]), "ext")
    return None


# CRyPTIC MUTATIONS-table name shapes.
_C_AA = re.compile(r"^(?P<ref>[A-Z!])(?P<codon>\d+)(?P<alt>[A-Z!])$")
_C_NT = re.compile(r"^(?P<ref>[acgt])(?P<pos>-?\d+)(?P<alt>[acgtxo])$")
_C_INDEL = re.compile(r"^(?P<pos>-?\d+)_indel$")


def cryptic_key(gene: str, mutation: str, *, in_promoter: bool, element_type: str,
                aa_number, indel_length, codon_number=None):
    """Canonical key for a CRyPTIC (GENE, MUTATION) pair, or None if unmappable.

    `aa_number` is AMINO_ACID_NUMBER, `codon_number` the fallback codon for an indel.
    Null (`x`) and filter-fail (`o`) alt alleles yield None: they are not a variant.
    """
    m = _C_AA.match(mutation)
    if m:
        alt = "*" if m["alt"] == "!" else m["alt"]
        if alt in ("O", "X"):  # null / filter-fail amino acid call
            return None
        codon = int(m["codon"])
        # WHO collapses every start-codon substitution into one `p.Met1?` entry, while
        # CRyPTIC names the specific replacement (`M1T`). Without this the start-lost
        # variants never join, and `katG_p.Met1?` is a group 1 isoniazid call.
        if codon == 1 and m["ref"] == "M":
            return ("aa", gene, 1, "?")
        return ("aa", gene, codon, alt)
    m = _C_NT.match(mutation)
    if m:
        if m["alt"] in ("x", "o"):
            return None
        pos = int(m["pos"])
        # A promoter SNV or an RNA gene is WHO's c.-n / n. space; a coding-sequence
        # nucleotide call is c. CDS numbering.
        tag = "nt" if (in_promoter or element_type == "RNA") else "cds"
        return (tag, gene, pos, m["ref"].upper(), m["alt"].upper())
    m = _C_INDEL.match(mutation)
    if m:
        if indel_length is None:
            return None
        length = int(indel_length)
        codon = aa_number if aa_number is not None else codon_number
        if codon is None:
            return None
        if length % 3 != 0:
            return ("fs", gene, int(codon))
        kind = "ins" if length > 0 else "del"
        return ("indel", gene, int(codon), kind, abs(length) // 3)
    return None


@dataclass(frozen=True)
class Catalogue:
    """Parsed WHO 2023 catalogue for one drug set."""

    # (drug, canonical key) -> grade digit 1..5
    grades: dict[tuple[str, tuple], int]
    # (drug, canonical key) -> the WHO variant name, for reporting
    names: dict[tuple[str, tuple], str]
    # drug -> {gene: tier}
    tiers: dict[str, dict[str, int]]
    # drug -> {gene-level LoF grade}, used for frameshift fallback
    lof_grades: dict[str, dict[str, int]]
    n_rows: int
    n_unmapped: int
    # WHO variant name -> {drug: grade}; the target of the coordinate join
    by_name: dict[str, dict[str, int]]

    def grade(self, drug: str, key) -> int | None:
        return self.grades.get((drug, key))

    def name(self, drug: str, key) -> str | None:
        return self.names.get((drug, key))

    def grade_by_name(self, drug: str, variant: str) -> int | None:
        return self.by_name.get(variant, {}).get(drug)


# Three-letter CRyPTIC drug codes to the catalogue's full drug names.
DRUG_NAMES = {
    "AMI": "Amikacin", "BDQ": "Bedaquiline", "CFZ": "Clofazimine",
    "DLM": "Delamanid", "EMB": "Ethambutol", "ETH": "Ethionamide",
    "INH": "Isoniazid", "KAN": "Kanamycin", "LEV": "Levofloxacin",
    "LZD": "Linezolid", "MXF": "Moxifloxacin", "RIF": "Rifampicin",
    "PZA": "Pyrazinamide", "STM": "Streptomycin", "CAP": "Capreomycin",
}
NAME_TO_DRUG = {v: k for k, v in DRUG_NAMES.items()}


def load_coordinates(coord_file: str | Path) -> dict[tuple[int, str, str], set[str]]:
    """(position, ref, alt) -> the WHO variant names that spelling realises.

    This is the nomenclature-free half of the bridge, and it is the only way to reach
    variants the two sources attribute to different genes. CRyPTIC calls the isoniazid
    promoter variant `fabG1` `c-15t`; WHO calls the same base `inhA_c.-777C>T`, because
    it measures the fabG1-inhA operon promoter from the inhA start codon. The canonical
    key misses it, the genomic coordinate 1673425 C>T does not.
    """
    out: dict[tuple[int, str, str], set[str]] = {}
    with open(coord_file, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (
                int(row["position"]),
                row["reference_nucleotide"].upper(),
                row["alternative_nucleotide"].upper(),
            )
            out.setdefault(key, set()).add(row["variant"])
    return out


def load_catalogue(master_file: str | Path) -> Catalogue:
    """Parse the catalogue master file into canonical keys, tiers and grades."""
    grades: dict[tuple[str, tuple], int] = {}
    names: dict[tuple[str, tuple], str] = {}
    tiers: dict[str, dict[str, int]] = {}
    lof: dict[str, dict[str, int]] = {}
    by_name: dict[str, dict[str, int]] = {}
    n_rows = 0
    n_unmapped = 0

    with open(master_file, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            drug_name = row["drug"]
            drug = NAME_TO_DRUG.get(drug_name)
            if drug is None:
                continue
            n_rows += 1
            gene = row["gene"]
            try:
                tier = int(row["tier"])
            except (TypeError, ValueError):
                tier = 0
            tiers.setdefault(drug, {})
            # A gene appears under one tier per drug; keep the lowest seen.
            if gene not in tiers[drug] or tier < tiers[drug][gene]:
                tiers[drug][gene] = tier

            gm = GRADE_RE.match(row["FINAL CONFIDENCE GRADING"] or "")
            if gm is None:
                continue
            grade = int(gm.group(1))
            variant = row["variant"]
            prev_n = by_name.setdefault(variant, {}).get(drug)
            if prev_n is None or grade < prev_n:
                by_name[variant][drug] = grade
            key = who_key(variant)
            if key is None:
                n_unmapped += 1
                continue
            if key[0] == "lof":
                lof.setdefault(drug, {})[key[1]] = grade
            # The master file can list a key twice (same variant, two rows); the
            # stronger grade (lower digit) wins so a group-1 call is never masked.
            prev = grades.get((drug, key))
            if prev is None or grade < prev:
                grades[(drug, key)] = grade
                names[(drug, key)] = variant

    return Catalogue(grades, names, tiers, lof, n_rows, n_unmapped, by_name)
