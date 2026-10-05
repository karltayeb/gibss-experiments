"""The CRyPTIC <-> WHO name bridge is the one place a silent error is most costly.

A failed join does not raise: it drops a grade, the variant is reported as ungraded, and a
known group 1 mutation quietly looks like a novel discovery. Every case below is a shape
that got the mapping wrong at some point during development.

    uv run pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from who import cryptic_key, load_catalogue, load_coordinates, who_key  # noqa: E402

MASTER = ROOT / "data/raw/who_catalogue_2023/Final Result Files/WHO-UCN-TB-2023.6-eng_catalogue_master_file.txt"
COORDS = ROOT / "data/raw/who_catalogue_2023/Final Result Files/WHO-UCN-TB-2023.7-eng_genomic_coordinates.txt"

needs_data = pytest.mark.skipif(not MASTER.exists(), reason="catalogue not downloaded")


def ck(gene, mutation, **kw):
    base = dict(in_promoter=False, element_type="GENE", aa_number=None, indel_length=None)
    base.update(kw)
    return cryptic_key(gene, mutation, **base)


# (WHO name, CRyPTIC gene, CRyPTIC mutation, extra CRyPTIC row fields)
SAME_VARIANT = [
    ("rpoB_p.Ser450Leu", "rpoB", "S450L", {}),
    ("katG_p.Ser315Thr", "katG", "S315T", {}),
    ("inhA_p.Ser94Ala", "inhA", "S94A", {}),
    ("embB_p.Met306Val", "embB", "M306V", {}),
    ("gyrA_p.Asp94Gly", "gyrA", "D94G", {}),
    ("rplC_p.Cys154Arg", "rplC", "C154R", {}),
    # a premature stop: CRyPTIC writes `!`, WHO writes `*`
    ("Rv2752c_p.Gln247*", "Rv2752c", "Q247!", {}),
    # non-coding RNA, numbered in n.
    ("rrl_n.2270G>C", "rrl", "g2270c", {"element_type": "RNA"}),
    # a promoter SNV, numbered negatively in c.
    ("ahpC_c.-48G>A", "ahpC", "g-48a", {"in_promoter": True}),
    # a frameshift: WHO names the reference residue, CRyPTIC gives only a codon
    ("Rv0678_p.Asp47fs", "Rv0678", "139_indel", {"aa_number": 47, "indel_length": 1}),
]


@pytest.mark.parametrize("who_name,gene,mutation,extra", SAME_VARIANT)
def test_both_sources_reach_one_key(who_name, gene, mutation, extra):
    assert who_key(who_name) == ck(gene, mutation, **extra) is not None


@pytest.mark.parametrize("mutation", ["M1T", "M1I", "M1V", "M1L"])
def test_start_lost_collapses_to_who_met1(mutation):
    """WHO files every start-codon substitution as one `p.Met1?`; CRyPTIC names each."""
    assert ck("katG", mutation) == who_key("katG_p.Met1?")


def test_codon_one_without_met_reference_is_ordinary():
    """The Met1 rule must not swallow an ordinary substitution at codon 1."""
    assert ck("rpoB", "A1V") == ("aa", "rpoB", 1, "V")


@pytest.mark.parametrize("gene,mutation,extra", [
    ("rpoB", "c-61x", {"in_promoter": True}),   # uncalled base
    ("rpoB", "A1075O", {}),                      # filter-failed codon
    ("rrl", "c344o", {"element_type": "RNA"}),   # filter-failed RNA base
])
def test_no_call_is_not_a_variant(gene, mutation, extra):
    """An uncalled or filter-failed site must not become a feature: it is missing data,
    and treating it as an observed variant would invent carriers."""
    assert ck(gene, mutation, **extra) is None


def test_inframe_indel_separates_from_frameshift():
    """A 3-base indel keeps the reading frame, so it is not a frameshift."""
    assert ck("rpoB", "1302_indel", aa_number=434, indel_length=-3)[0] == "indel"
    assert ck("rpoB", "1302_indel", aa_number=434, indel_length=-1)[0] == "fs"


@needs_data
def test_grades_resolve_for_canonical_mutations():
    cat = load_catalogue(MASTER)
    assert cat.grade("RIF", who_key("rpoB_p.Ser450Leu")) == 1
    assert cat.grade("INH", who_key("katG_p.Ser315Thr")) == 1
    assert cat.grade("INH", who_key("katG_p.Met1?")) == 1
    assert cat.grade("BDQ", who_key("Rv0678_LoF")) == 1
    assert cat.grade("LZD", who_key("rplC_p.Cys154Arg")) == 1


@needs_data
def test_who_lists_no_fabg1_tier_for_isoniazid():
    """The reason the join must go through genomic coordinates.

    CRyPTIC books the main isoniazid promoter variant under fabG1; WHO books the same base
    under inhA, measuring the shared fabG1-inhA operon promoter from the inhA start codon.
    If this ever stops being true the coordinate route can be revisited.
    """
    cat = load_catalogue(MASTER)
    assert "fabG1" not in cat.tiers["INH"]


@needs_data
def test_promoter_variant_joins_by_coordinate_only():
    cat = load_catalogue(MASTER)
    coords = load_coordinates(COORDS)
    # CRyPTIC: fabG1 c-15t at NC_000962.3 position 1673425, C>T
    names = coords[(1673425, "C", "T")]
    assert "inhA_c.-777C>T" in names
    assert any(cat.grade_by_name("INH", n) == 1 for n in names)
    # and the name route genuinely cannot reach it
    assert cat.grade("INH", ck("fabG1", "c-15t", in_promoter=True)) is None
