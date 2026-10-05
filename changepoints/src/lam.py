"""Lambda phage genome: sequence, GC indicator, binning, GenBank CDS table."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed/lambda"


def load_sequence() -> str:
    lines = (RAW / "NC_001416.fasta").read_text().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">")).upper()


def gc_bits(seq: str) -> np.ndarray:
    a = np.frombuffer(seq.encode(), dtype=np.uint8)
    return ((a == ord("G")) | (a == ord("C"))).astype(np.int64)


def bin_counts(bits: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin start positions 1-based, GC count k, trials m). A trailing remainder shorter
    than half a bin is merged into the last full bin, so the last bin may be longer."""
    n = len(bits)
    starts = np.arange(0, n, width)
    if len(starts) > 1 and n - starts[-1] < width / 2:
        starts = starts[:-1]
    ends = np.append(starts[1:], n)
    k = np.array([bits[s:e].sum() for s, e in zip(starts, ends)])
    m = ends - starts
    return starts + 1, k, m


def cds_table() -> pl.DataFrame:
    """CDS features from the GenBank record: start, end (1-based, inclusive), strand, gene,
    product."""
    text = (RAW / "NC_001416.gb").read_text()
    feats = text.split("FEATURES")[1].split("ORIGIN")[0]
    rows = []
    cur = None
    for line in feats.splitlines():
        m = re.match(r"^ {5}(\S+) +(.+)$", line)
        if m:
            if cur and cur["type"] == "CDS":
                rows.append(cur)
            ftype, loc = m.groups()
            strand = "-" if loc.startswith("complement") else "+"
            nums = [int(x) for x in re.findall(r"\d+", loc)]
            cur = {"type": ftype, "start": min(nums), "end": max(nums), "strand": strand,
                   "gene": "", "product": "", "note": ""}
            continue
        q = re.match(r'^ {21}/(\w+)="?([^"]*)"?', line)
        if q and cur is not None:
            key, val = q.groups()
            if key in ("gene", "product", "note") and not cur[key]:
                cur[key] = val
    if cur and cur["type"] == "CDS":
        rows.append(cur)
    return pl.DataFrame(rows).drop("type").sort("start")
