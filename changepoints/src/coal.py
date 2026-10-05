"""Coal-mining disaster data: load, cross-check, bin."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed/coal"

# Yearly counts 1851-1962 in the form circulated from Carlin, Gelfand & Smith (1992) and
# used by most single-changepoint analyses. TRANSCRIBED FROM MEMORY, not from the paper:
# it is a plausibility check on the boot::coal binning, not an authority. It sums to 190.
CARLIN_1992_TRANSCRIBED = np.array([
    4, 5, 4, 1, 0, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5,
    3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 1, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1,
    0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1,
    2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
])


def load_dates(which: str = "boot") -> np.ndarray:
    path = RAW / f"coal_{which}.csv"
    first = path.read_text().splitlines()[0].strip()
    header = not first[0].isdigit()  # boot copy has a "date" header, the pymc copy none
    return np.loadtxt(path, skiprows=int(header), dtype=float)


def yearly_counts(dates: np.ndarray, year_start: int, year_end: int) -> tuple[np.ndarray, np.ndarray]:
    years = np.arange(year_start, year_end + 1)
    yr = np.floor(dates).astype(int)
    counts = np.array([(yr == y).sum() for y in years])
    return years, counts


def monthly_counts(dates: np.ndarray, year_start: int, year_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts per calendar month; the month label is the decimal year at the month start."""
    yr = np.floor(dates).astype(int)
    mo = np.minimum(np.floor((dates - yr) * 12).astype(int), 11)
    idx = (yr - year_start) * 12 + mo
    n = (year_end - year_start + 1) * 12
    counts = np.bincount(idx, minlength=n)[:n]
    labels = year_start + np.arange(n) / 12.0
    return labels, counts
