#!/usr/bin/env python3
"""Create symlink views of results/supercollections/{sc}/{plot_type}/{settings}.pdf.

Two views under results/plots/:
  by_type/{plot_type}/{settings}/{supercollection}.pdf   -- compare across supercollections
  by_sc/{supercollection}/{settings}/{plot_type}.pdf     -- browse one supercollection

Usage:
    uv run python scripts/symlink_plots.py
    uv run python scripts/symlink_plots.py --results /path/to/results
"""
import argparse
import os
from pathlib import Path


def _link(src: Path, dst: Path) -> bool:
    rel_target = os.path.relpath(src, dst.parent)
    if dst.is_symlink():
        if os.readlink(dst) == rel_target:
            return False
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(rel_target)
    return True


def symlink_plots(results: Path) -> None:
    sc_root = results / "supercollections"
    by_type = results / "plots" / "by_type"
    by_sc = results / "plots" / "by_sc"

    count = 0
    for pdf in sorted(sc_root.glob("*/*/*.pdf")):
        sc = pdf.parts[-3]
        plot_type = pdf.parts[-2]
        settings = pdf.stem

        count += _link(pdf, by_type / plot_type / settings / f"{sc}.pdf")
        count += _link(pdf, by_sc / sc / settings / f"{plot_type}.pdf")

    print(f"Created/updated {count} symlinks under {results / 'plots'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", type=Path)
    args = parser.parse_args()
    symlink_plots(args.results)
