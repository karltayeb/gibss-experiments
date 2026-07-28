"""Export the landscape-notebook figures to named files (vector PDF by default) for slides.

Regenerates every figure straight from the reductions (no quarto render needed), writing
one file per (metric, scenario, m) with a descriptive name:
  power_fdp__loc_m200.pdf   coverage_size__scale_m50.pdf   calibration__loc_m100.pdf
  detection_roc__loc_m400.pdf   landscape_cssize__scale.pdf

Run:  uv run --project .. python scripts/export_figures.py 011-gobp-operating-points \
        --out analysis/figures --format pdf [--tables]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "analysis"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import op_landscape as ol  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export landscape-notebook figures for slides.")
    ap.add_argument("supercollection", nargs="?", default="011-gobp-operating-points",
                    help="supercollection name (default: 011-gobp-operating-points)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: analysis/figures/<supercollection>)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                    help="figure format (default: pdf, vector)")
    ap.add_argument("--dpi", type=int, default=200, help="raster DPI for png (default 200)")
    ap.add_argument("--tables", action="store_true",
                    help="also dump the coverage-size table as CSV per (scenario, m)")
    args = ap.parse_args(argv)

    outdir = args.out or str(_HERE / "analysis" / "figures" / args.supercollection)
    print(f"exporting {args.supercollection} figures -> {outdir} ({args.format})", flush=True)
    written = ol.export_figures(args.supercollection, outdir, fmt=args.format,
                                dpi=args.dpi, tables=args.tables)
    for p in written:
        print("  " + p)
    print(f"\n{len(written)} files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
