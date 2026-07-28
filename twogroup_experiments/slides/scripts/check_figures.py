"""Figure-geometry checker: keep slide figures lined up.

Every slide figure is placed by \\slidefig, which centres it in a fixed box
(BOX_W x BOX_H, in cm, matching the Beamer 16:9 body area) with keepaspectratio.
Figures line up on the slide iff they render at a consistent size in that box -
which happens when their aspect ratios sit in a sane band. This script reads each
figures/*.pdf, computes its aspect ratio and the size it would render at inside
the box, and flags anything that would look out of line.

    uv run --with pymupdf python scripts/check_figures.py
    (also run by `snakemake -s figures.smk check` / `make check`)

Exit code is nonzero if any figure violates the aspect band, so it can gate the
build.
"""
import argparse
import glob
import os
import sys

import fitz  # pymupdf

# Beamer 16:9 body box the deck reserves for a figure (cm). Matches \slidefig in
# deck.tex: width 0.70\textwidth x height 0.72\textheight (aspect ~1.76).
BOX_W_CM = 9.0
BOX_H_CM = 5.1

# Allowed aspect band (width / height). Figures at or above the box aspect render
# at the full box width -> they line up. Outliers render noticeably narrower.
MIN_ASPECT = 1.70
MAX_ASPECT = 2.75
# Rendered width must be within this fraction of the box width to count as "lined up".
MIN_FILL = 0.85

# Figures allowed to be out of band on purpose (with reason).
ALLOWLIST = {
    "figB2_cox_poisson.pdf": "4:3 by request (density comparison)",
}

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def rendered(aspect):
    """Width, height (cm) the figure renders at inside the box (keepaspectratio)."""
    box_aspect = BOX_W_CM / BOX_H_CM
    if aspect >= box_aspect:            # width-limited
        return BOX_W_CM, BOX_W_CM / aspect
    return BOX_H_CM * aspect, BOX_H_CM  # height-limited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit nonzero on any violation")
    a = ap.parse_args()

    pdfs = sorted(glob.glob(os.path.join(FIGDIR, "*.pdf")))
    rows, violations = [], []
    for p in pdfs:
        r = fitz.open(p)[0].rect
        w_in, h_in = r.width / 72.0, r.height / 72.0
        aspect = w_in / h_in
        rw, rh = rendered(aspect)
        fill = rw / BOX_W_CM
        name = os.path.basename(p)
        bad = []
        if name in ALLOWLIST:
            bad = []                      # documented exception
        else:
            if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                bad.append(f"aspect {aspect:.2f} outside [{MIN_ASPECT},{MAX_ASPECT}]")
            if fill < MIN_FILL:
                bad.append(f"fills only {fill:.0%} of box width")
        rows.append((name, w_in, h_in, aspect, rw, fill, bad))
        if bad:
            violations.append((name, bad))

    name_w = max(len(r[0]) for r in rows)
    print(f"box = {BOX_W_CM} x {BOX_H_CM} cm   aspect band [{MIN_ASPECT}, {MAX_ASPECT}]   min fill {MIN_FILL:.0%}\n")
    print(f"{'figure':<{name_w}}  {'in (WxH)':>11}  {'aspect':>6}  {'rend.w':>7}  {'fill':>5}  flags")
    for name, wi, hi, asp, rw, fill, bad in rows:
        flag = "OK" if not bad else "  <-- " + "; ".join(bad)
        print(f"{name:<{name_w}}  {wi:5.2f}x{hi:4.2f}  {asp:6.2f}  {rw:6.1f}cm  {fill:4.0%}  {flag}")

    if violations:
        print(f"\n{len(violations)} figure(s) would not line up:")
        for n, bad in violations:
            print(f"  - {n}: {'; '.join(bad)}")
        if a.strict:
            sys.exit(1)
    else:
        print("\nAll figures within the aspect band - they will line up.")


if __name__ == "__main__":
    main()
