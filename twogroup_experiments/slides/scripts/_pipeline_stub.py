"""Placeholder for figures whose real content comes from the results pipeline
(heavy simulations) rather than a standalone script. Used for the backup /
rerun-gated slides until the gibss-mono migration lands and the collections are
regenerated. Invoked by figures.smk with per-figure title/note.
"""
import argparse

from _common import placeholder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--source", default="results pipeline (post-migration)")
    a = ap.parse_args()
    placeholder(
        a.out, a.slide, a.title,
        f"{a.note}\n\nsource: {a.source}",
        {"slide": a.slide, "source": a.source}, __file__,
    )


if __name__ == "__main__":
    main()
