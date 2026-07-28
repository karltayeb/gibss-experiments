"""Build a committed GO:BP gene-set collection for the ORA-redundancy figure.

Reads the MSigDB C5 GMT (GO gene sets, Entrez ids), keeps the GO:BP subset,
filters to a size band (~4,900 sets, matching the real covid GO:BP ORA scale of
4,815 sets), and records 10 chosen "causal" sets (2 large / 3 medium / 5 small).
Writes:

    resources/gobp_collection.gmt        - the full-band collection (GMT text)
    resources/gobp_collection.meta.json  - provenance + the causal set names

The default GMT lives inside the covid example's venv; it is only needed to
*regenerate* the committed collection, never to build the figure. Run once:

    uv run python scripts/gobp_prep.py

The figure script (fig_ora_redundancy.py) reads only the committed GMT, so the
figure is fully reproducible from the repo without any external path.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

_HERE = pathlib.Path(__file__).resolve().parent
_SLIDES = _HERE.parent
_DEFAULT_GMT = (
    _SLIDES.parent.parent  # gibss-experiments root
    / "gsea_examples/covid/.venv/lib/python3.13/site-packages/gseasusie"
    / "msigdb/c5.all.v2026.1.Hs.entrez.gmt"
)

# Causal size classes: (n genes lo, hi, how many sets to draw from this class).
# 10 causal sets total (2 large / 3 medium / 5 small) - enough independent
# signals to give a covid-like diffuse cloud with a natural taper.
CAUSAL_BANDS = {
    "large":  (350, 700, 2),
    "medium": (70, 160, 3),
    "small":  (22, 45, 5),
}
# Collection membership: all GO:BP sets in this size band (~4,900 sets, matching
# the real covid GO:BP ORA collection, 4,815 sets). No subsampling - the full
# collection is what makes the marginal redundancy realistic.
SET_SIZE_BAND = (12, 800)
SEED = 20260730


def read_gobp(gmt_path: pathlib.Path) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with open(gmt_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            name = parts[0]
            if not name.startswith("GOBP_"):
                continue
            genes = [g for g in parts[2:] if g]
            sets[name] = genes
    return sets


def pick_causal(sets, rng) -> dict[str, list[str]]:
    """Draw `count` random sets per size class (seeded), varied in size."""
    chosen: dict[str, list[str]] = {}
    used: set[str] = set()
    for band, (lo, hi, count) in CAUSAL_BANDS.items():
        cands = [n for n, g in sets.items() if lo <= len(g) <= hi and n not in used]
        rng.shuffle(cands)
        picks = cands[:count]
        chosen[band] = picks
        used.update(picks)
    return chosen


def main(gmt_path, out_gmt, out_meta):
    rng = random.Random(SEED)
    all_sets = read_gobp(pathlib.Path(gmt_path))

    causal = pick_causal(all_sets, rng)   # band -> [names]
    causal_names = {n for names in causal.values() for n in names}

    collection = sorted(
        n for n, g in all_sets.items()
        if SET_SIZE_BAND[0] <= len(g) <= SET_SIZE_BAND[1]
    )
    # causal sets are chosen from inside the band, so already present
    assert causal_names <= set(collection)

    out_gmt = pathlib.Path(out_gmt)
    out_gmt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_gmt, "w") as fh:
        for name in collection:
            genes = all_sets[name]
            fh.write("\t".join([name, "na", *genes]) + "\n")

    universe = sorted({g for n in collection for g in all_sets[n]}, key=int)
    meta = {
        "source_gmt": str(gmt_path),
        "seed": SEED,
        "set_size_band": SET_SIZE_BAND,
        "n_sets_total_gobp": len(all_sets),
        "n_sets_collection": len(collection),
        "n_genes_universe": len(universe),
        "n_causal": len(causal_names),
        "causal": {
            band: [{"name": n, "n_genes": len(all_sets[n])} for n in names]
            for band, names in causal.items()
        },
    }
    pathlib.Path(out_meta).write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {out_gmt}  ({len(collection)} sets, {len(universe)} genes, "
          f"{len(causal_names)} causal)")
    for band, infos in meta["causal"].items():
        for info in infos:
            print(f"  causal[{band:6}] {info['name']}  (n={info['n_genes']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmt", default=str(_DEFAULT_GMT))
    ap.add_argument("--out-gmt", default=str(_SLIDES / "resources/gobp_collection.gmt"))
    ap.add_argument("--out-meta", default=str(_SLIDES / "resources/gobp_collection.meta.json"))
    a = ap.parse_args()
    main(a.gmt, a.out_gmt, a.out_meta)
