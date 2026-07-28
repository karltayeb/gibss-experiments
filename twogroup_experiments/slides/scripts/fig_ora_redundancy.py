"""Marginal ORA is redundant and uninterpretable - a fresh simulated example.

Mirrors the real covid GO:BP ORA (gsea_examples/covid/eval/covid_volcano): a
~4,900-set GO:BP collection, ORA (Fisher + BH), plotted as -log10 FDR vs the
log odds ratio of set enrichment in the hit list, ~11% of sets significant.

We simulate gene-level z-scores under a covariate-moderated two-group model:
10 GO:BP sets (2 large / 3 medium / 5 small) are truly *causal* - their member
genes have an elevated probability of being active (non-null). Active genes draw
z ~ N(mu1, 1); everything else draws z ~ N(0, 1). Thresholding gives a hit list;
ORA runs against every set in the collection. A modest number of independent
signals (not one dominant set) reproduces covid's diffuse cloud that tapers
naturally instead of piling up at the top.

Two reveal stages (same axes, so the slide can overlay them):
  stage "marginal" - the volcano with every FDR<0.05 set in red. Hundreds cross
                     the line; overlap with the causal sets drags a redundant
                     cloud along. Which ones matter is unreadable.
  stage "causal"   - the *same* volcano, cloud muted, the 10 causal sets marked
                     (marker = size class). Recovering *those* is the
                     interpretable analysis - the motivation for a joint SuSiE
                     model over the marginal tests.

Data: resources/gobp_collection.gmt (committed; built by gobp_prep.py from the
MSigDB C5 GO:BP sets). Simulation is seeded, so the figure is reproducible.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import hypergeom, false_discovery_control
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _common import save, INK, MUT, NONNULL_C

_HERE = pathlib.Path(__file__).resolve().parent
_SLIDES = _HERE.parent
GMT = _SLIDES / "resources/gobp_collection.gmt"
META = _SLIDES / "resources/gobp_collection.meta.json"

PARAMS = {
    "seed": 20260730,
    "pi0": 0.03,          # background P(active) for a gene in no causal set
    # P(active) for a gene in a causal set, per size class. Large sets need less
    # per-gene signal to be significant, so a lower pi keeps them from dominating
    # the top of the volcano; small sets get more so they clear the line.
    "pi_causal": {"large": 0.32, "medium": 0.48, "small": 0.65},
    "mu1": 2.6,           # mean z of an active gene (null genes ~ N(0,1))
    "tau": 2.0,           # hit threshold on z (one-sided)
    "fdr": 0.05,
}

# Distinct highlight markers per causal size class.
CAUSAL_STYLE = {
    "large":  dict(marker="*", s=300),
    "medium": dict(marker="D", s=120),
    "small":  dict(marker="^", s=120),
}
YMAX = 15.0   # -log10 FDR display ceiling (covid tops out ~14); outliers clipped
XLIM = (-2.6, 3.6)


def load_collection():
    sets = {}
    for line in open(GMT):
        parts = line.rstrip("\n").split("\t")
        sets[parts[0]] = [g for g in parts[2:] if g]
    meta = json.loads(META.read_text())
    return sets, meta


def simulate_and_ora():
    p = PARAMS
    rng = np.random.default_rng(p["seed"])
    sets, meta = load_collection()
    # meta["causal"]: band -> [{name, n_genes}, ...]
    causal_by_band = {band: [i["name"] for i in infos]
                      for band, infos in meta["causal"].items()}
    causal_names = {n for names in causal_by_band.values() for n in names}

    universe = sorted({g for gs in sets.values() for g in gs}, key=int)
    idx = {g: i for i, g in enumerate(universe)}
    N = len(universe)

    # covariate-moderated active probability: p0 background, raised to the size
    # class's pi inside a causal set (max over classes if a gene is in several).
    p_active = np.full(N, p["pi0"])
    for band, names_b in causal_by_band.items():
        pic = p["pi_causal"][band]
        for name in names_b:
            for g in sets[name]:
                p_active[idx[g]] = max(p_active[idx[g]], pic)
    active = rng.random(N) < p_active

    # gene-level z-scores, then a hit list
    z = rng.normal(np.where(active, p["mu1"], 0.0), 1.0)
    hit = z > p["tau"]
    n_hit = int(hit.sum())

    # ORA: Fisher exact (hypergeometric) per set + Haldane-corrected odds ratio
    names = list(sets.keys())
    logor = np.empty(len(names))
    pval = np.empty(len(names))
    set_size = np.empty(len(names), dtype=int)
    for j, name in enumerate(names):
        members = np.fromiter((idx[g] for g in sets[name]), dtype=int, count=len(sets[name]))
        m = len(members)
        k = int(hit[members].sum())                 # hits in set
        set_size[j] = m
        a, b = k, m - k
        c, d = n_hit - k, (N - m) - (n_hit - k)
        pval[j] = hypergeom.sf(k - 1, N, m, n_hit)   # one-sided enrichment
        # Haldane-Anscombe 0.5 correction so log OR is finite at the extremes
        logor[j] = np.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))

    qval = false_discovery_control(pval, method="bh")
    sig = qval < p["fdr"]

    # causal set indices, grouped by size class
    causal_idx = {band: [names.index(n) for n in names_b]
                  for band, names_b in causal_by_band.items()}
    return dict(
        names=names, logor=logor, qval=qval, sig=sig, set_size=set_size,
        causal_idx=causal_idx, meta=meta, n_hit=n_hit, n_universe=N,
    )


def _new_ax():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlabel("log odds ratio (set enrichment in hit list)")
    ax.set_ylabel(r"$-\log_{10}$ FDR")
    ax.axvline(0.0, color=MUT, lw=0.8, ls=":", zorder=0)
    ax.axhline(-np.log10(PARAMS["fdr"]), color=INK, lw=1.0, ls="--", zorder=1)
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.4, YMAX)
    return fig, ax


def _nlq(qval):
    return np.clip(-np.log10(np.clip(qval, 1e-300, None)), None, YMAX - 0.25)


def render(out, stage):
    r = simulate_and_ora()
    nlq, sig = _nlq(r["qval"]), r["sig"]
    logor = r["logor"]
    n_sig, n_tot = int(sig.sum()), len(r["names"])
    frac = 100 * n_sig / n_tot

    fig, ax = _new_ax()
    if stage == "marginal":
        ax.scatter(logor[~sig], nlq[~sig], s=9, c=MUT, alpha=0.45, lw=0, rasterized=True)
        ax.scatter(logor[sig], nlq[sig], s=11, c=NONNULL_C, alpha=0.65, lw=0,
                   rasterized=True, label=f"FDR < {PARAMS['fdr']:g}  ({n_sig} / {n_tot} sets)")
        ax.set_title(f"Marginal GO:BP ORA: {n_sig} / {n_tot:,} sets at FDR<0.05 ({frac:.0f}%)",
                     fontsize=10.5)
        ax.legend(loc="upper left", frameon=False, fontsize=9, markerscale=1.4,
                  handletextpad=0.3)
    elif stage == "causal":
        # muted cloud: the same points, faded; causal sets highlighted on top
        ax.scatter(logor[~sig], nlq[~sig], s=8, c=MUT, alpha=0.13, lw=0, rasterized=True)
        ax.scatter(logor[sig], nlq[sig], s=9, c=NONNULL_C, alpha=0.16, lw=0, rasterized=True)
        n_causal = sum(len(v) for v in r["causal_idx"].values())
        handles = []
        for band, idxs in r["causal_idx"].items():
            st = CAUSAL_STYLE[band]
            jj = np.array(idxs)
            ax.scatter(logor[jj], nlq[jj], marker=st["marker"], s=st["s"], c=NONNULL_C,
                       edgecolors=INK, linewidths=0.9, zorder=5)
            handles.append(Line2D([], [], marker=st["marker"], color="none", lw=0,
                                  markerfacecolor=NONNULL_C, markeredgecolor=INK,
                                  markersize=11, label=f"{band}  ({len(idxs)} sets)"))
        ax.set_title(f"Underneath the cloud: {n_causal} causal sets  "
                     f"({n_sig:,} / {n_tot:,} marginally sig.)", fontsize=10.5)
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
                  handletextpad=0.4, labelspacing=0.6, title="causal (truth)",
                  title_fontsize=9)
    else:
        raise ValueError(stage)

    fig.tight_layout()
    params = {**PARAMS, "stage": stage, "n_sig": n_sig, "n_sets": n_tot,
              "n_hit": r["n_hit"], "n_universe": r["n_universe"]}
    save(fig, out, params, __file__)


def main(out_marginal, out_causal, diag=False):
    if diag:
        r = simulate_and_ora()
        nlq = _nlq(r["qval"])
        print(f"universe={r['n_universe']}  hits={r['n_hit']}  sets={len(r['names'])}  "
              f"sig={int(r['sig'].sum())} ({100*r['sig'].mean():.1f}%)  "
              f"raw max -log10FDR={-np.log10(np.clip(r['qval'].min(),1e-300,None)):.0f}")
        n_ceil = int((nlq >= YMAX - 0.3).sum())
        print(f"  points at ceiling (>= {YMAX-0.3:.1f}): {n_ceil}")
        for band, idxs in r["causal_idx"].items():
            for j in idxs:
                print(f"  causal[{band:6}] size={r['set_size'][j]:4d}  "
                      f"logOR={r['logor'][j]:+.2f}  -log10FDR(clip)={nlq[j]:.1f}  q={r['qval'][j]:.1e}")
        return
    render(out_marginal, "marginal")
    render(out_causal, "causal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-marginal", default=str(_SLIDES / "figures/fig_ora_redundancy_marginal.pdf"))
    ap.add_argument("--out-causal", default=str(_SLIDES / "figures/fig_ora_redundancy_causal.pdf"))
    ap.add_argument("--diag", action="store_true", help="print sim diagnostics, no plot")
    a = ap.parse_args()
    main(a.out_marginal, a.out_causal, diag=a.diag)
