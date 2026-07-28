"""Slide 15 - late arrivals dominate the score.  [spotlight, local]

Cox / Plackett-Luce score for a single binary covariate at the null:
    g = n1 - sum_i p_i.
Move ONE observation, currently at arrival position k, into the enriched group.
The score changes by two pieces:
  * count term:  n1 -> n1 + 1, contributing +1 (the SAME for every k);
  * position term: the observation sits in the risk set for stages i = 1..k, so
    sum_i p_i rises by  P(k) = sum_{i=1}^{k} 1/(n-i+1) = H_n - H_{n-k},
    where H_m = sum_{i=1}^{m} 1/i is the m-th harmonic number (H_0 = 0).
Net:
    Delta_g(k) = 1 - (H_n - H_{n-k}),   k = 1..n  (1-indexed; k=1 = first arrival).
  * early swap (k=1):  1 - 1/n            ~ +1        = O(1)
  * late  swap (k=n):  1 - H_n ~ 1 - log n ~ -log n   = O(log n)

So swapping an early arrival barely nudges the score (+1), while swapping the last
arrival swings it by ~ -log n: late arrivals dominate. This plot is agnostic to how
we ordered the data - it just shows influence vs arrival position.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from _common import save, AMBER, INK, MUT

PARAMS = {"n": 1000}


def main(out: str) -> None:
    n = PARAMS["n"]
    k = np.arange(1, n + 1)                         # 1-indexed arrival position
    Hm = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])  # H_0..H_n
    Hn = Hm[n]
    dg = 1.0 - (Hn - Hm[n - k])                     # Delta_g(k)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axhline(0, color=MUT, lw=0.8, ls=":")
    ax.plot(k, dg, color=AMBER, lw=2.6)

    ax.scatter([1], [dg[0]], color=INK, zorder=5, s=22)
    ax.scatter([n], [dg[-1]], color=INK, zorder=5, s=22)
    ax.annotate(r"early swap: $1-\frac{1}{n}\approx +1$", xy=(1, dg[0]),
                xytext=(90, dg[0] - 1.6), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.annotate(r"late swap: $1-H_n\approx -\log n$", xy=(n, dg[-1]),
                xytext=(n * 0.30, dg[-1] + 0.4), fontsize=9, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1))

    ax.set_xlabel(r"arrival position $k$ of the swapped observation  (1 = first arrival)")
    ax.set_ylabel(r"score perturbation  $\Delta_g(k)$")
    ax.set_title("a late arrival swings the score by $\\sim\\log n$; an early one by $\\sim1$",
                 loc="left", fontsize=10)
    ax.text(0.40, 0.86,
            r"$\Delta_g(k)=1-(H_n-H_{n-k})$" "\n" r"$H_m=\sum_{i=1}^{m}\frac{1}{i}$",
            transform=ax.transAxes, fontsize=8.5, color=MUT, va="top")
    fig.tight_layout()
    save(fig, out, PARAMS, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
