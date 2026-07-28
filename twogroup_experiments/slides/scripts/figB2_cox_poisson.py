"""Backup B2 - exponential arrivals: where enriched genes land.  [local]

Background X ~ Exp(lambda0), enriched Y ~ Exp(lambda1), ratio rho = lambda1/lambda0.
The enriched gene's background quantile U = F0(Y) ~ Beta(1, rho).

The point: the LEFT tail of the enriched-slower scenario (an enriched gene
arriving early) is much heavier than the RIGHT tail of the enriched-faster
scenario (an enriched gene arriving late). We shade
  enriched-slower on [0, 0.2]  and  enriched-faster on [0.8, 1.0]
and report the two tail probabilities. Coloured by the ordering each favours
(forward = cox green, reverse = cox-reversed amber).
"""
import argparse

import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt

from _common import save, M_COX, M_COX_REVERSED, INK

PARAMS = {"rho_fast": 4.0, "rho_slow": 0.25, "grid": 600, "pdf_ycap": 4.5,
          "left": 0.2, "right": 0.8}


def main(out: str) -> None:
    p = PARAMS
    u = np.linspace(1e-3, 1 - 1e-3, p["grid"])
    fast = beta_dist.pdf(u, 1.0, p["rho_fast"])
    slow = beta_dist.pdf(u, 1.0, p["rho_slow"])

    fig, ax = plt.subplots(figsize=(5.2, 3.9))       # 4:3 landscape
    ax.plot(u, fast, color=M_COX, lw=2.4, label=r"enriched faster $\rho{=}4$ (forward)")
    ax.plot(u, slow, color=M_COX_REVERSED, lw=2.4, label=r"enriched slower $\rho{=}0.25$ (reverse)")

    # slower's left tail [0, left]; faster's right tail [right, 1]
    lm = u <= p["left"]
    rm = u >= p["right"]
    ax.fill_between(u[lm], 0, slow[lm], color=M_COX_REVERSED, alpha=0.30)
    ax.fill_between(u[rm], 0, fast[rm], color=M_COX, alpha=0.30)

    p_slow_left = float(beta_dist.cdf(p["left"], 1.0, p["rho_slow"]))
    p_fast_right = float(beta_dist.sf(p["right"], 1.0, p["rho_fast"]))
    # tail probabilities in the clear central band, colour-coded (no arrows / prefixes)
    ax.text(0.42, 3.05, r"$P(U>%.1f)=%.4f$" % (p["right"], p_fast_right),
            color=M_COX, fontsize=9, ha="left", va="center")
    ax.text(0.42, 2.45, r"$P(U<%.1f)=%.3f$" % (p["left"], p_slow_left),
            color=M_COX_REVERSED, fontsize=9, ha="left", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, p["pdf_ycap"])
    ax.set_xlabel(r"arrival quantile among background  $u = F_0(Y)$")
    ax.set_ylabel("density  (capped)")
    ax.set_title(r"$U \sim \mathrm{Beta}(1,\rho)$: the slow early tail $\gg$ the fast late tail",
                 loc="left", fontsize=10)
    ax.legend(loc="upper center", frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, out, {**PARAMS, "p_slow_left": p_slow_left, "p_fast_right": p_fast_right}, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
