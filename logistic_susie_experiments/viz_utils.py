"""Plot helpers retained after the faceting redesign. The old per-analysis
renderers (render_*/make_*) were dead duplicates of the analyses/ hooks and
were removed; only these three live helpers remain."""
from __future__ import annotations

import numpy as np


def method_family_label_map() -> dict[str, str]:
    return {
        "logistic_threshold":    "Logistic",
        "cox":                   "Cox",
        "twogroup":              "Twogroup",
        "twogroup_oracle":       "Twogroup",
        "twogroup_oracle_global": "Twogroup Global EM",
        "logistic_oracle":       "Logistic",
        "cox_reversed":          "Cox (reversed)",
        "cox_reversed_censored": "Cox reversed (censored)",
        "cox_uncensored":        "Cox (uncensored)",
        "twogroup_oracle_init":  "TG Oracle Init",
        "twogroup_scale_fam":    "Twogroup Scale",
        "twogroup_loc_fam":      "Twogroup Loc",
        "linear_fixed":          "Linear",
        "linear_estimated":      "Linear (est. var)",
        "depletion":             "Depletion",
        "enrichment":            "Enrichment",
        "globaljj":              "Global JJ",
        "localjj":               "Local JJ",
        "quadrature":            "Quadrature",
        "irls":                  "IRLS (Laplace)",
        "irls_block":            "IRLS block",
        "globaljj_block":        "Global JJ block",
        "globaljj_block_1":      "Global JJ block-1",
        # 002_global families (factor = family x step x prior; center via __suffix)
        "irls_block_1_eb":       "IRLS 1-step EB",
        "irls_block_1_fixed":    "IRLS 1-step fixed",
        "irls_block_eb":         "IRLS conv EB",
        "irls_block_fixed":      "IRLS conv fixed",
        "globaljj_block_1_eb":   "JJ 1-step EB",
        "globaljj_block_1_fixed": "JJ 1-step fixed",
        "globaljj_block_eb":     "JJ conv EB",
        "globaljj_block_fixed":  "JJ conv fixed",
        "block_irls":            "Block IRLS",
        "irls_1step":            "IRLS 1-step",
        "irls_conv":             "IRLS converged",
        "profile_cheb":          "Profile (Cheb)",
        "score":                 "Score (b0=0)",
        "score_null":            "Score (null b0)",
        "score_null_intercept":  "Score (null b0)",
    }


def dim_palette(values: list) -> dict:
    """Stable colorblind-safe palette keyed on a dimension's distinct values.
    Booleans/strings/numbers all supported; order-stable for legend determinism."""
    okabe_ito = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
                 "#56B4E9", "#AA4499", "#882255", "#332288", "#999999"]
    out = {}
    for i, v in enumerate(values):
        out[v] = okabe_ito[i % len(okabe_ito)]
    return out


def _bins_to_power_fdp(
    counts: np.ndarray,
    causal_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rev_cum_counts = np.cumsum(counts[::-1])[::-1]
    rev_cum_causal = np.cumsum(causal_counts[::-1])[::-1]
    total_causal = int(causal_counts.sum())
    power = rev_cum_causal / max(total_causal, 1)
    fdp = (rev_cum_counts - rev_cum_causal) / np.maximum(rev_cum_counts, 1)
    return power.astype(float), fdp.astype(float)
