"""The three NB routes against exact CAVI-Q2, per replicate.

gibss fits the negative binomial three ways, named by the base regression problem each
reduces to: the logistic cumulant at eta - log r (`nb_est`, gIBSS on the free-form Q1
kernel; `nb_cavi`, exact CAVI in Q2 on the same likelihood), the local Polya-Gamma
augmentation (`nb_polya_gamma`, the JJ bound on the NB base) and the local Poisson-Gamma
augmentation (`nb_poisson_gamma`, Jensen on log(r + e^eta)). `nb_cavi` maximizes the exact
NB ELBO over Gaussian q, so it is the reference; the augmentations are certified lower
bounds on that ELBO and should score below it. Everything here is read off the pipeline
tables: `fit.elbo` is the exact ELBO of every count-arm state (gibss.elbo.compute_elbo),
`pip`, `cs` the usual.

Per (cell, L, rep, arm != ref):
    d_elbo        exact ELBO, arm minus ref (nats; <= 0 means the ref found the better q)
    pip_max_diff  max_j |pip_j - pip_j^ref|
    same_sets     every declared set (ser_log_bf >= 2) of the arm is a declared set of the
                  ref with identical membership, and vice versa
    sd_ratio      beta_sd of the top declared component, arm over ref (matched by the
                  component's top feature; null when the top features differ). The
                  Poisson-Gamma docstring predicts about sqrt(1 - s), s = mu/(mu + r).
    t_ratio       wall seconds, arm over ref
    s_nb          mu0 / (mu0 + r_true): the Poisson-Gamma variance-deflation driver
"""

from __future__ import annotations

import numpy as np
import polars as pl

NB_ARMS = ["nb_est", "nb_polya_gamma", "nb_poisson_gamma", "nb_cavi"]
LABEL = {"nb_est": "gIBSS-Q1 (logistic)", "nb_polya_gamma": "Polya-Gamma", "nb_poisson_gamma": "Poisson-Gamma",
         "nb_cavi": "CAVI-Q2 (logistic)", "poisson": "gIBSS-Q1 (Poisson)", "poisson_cavi": "CAVI-Q2 (Poisson)"}
COLOR = {"nb_est": "#E69F00", "nb_polya_gamma": "#882255", "nb_poisson_gamma": "#44AA99", "nb_cavi": "#000000",
         "poisson": "#117733", "poisson_cavi": "#332288"}
KEY = ["cell", "L", "rep"]


def _declared_sets(cs: pl.DataFrame, min_log_bf: float = 2.0) -> pl.DataFrame:
    """One dict per (cell, L, rep, arm) with the declared sets as a frozenset of member
    frozensets, plus the top declared component's top feature and beta_sd."""
    d = cs.filter(pl.col("ser_log_bf") >= min_log_bf)
    rows = []
    for (cell, L, rep, arm), g in d.group_by(KEY + ["arm"]):
        sets = frozenset(frozenset(map(int, m)) for m in g["cs"].to_list())
        top = g.sort("ser_log_bf", descending=True).row(0, named=True)
        rows.append({"cell": cell, "L": L, "rep": rep, "arm": arm, "sets": sets,
                     "top": int(top["top"]), "top_sd": float(top["beta_sd"])})
    return rows


EMPTY = pl.DataFrame(schema={**{k: pl.Utf8 for k in KEY}, "arm": pl.Utf8})


def route_frame(pip: pl.DataFrame, cs: pl.DataFrame, fit: pl.DataFrame, ref: str = "nb_cavi", arms=None) -> pl.DataFrame:
    """`arms` defaults to the NB routes; pass e.g. ["poisson"] with ref="poisson_cavi" for
    the Poisson gIBSS-vs-CAVI check (the exact ELBOs are only comparable within a family)."""
    present = set(fit["arm"])
    arms = [a for a in (NB_ARMS if arms is None else arms) if a != ref and a in present]
    if ref not in present or not arms:
        return EMPTY
    f = fit.filter(pl.col("arm").is_in(arms + [ref])).with_columns(pl.col("L").cast(pl.Utf8))
    coords = f.select(KEY + ["n", "mu0", "phi", "beta_abs", "k_causal", "libsize_sd", "r_true"]).unique(KEY)
    fr = f.filter(pl.col("arm") == ref).select(KEY + [pl.col("elbo").alias("elbo_ref"), pl.col("seconds").alias("seconds_ref")])
    base = (f.filter(pl.col("arm").is_in(arms)).select(KEY + ["arm", "elbo", "seconds"])
            .join(fr, on=KEY, how="inner")
            .with_columns((pl.col("elbo") - pl.col("elbo_ref")).alias("d_elbo"),
                          (pl.col("seconds") / pl.col("seconds_ref")).alias("t_ratio")))

    # PIP max abs difference
    p = pip.filter(pl.col("arm").is_in(arms + [ref])).with_columns(pl.col("L").cast(pl.Utf8))
    pr = p.filter(pl.col("arm") == ref).select(KEY + ["feature", pl.col("pip").alias("pip_ref")])
    pd_ = (p.filter(pl.col("arm").is_in(arms)).join(pr, on=KEY + ["feature"], how="inner")
           .group_by(KEY + ["arm"]).agg((pl.col("pip") - pl.col("pip_ref")).abs().max().alias("pip_max_diff")))

    # declared-set identity and top-component sd ratio
    rows = _declared_sets(cs.with_columns(pl.col("L").cast(pl.Utf8)))
    by = {(r["cell"], r["L"], r["rep"], r["arm"]): r for r in rows}
    agree = []
    for r in base.select(KEY + ["arm"]).iter_rows(named=True):
        k = (r["cell"], r["L"], r["rep"])
        a, b = by.get((*k, r["arm"])), by.get((*k, ref))
        sa = a["sets"] if a else frozenset()
        sb = b["sets"] if b else frozenset()
        sd = (a["top_sd"] / b["top_sd"]) if (a and b and a["top"] == b["top"]) else None
        agree.append({**{c: r[c] for c in KEY + ["arm"]}, "same_sets": sa == sb,
                      "n_decl": len(sa), "n_decl_ref": len(sb), "sd_ratio": sd})
    ag = pl.DataFrame(agree, schema_overrides={"sd_ratio": pl.Float64})
    out = (base.join(pd_, on=KEY + ["arm"], how="left").join(ag, on=KEY + ["arm"], how="left")
           .join(coords, on=KEY, how="left")
           .with_columns((pl.col("mu0") / (pl.col("mu0") + pl.col("r_true"))).alias("s_nb")))
    return out.sort(KEY + ["arm"])


def route_summary(rf: pl.DataFrame) -> pl.DataFrame:
    return (rf.group_by(["arm", "L"]).agg(
        pl.len().alias("fits"),
        pl.col("d_elbo").median().alias("median_d_elbo"),
        pl.col("d_elbo").min().alias("min_d_elbo"),
        pl.col("d_elbo").max().alias("max_d_elbo"),
        pl.col("pip_max_diff").median().alias("median_pip_max_diff"),
        pl.col("pip_max_diff").max().alias("max_pip_max_diff"),
        pl.col("same_sets").mean().alias("same_sets_rate"),
        pl.col("sd_ratio").median().alias("median_sd_ratio"),
        pl.col("t_ratio").median().alias("median_t_ratio"),
    ).sort(["L", "arm"]))


def route_figure(rf: pl.DataFrame, ref: str = "nb_cavi", L: str = "5", seed: int = 0):
    """Per-rep strips, one band per arm: ELBO gap to the reference, PIP max difference, top
    component sd ratio, wall-time ratio. Point colour = s = mu0 / (mu0 + r), the predicted
    Poisson-Gamma deflation driver (dotted = sqrt(1 - s) at the colour-scale extremes)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    rng = np.random.default_rng(seed)
    d = rf.filter(pl.col("L") == str(L))
    arms = [a for a in (NB_ARMS + ["poisson"]) if a in set(d["arm"])]
    panels = [("d_elbo", "exact ELBO, arm - ref (nats)", "symlog"), ("pip_max_diff", "max |PIP - PIP_ref|", "linear"),
              ("sd_ratio", "sd of top component, arm / ref", "log"), ("t_ratio", "seconds, arm / ref", "log")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 2.6), constrained_layout=True)
    norm = Normalize(0, 1)
    cmap = plt.get_cmap("viridis")
    for ax, (col, lab, scale) in zip(axes, panels):
        for i, arm in enumerate(arms):
            g = d.filter(pl.col("arm") == arm).drop_nulls(col)
            y = g[col].to_numpy()
            x = i + rng.uniform(-0.3, 0.3, len(y))
            ax.axvspan(i - 0.45, i + 0.45, color=COLOR[arm], alpha=0.08, lw=0)
            ax.scatter(x, y, s=9, c=cmap(norm(g["s_nb"].to_numpy())), lw=0, alpha=0.85)
            med = np.nanmedian(y) if len(y) else np.nan
            ax.plot([i - 0.4, i + 0.4], [med, med], color=COLOR[arm], lw=1.4)
        ref_line = 0.0 if col in ("d_elbo", "pip_max_diff") else 1.0
        ax.axhline(ref_line, color="0.5", lw=0.8, ls="--")
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=1.0)
        elif scale == "log":
            ax.set_yscale("log")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([LABEL[a].split(" ")[0] for a in arms], fontsize=7, rotation=20)
        ax.set_title(lab, fontsize=8)
        ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.01).set_label("s = mu0 / (mu0 + r)", fontsize=7)
    fig.suptitle(f"NB routes vs {LABEL[ref]}, L = {L}, one point per rep", fontsize=9)
    return fig
