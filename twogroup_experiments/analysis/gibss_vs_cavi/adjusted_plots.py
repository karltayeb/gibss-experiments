"""Plotters and tables for `adjusted_vs_gibss.qmd`, reading the artifact `generate_adjusted.py`
writes. Kept out of the notebook so they can be exercised against the smoke artifact."""
from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import adjusted_lib as al

INK = "#16202E"
COLOR = dict(al.COLOR, adjusted_nr="#005A42")  # darker green, dashed
LABEL = dict(al.LABEL, adjusted_nr="adjusted, null not re-anchored")
LS = {"gibss": "-", "taylor2": "-", "adjusted": "-", "adjusted_nr": "--", "exact": "-", "cavi": "-"}
METRIC_LABEL = {
    "alpha_gap": r"max$_j\,|\alpha_j - \alpha_j^{\rm exact}|$",
    "pip_gap": r"max$_j\,|{\rm PIP}_j - {\rm PIP}_j^{\rm ref}|$",
    "logbf_gap": r"$\log{\rm BF} - \log{\rm BF}^{\rm ref}$ (nats)",
    "eb_gap": r"$\mathbb{E}[b] - \mathbb{E}[b]^{\rm ref}$ (top feature)",
    "sd_ratio": r"posterior sd / sd$^{\rm ref}$",
    "icpt_gap": r"$\hat b_0 - \hat b_0^{\rm ref}$",
}
LOGBF_CS = 2.0  # declared credible set


def load(path: str) -> dict:
    with open(path, "rb") as f:
        art = pickle.load(f)
    rows = art["rows"]
    for r in rows:
        for k in ("structure", "V", "signal", "rho", "n"):
            r.setdefault(k, None)
        r["causal"] = str(r["causal"])
    art["df"] = pl.DataFrame(rows, infer_schema_length=None)
    return art


def _band(ax, x, ys, color, ls, label, q=(0.1, 0.9)):
    """Mean over the seeds that produced a finite value, with a 10-90% band. `ys` is
    (n_x, n_seeds). A point where every seed failed (NaN fit) is left out, so the line
    stops; a point where some seeds failed gets a hollow marker."""
    fin = [v[np.isfinite(v)] for v in ys]
    keep = np.array([len(v) > 0 for v in fin])
    if not keep.any():
        return
    x = np.asarray(x)[keep]
    fin = [v for v, k in zip(fin, keep) if k]
    partial = np.array([len(v) < len(w) for v, w in zip(fin, [u for u, k in zip(ys, keep) if k])])
    m = np.array([np.mean(v) for v in fin])
    lo = np.array([np.quantile(v, q[0]) for v in fin])
    hi = np.array([np.quantile(v, q[1]) for v in fin])
    ax.plot(x, m, color=color, ls=ls, marker="o", ms=3.5, lw=1.4, label=label,
            markerfacecolor=color)
    if partial.any():
        ax.plot(x[partial], m[partial], ls="none", marker="o", ms=3.5, markerfacecolor="white",
                markeredgecolor=color)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)


def _series(df: pl.DataFrame, arm: str, metric: str, xcol: str = "V"):
    sub = df.filter(pl.col("arm") == arm).sort(xcol)
    xs = sorted(sub[xcol].unique().to_list())
    ys = [sub.filter(pl.col(xcol) == x)[metric].to_numpy() for x in xs]
    return np.array(xs), ys


def plot_metrics_vs_v(art, block: str, structure: str, signal: str, b0: dict,
                      metrics=("alpha_gap", "eb_gap", "icpt_gap", "logbf_gap"),
                      arms=("gibss", "taylor2", "adjusted"),
                      logbf_arms=("gibss", "taylor2", "adjusted", "adjusted_nr"),
                      figsize=(6.5, 4.6)):
    """Rows = family, columns = metric, x = V, one line per arm (mean over seeds, 10-90%
    band). `b0` picks one baseline per family. The reference arm is the zero line. A line
    that stops early is an arm whose every seed failed (NaN) beyond that V."""
    df = art["df"].filter((pl.col("block") == block) & (pl.col("structure") == structure)
                          & (pl.col("signal") == signal))
    fams = ("logistic", "poisson")
    fig, axes = plt.subplots(len(fams), len(metrics), figsize=figsize, sharex=True)
    for r, fam in enumerate(fams):
        d = df.filter((pl.col("family") == fam) & (pl.col("b0") == b0[fam]))
        for c, met in enumerate(metrics):
            ax = axes[r][c]
            use = logbf_arms if met == "logbf_gap" else arms
            for arm in use:
                x, ys = _series(d, arm, met)
                if len(x) == 0:
                    continue
                _band(ax, x, ys, COLOR[arm], LS[arm], LABEL[arm])
            ax.axhline(1.0 if met == "sd_ratio" else 0.0, color="0.6", lw=0.8, ls=":")
            if met == "logbf_gap":
                ax.set_yscale("symlog", linthresh=1.0)
            if r == 0:
                ax.set_title(METRIC_LABEL[met], fontsize=8.5, color=INK)
            if r == len(fams) - 1:
                ax.set_xlabel("offset variance $V$")
            if c == 0:
                ax.set_ylabel(f"{al.FAMILY_LABEL[fam]}\n($b_0={b0[fam]}$)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plot_b0_sweep(art, family: str, arm: str = "adjusted", structure: str = "homo",
                  metrics=("alpha_gap", "eb_gap", "icpt_gap", "logbf_gap"), figsize=(6.5, 2.4)):
    """One arm, one family: metric vs V with one line per baseline b0 (the logistic pole at
    p = 1/2 is b0 = 0). Mean over seeds."""
    df = art["df"].filter((pl.col("block") == "exo_ser") & (pl.col("structure") == structure)
                          & (pl.col("signal") == "signal") & (pl.col("family") == family)
                          & (pl.col("arm") == arm))
    b0s = sorted(df["b0"].unique().to_list(), reverse=True)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    for c, met in enumerate(metrics):
        ax = axes[c]
        for i, b0 in enumerate(b0s):
            x, ys = _series(df.filter(pl.col("b0") == b0), arm, met)
            m = np.array([np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan
                          for v in ys])
            ax.plot(x, m, color=cmap(i / max(len(b0s) - 1, 1)), marker="o", ms=3, lw=1.3,
                    label=f"$b_0={b0:g}$")
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
        if met == "logbf_gap":
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(METRIC_LABEL[met], fontsize=8.5, color=INK)
        ax.set_xlabel("$V$")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{al.FAMILY_LABEL[family]}, {LABEL[arm]}", fontsize=9.5, color=INK, y=1.02)
    fig.tight_layout()
    return fig


def plot_false_cs(art, b0: dict, arms=("gibss", "taylor2", "adjusted", "adjusted_nr", "exact"),
                  figsize=(6.5, 2.6)):
    """Null-mean, heteroskedastic V (on half the rows): rate of a declared credible set
    (SER log BF > LOGBF_CS) over seeds vs V, per arm. Any declared set is a false positive."""
    df = art["df"].filter((pl.col("block") == "exo_ser") & (pl.col("structure") == "hetero")
                          & (pl.col("signal") == "null"))
    fams = ("logistic", "poisson")
    fig, axes = plt.subplots(1, len(fams), figsize=figsize, sharey=True)
    for c, fam in enumerate(fams):
        ax = axes[c]
        d = df.filter((pl.col("family") == fam) & (pl.col("b0") == b0[fam]))
        for arm in arms:
            x, ys = _series(d, arm, "logbf")
            if len(x) == 0:
                continue
            rate = np.array([np.mean(v[np.isfinite(v)] > LOGBF_CS) if np.isfinite(v).any()
                             else np.nan for v in ys])
            ax.plot(x, rate, color=COLOR[arm], ls=LS[arm], marker="o", ms=3.5, lw=1.4,
                    label=LABEL[arm])
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{al.FAMILY_LABEL[fam]} ($b_0={b0[fam]}$)", fontsize=9.5, color=INK)
        ax.set_xlabel("offset variance $V$ on subgroup $H$")
        if c == 0:
            ax.set_ylabel(f"false-CS rate (log BF > {LOGBF_CS:g})")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return fig


def plot_pip_example(art, block: str, family: str, V: float,
                     arms=("gibss", "taylor2", "adjusted", "exact"),
                     figsize=(6.5, 3.4)):
    """PIP stems of the seed-0 example of one cell, one row per arm. Causal columns in
    vermillion; the band spans +-2 columns (corr >= 0.81)."""
    ex = next(e for e in art["examples"] if e["block"] == block and e["cell"]["family"] == family
              and e["cell"]["V"] == V and e["cell"]["signal"] == "signal"
              and e["cell"]["structure"] == "homo")
    causal = list(ex["cell"]["causal"])
    p = ex["cell"]["p"]
    figsize = (figsize[0], figsize[1] + 0.35 * len(arms))
    fig, axes = plt.subplots(len(arms), 1, figsize=figsize, sharex=True, sharey=True)
    for r, arm in enumerate(arms):
        ax = axes[r]
        rec = ex["recs"][arm]
        pip = rec["pip"]
        x = np.arange(p)
        if not np.all(np.isfinite(pip)):
            ax.set_title(f"{LABEL[arm]}:  fit failed (non-finite posterior)", loc="left",
                         fontsize=8, color=COLOR[arm], pad=3)
            ax.set_ylim(-0.02, 1.08)
            ax.set_ylabel("PIP", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            continue
        for cc in causal:
            ax.axvspan(cc - 2.5, cc + 2.5, color="#D55E00", alpha=0.08, lw=0)
        is_c = np.zeros(p, bool)
        is_c[causal] = True
        ax.vlines(x[~is_c], 0, pip[~is_c], color=COLOR[arm], alpha=0.8, lw=1.6)
        ax.vlines(x[is_c], 0, pip[is_c], color="#D55E00", lw=2.4)
        ax.scatter(x[is_c], pip[is_c], s=28, color="#D55E00", zorder=5)
        eb = ", ".join(f"{rec['eb'][cc]:.2f}" for cc in causal)
        ax.set_title(f"{LABEL[arm]}:  $\\hat b_0$ = {rec['intercept']:.2f},  "
                     f"$\\mathbb{{E}}[b]$ at the causal columns = {eb}",
                     loc="left", fontsize=8, color=COLOR[arm], pad=3)
        ax.set_ylim(-0.02, 1.08)
        ax.set_ylabel("PIP", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("column")
    fig.suptitle(f"{al.FAMILY_LABEL[family]}, L={ex['cell']['L']}, V={V:g}, "
                 f"$\\beta$={ex['cell']['beta']:g}, seed 0", fontsize=9.5, color=INK)
    fig.tight_layout()
    return fig


def plot_endo(art, metric: str = "pip_gap", figsize=(6.5, 2.8)):
    """Endogenous V: each dataset is a point, x = realized 95th-percentile message variance
    at convergence (the V the SERs actually saw), y = the metric vs exact Q1 CAVI; colour =
    arm. One panel per family."""
    df = art["df"].filter((pl.col("block") == "endo") & (pl.col("arm") != "cavi"))
    fams = ("logistic", "poisson")
    fig, axes = plt.subplots(1, len(fams), figsize=figsize)
    for c, fam in enumerate(fams):
        ax = axes[c]
        d = df.filter(pl.col("family") == fam)
        for arm in ("gibss", "taylor2", "adjusted"):
            s = d.filter(pl.col("arm") == arm)
            yv = s[metric].to_numpy()
            if metric in ("pip_gap", "alpha_gap"):
                yv = np.maximum(yv, 1e-6)  # log axis: floor exact agreement
            ax.scatter(s["msg_var_p95"].to_numpy(), yv, s=14,
                       color=COLOR[arm], alpha=0.65, lw=0, label=LABEL[arm])
        ax.set_xscale("log")
        if metric in ("pip_gap", "alpha_gap"):
            ax.set_yscale("log")
        elif metric == "logbf_gap":
            ax.set_yscale("symlog", linthresh=0.1)
            ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(al.FAMILY_LABEL[fam], fontsize=9.5, color=INK)
        ax.set_xlabel("realized $V$ (95th pct of rows)")
        if c == 0:
            ax.set_ylabel(METRIC_LABEL[metric], fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def endo_table(art) -> str:
    """Markdown: per (family, n, rho) the realized V and each arm's mean gaps vs CAVI."""
    df = art["df"].filter(pl.col("block") == "endo")
    g = (df.filter(pl.col("arm") == "cavi")
         .group_by(["family", "n", "rho"]).agg(pl.col("msg_var_mean").mean().alias("V_mean"),
                                               pl.col("msg_var_p95").mean().alias("V_p95")))
    arms = ("gibss", "taylor2", "adjusted")
    lines = ["| family | n | rho | mean V | p95 V | PIP gap: gIBSS | Taylor2 | adjusted | "
             "logBF gap: gIBSS | Taylor2 | adjusted | s: gIBSS / Taylor2 / adjusted / CAVI |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in g.sort(["family", "n", "rho"]).iter_rows(named=True):
        sel = df.filter((pl.col("family") == row["family"]) & (pl.col("n") == row["n"])
                        & (pl.col("rho") == row["rho"]))
        def m(arm, col):
            return float(sel.filter(pl.col("arm") == arm)[col].mean())
        pip = " | ".join(f"{m(a, 'pip_gap'):.4f}" for a in arms)
        lbf = " | ".join(f"{m(a, 'logbf_gap'):+.2f}" for a in arms)
        secs = " / ".join(f"{m(a, 'secs'):.1f}" for a in arms + ("cavi",))
        lines.append(
            f"| {row['family']} | {row['n']} | {row['rho']:g} | {row['V_mean']:.3f} | "
            f"{row['V_p95']:.3f} | {pip} | {lbf} | {secs} |")
    return "\n".join(lines)


def exo_table(art, structure: str, signal: str, b0: dict, L: int = 1) -> str:
    """Markdown: per (family, V) the mean gaps of each arm vs the exact GH fit."""
    block = "exo_ser" if L == 1 else "exo_multi"
    df = art["df"].filter((pl.col("block") == block) & (pl.col("structure") == structure)
                          & (pl.col("signal") == signal))
    lines = ["| family | V | arm | alpha gap | PIP gap | E[b] gap | sd ratio | b0 gap | logBF gap | logBF |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for fam in ("logistic", "poisson"):
        d = df.filter((pl.col("family") == fam) & (pl.col("b0") == b0[fam]))
        for V in sorted(d["V"].unique().to_list()):
            for arm in ("gibss", "taylor2", "adjusted", "adjusted_nr", "exact"):
                s = d.filter((pl.col("V") == V) & (pl.col("arm") == arm))
                if s.height == 0:
                    continue
                nfail = int(s["failed"].sum())
                if nfail == s.height:
                    lines.append(f"| {fam} | {V:g} | {LABEL[arm]} | failed {nfail}/{s.height} "
                                 f"(non-finite fit) | | | | | | |")
                    continue
                if nfail:
                    s = s.filter(~pl.col("failed"))
                lines.append(
                    f"| {fam} | {V:g} | {LABEL[arm]} | {float(s['alpha_gap'].mean()):.4f} | "
                    f"{float(s['pip_gap'].mean()):.4f} | {float(s['eb_gap'].mean()):+.3f} | "
                    f"{float(s['sd_ratio'].mean()):.3f} | {float(s['icpt_gap'].mean()):+.3f} | "
                    f"{float(s['logbf_gap'].mean()):+.2f} | {float(s['logbf'].mean()):.1f}"
                    + (f" ({nfail}/{nfail + s.height} failed, excluded) |" if nfail else " |"))
    return "\n".join(lines)


def failure_table(art) -> str:
    """Markdown: every (block, family, structure, b0, V, arm) with at least one failed
    (non-finite) fit, with the failed count over seeds. Empty string if none failed."""
    df = art["df"].filter(pl.col("block") != "endo")
    g = (df.group_by(["block", "family", "structure", "signal", "b0", "V", "arm"])
           .agg(pl.col("failed").sum().alias("nfail"), pl.len().alias("n"))
           .filter(pl.col("nfail") > 0)
           .sort(["block", "family", "structure", "signal", "b0", "V", "arm"]))
    if g.height == 0:
        return ""
    lines = ["| block | family | V structure | signal | b0 | V | arm | failed fits |",
             "|---|---|---|---|---:|---:|---|---:|"]
    for r in g.iter_rows(named=True):
        lines.append(f"| {r['block']} | {r['family']} | {r['structure']} | {r['signal']} | "
                     f"{r['b0']:g} | {r['V']:g} | {LABEL[r['arm']]} | {r['nfail']}/{r['n']} |")
    return "\n".join(lines)
