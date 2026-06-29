"""Compute the method-comparison tables for the 000_binary05 technical report.

Every number in the report comes from here (run: `uv run python scripts/report_binary05.py`).
Groups the 20 methods by (method-name, prior) and reports them under the three
contrasts the report analyses: global vs local, centered vs not, EB vs fixed,
plus the exact references (quadrature, profile).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from experiments import loader
from analyses.hooks import HOOKS

SC = "000_binary05"
GRID = (4, 8, 16, 32)  # signal rungs (0 = null, handled separately)


def load():
    cfg = loader.load_config()
    b = loader.load_sc_bundle(cfg, SC, ["pip", "cs"])
    return b["pip_plot_data"], b["cs_plot_data"]


def _key(r):  # (method, prior) identity
    return (r["method"], r["prior"])


def subset(df, **eq):
    out = df
    for k, v in eq.items():
        out = out.filter(pl.col(k) == v)
    return out


# --------------------------------------------------------------------------- #
# BF approximation accuracy vs the exact marginal (matched per sim/rep).
# Non-centered methods are compared to `quadrature` (exact, shared intercept);
# centered methods to `profile` (exact, profiled intercept).
# --------------------------------------------------------------------------- #
_CENTERED = {"taylor_local_c", "taylor_global_c", "jj_local_c", "jj_global_c", "profile"}


def bf_accuracy(cs, signal_only=True):
    """Per (method, prior): SER-logBF error vs the matched EXACT marginal on the
    same sims+prior. Non-centered -> quadrature; centered -> profile."""
    base = cs.filter(pl.col("signal")) if signal_only else cs
    rows = []
    for (m, prior) in sorted({(r["method"], r["prior"]) for r in base.iter_rows(named=True)}):
        ref_m = "profile" if m in _CENTERED else "quadrature"
        ref = subset(base, method=ref_m, prior=prior).select(
            ["batch_hash", "sample_id", "l", "ser_log_bf"]).rename({"ser_log_bf": "ref"})
        g = subset(base, method=m, prior=prior)
        j = g.join(ref, on=["batch_hash", "sample_id", "l"], how="inner")
        if j.height == 0:
            continue
        err = (j["ser_log_bf"] - j["ref"]).to_numpy()
        rows.append({
            "method": m, "prior": prior, "ref": ref_m,
            "bias": round(float(np.nanmean(err)), 3),
            "rmse": round(float(np.sqrt(np.nanmean(err ** 2))), 3),
            "max_abs": round(float(np.nanmax(np.abs(err))), 2),
            "n": int(np.isfinite(err).sum()),
        })
    return pl.DataFrame(rows).sort(["prior", "method"])


# --------------------------------------------------------------------------- #
# Per (method, prior): calibration + detection + resolution via the hooks.
# --------------------------------------------------------------------------- #
def per_method_table(pip, cs):
    rows = []
    for (m, prior) in sorted({(r["method"], r["prior"]) for r in cs.iter_rows(named=True)}):
        cs_m = subset(cs, method=m, prior=prior)
        pip_m = subset(pip, method=m, prior=prior)
        cs_sig = cs_m.filter(pl.col("signal"))
        # --- PIP calibration: class-stratified Brier (pooled, signal+null) ---
        pc = HOOKS["pip_calibration"].aggregate(pip_m.to_dicts())
        # --- detection: PIP power@FDP<=0.1 ---
        pf = HOOKS["power_fdp"].aggregate(pip_m.to_dicts())
        fdp, pwr = np.asarray(pf["fdp"]), np.asarray(pf["power"])
        p10 = float(pwr[fdp <= 0.1].max()) if (fdp <= 0.1).any() else float("nan")
        # --- detection: logBF ROC AUC (signal vs null) ---
        roc = HOOKS["cs_roc"].aggregate(cs_m.to_dicts())
        auc = float(roc.get("auc", float("nan")))
        # --- CS coverage at nominal beta=0.95 + size, calibrated beta ---
        cz = HOOKS["cs_coverage_size"].aggregate(cs_sig.to_dicts())
        betas = np.asarray(cz["betas"])
        i95 = int(np.argmin(np.abs(betas - 0.95)))
        cov95 = float(np.asarray(cz["coverage"])[i95])
        size95 = float(np.asarray(cz["cs_size"])[i95])
        sp = HOOKS["cs_size_power"].aggregate(cs_sig.to_dicts())
        rows.append({
            "method": m, "prior": prior,
            "B_causal": round(pc.get("brier_causal", float("nan")), 3),
            "B_null": round(pc.get("brier_null", float("nan")), 4),
            "pow@fdp.1": round(p10, 3),
            "auc": round(auc, 3),
            "cov@95": round(cov95, 3),
            "size@95": round(size95, 2),
            "cal_beta": round(sp.get("cal_beta", float("nan")), 2),
        })
    return pl.DataFrame(rows).sort(["method", "prior"])


if __name__ == "__main__":
    pip, cs = load()
    print("=== rows:", "pip", pip.shape, "cs", cs.shape)
    print("=== methods x priors present ===")
    print(sorted({(r["method"], r["prior"]) for r in cs.iter_rows(named=True)}))
    print("\n=== BF accuracy vs exact (matched) ===")
    print(bf_accuracy(cs))
    print("\n=== per-method metrics ===")
    with pl.Config(tbl_rows=40, tbl_cols=20):
        print(per_method_table(pip, cs))
