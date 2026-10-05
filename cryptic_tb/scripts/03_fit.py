#!/usr/bin/env python
"""Fit logistic SuSiE per drug and write annotated credible-set tables.

Covariates (site, plate design, lineage) enter as a fixed per-row offset: gibss's front
door takes an `offset` but no fixed-covariate block, so the covariate-only logistic fit's
linear predictor is passed through. That is an approximation -- the covariate
coefficients are not re-estimated jointly with the variant effects -- and both the
with-offset and no-offset fits are reported so the reader can see what it costs.

Run from cryptic_tb/:
    uv run python scripts/03_fit.py            # every drug in config.yaml
    uv run python scripts/03_fit.py RIF INH    # named drugs only
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import sparse
from scipy.special import expit
from scipy.stats import chi2, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROC = ROOT / "data/processed"
RESULTS = ROOT / "results"


def logistic_irls(Z: np.ndarray, y: np.ndarray, *, ridge: float = 1.0,
                  offset: np.ndarray | None = None, max_iter: int = 200,
                  tol: float = 1e-9, max_step: float = 2.0) -> np.ndarray:
    """Ridge-penalised logistic regression by damped IRLS.

    Both the penalty and the damping are load-bearing here. Resistance is close to
    deterministic and several sampling sites contribute almost only susceptible isolates,
    so plenty of covariate levels separate the outcome perfectly. Undamped Newton on a
    separating level walks the coefficient toward infinity, the weights underflow, the
    Hessian goes singular and the fit returns non-finite values -- which then poison every
    downstream score test through the offset. Capping the per-iteration step and keeping a
    real ridge on the non-intercept columns holds the fit finite.
    """
    n, k = Z.shape
    beta = np.zeros(k)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    pen = ridge * np.eye(k)
    pen[0, 0] = 0.0  # never penalise the intercept
    for _ in range(max_iter):
        eta = Z @ beta + off
        mu = expit(eta)  # stable at |eta| >> 1
        w = np.clip(mu * (1 - mu), 1e-8, None)
        grad = Z.T @ (y - mu) - pen @ beta
        hess = (Z * w[:, None]).T @ Z + pen
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        scale = np.max(np.abs(step))
        if scale > max_step:            # damp: a separating level would otherwise run off
            step = step * (max_step / scale)
        beta = beta + step
        if scale < tol:
            break
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError("logistic IRLS did not stay finite")
    return beta


def covariate_matrix(meta: pl.DataFrame, cfg) -> tuple[np.ndarray, list[str]]:
    """Dense (n, k) covariate matrix Z for `fit_glm_susie(covariates=Z)`.

    No intercept column: gibss forms the nuisance design W = [1 | Z] itself and rejects a
    constant column as rank deficient. One level per factor is dropped as the reference.
    """
    n = meta.height
    cols: list[np.ndarray] = []
    names: list[str] = []
    cov = cfg["covariates"]
    for field, enabled in (("site", cov.get("site")), ("plate", cov.get("plate_type")),
                           ("lineage", cov.get("lineage", True))):
        if not enabled:
            continue
        values = meta[field].to_list()
        levels = sorted(set(values))[1:]  # drop the first level as the reference
        for lv in levels:
            cols.append(np.array([1.0 if v == lv else 0.0 for v in values]))
            names.append(f"{field}={lv}")
    if not cols:
        return np.zeros((n, 0)), []
    return np.column_stack(cols), names


def covariate_offset(meta: pl.DataFrame, cfg) -> tuple[np.ndarray, list[str]]:
    """The OLD route, kept only to quantify what the approximation cost.

    Fits the covariates once on their own and freezes their linear predictor as an offset.
    The variant effects then never feed back into the covariate coefficients. Superseded by
    `covariates=Z`, which fits (b0, gamma) jointly with the effects.
    """
    y = meta["y"].to_numpy().astype(float)
    n = len(y)
    cols: list[np.ndarray] = [np.ones(n)]
    names = ["intercept"]
    cov = cfg["covariates"]
    for field, enabled in (("site", cov.get("site")), ("plate", cov.get("plate_type")),
                           ("lineage", cov.get("lineage", True))):
        if not enabled:
            continue
        values = meta[field].to_list()
        levels = sorted(set(values))[1:]  # drop the first level as the reference
        for lv in levels:
            cols.append(np.array([1.0 if v == lv else 0.0 for v in values]))
            names.append(f"{field}={lv}")
    Z = np.column_stack(cols)
    beta = logistic_irls(Z, y, ridge=1.0)
    eta = Z @ beta
    return eta - eta.mean(), names  # centred: gibss estimates its own intercept


def marginal_tests(X: sparse.csr_matrix, y: np.ndarray,
                   offset: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-variant marginal statistics, as the comparator SuSiE is judged against.

    Two tests, because neither alone is trustworthy on this data:

    `lrt_p` is a likelihood-ratio test from a damped, lightly penalised logistic fit that
    carries the covariate offset. It is the like-for-like comparison with the SuSiE fit,
    which also conditions on the covariates.

    `fisher_p` is Fisher's exact test on the 2x2 table, ignoring covariates. It is exact
    at any cell count and finite under complete separation, both of which matter when a
    variant has five carriers that are all resistant.

    A score test was tried first and rejected: evaluated at beta=0 it uses the null
    variance, so for a variant with an odds ratio in the hundreds it returned p ~ 1e-283
    where the likelihood-ratio test gives 1e-43. Valid as a test, useless as a reported
    number.
    """
    n, p = X.shape
    Xc = X.tocsc()
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    b0 = logistic_irls(np.ones((n, 1)), y, offset=off, ridge=0.0)
    mu0 = np.clip(expit(off + b0[0]), 1e-12, 1 - 1e-12)
    ll0 = float(np.sum(y * np.log(mu0) + (1 - y) * np.log1p(-mu0)))

    beta = np.full(p, np.nan)
    lrt_p = np.ones(p)
    fisher_p = np.ones(p)
    odds = np.full(p, np.nan)
    ones = np.ones(n)
    for j in range(p):
        xj = np.asarray(Xc[:, j].todense()).ravel()
        Z = np.column_stack([ones, xj])
        try:
            bj = logistic_irls(Z, y, offset=off, ridge=1e-6, max_iter=60)
        except FloatingPointError:
            bj = np.array([b0[0], np.nan])
        if np.isfinite(bj[1]):
            mu = np.clip(expit(Z @ bj + off), 1e-12, 1 - 1e-12)
            ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log1p(-mu)))
            beta[j] = bj[1]
            lrt_p[j] = chi2.sf(max(2.0 * (ll - ll0), 0.0), df=1)
        carrier = xj > 0
        a11 = int(np.sum(carrier & (y > 0)))
        a12 = int(np.sum(carrier & (y == 0)))
        a21 = int(np.sum(~carrier & (y > 0)))
        a22 = int(np.sum(~carrier & (y == 0)))
        odds[j], fisher_p[j] = fisher_exact([[a11, a12], [a21, a22]])
    return {"beta": beta, "lrt_p": lrt_p, "fisher_p": fisher_p, "odds_ratio": odds}


def to_bcoo(X: sparse.csr_matrix):
    """scipy CSR -> jax BCOO, keeping the design sparse through the fit."""
    from jax.experimental import sparse as jsparse
    import jax.numpy as jnp

    coo = X.tocoo()
    idx = jnp.stack([jnp.asarray(coo.row), jnp.asarray(coo.col)], axis=1)
    return jsparse.BCOO((jnp.asarray(coo.data), idx), shape=X.shape)


def fit_one(drug: str, cfg, *, L: int, method: str | None, use_offset: bool = True,
            covariate_mode: str = "joint",
            estimate_prior_variance: bool = True, prior_variance: float = 1.0,
            sparse_design: bool = True):
    """Fit one drug.

    `covariate_mode` is "joint" (covariates fit with the intercept as one Gaussian nuisance
    factor), "offset" (the old frozen-coefficient approximation) or "none". `use_offset`
    False forces "none" and is kept so older call sites keep working.
    """
    from gibss.methods import fit_glm_susie
    from gibss.summary import summarize_fit

    d = PROC / drug
    X = sparse.load_npz(d / "X.npz").tocsr()
    meta = pl.read_csv(d / "y.csv", schema_overrides={"mic": pl.String,
                                                  "site": pl.String})
    features = pl.read_csv(d / "features.csv")
    y = meta["y"].to_numpy().astype(float)

    if not use_offset:
        covariate_mode = "none"
    offset: float | np.ndarray = 0.0
    covariates = None
    cov_names: list[str] = []
    if covariate_mode == "joint":
        Z, cov_names = covariate_matrix(meta, cfg)
        covariates = Z if Z.shape[1] else None
    elif covariate_mode == "offset":
        offset, cov_names = covariate_offset(meta, cfg)
    elif covariate_mode != "none":
        raise ValueError(f"covariate_mode must be joint/offset/none, got {covariate_mode!r}")
    names = features["feature"].to_list()

    Xfit = to_bcoo(X) if sparse_design else X.toarray()
    t0 = time.time()
    state = fit_glm_susie(
        Xfit, y, L=L, method=method,
        offset=offset, covariates=covariates,
        center=True, estimate_intercept=True,
        estimate_prior_variance=estimate_prior_variance,
        prior_variance=prior_variance,
    )
    elapsed = time.time() - t0
    summary = summarize_fit(state, Xfit, feature_names=names,
                            coverage=cfg["fit"]["coverage"], expand_cs=True,
                            **({"covariate_names": cov_names} if covariate_mode == "joint"
                               and cov_names else {}))
    return {
        "drug": drug, "L": L, "method": method or "logistic",
        "covariate_mode": covariate_mode,
        "estimate_prior_variance": estimate_prior_variance,
        "state": state, "summary": summary, "features": features, "meta": meta,
        "X": X, "offset": offset, "cov_names": cov_names, "seconds": elapsed,
        "n": X.shape[0], "p": X.shape[1], "cases": int(y.sum()),
    }


def cs_table(fit, *, marginal: dict[str, np.ndarray] | None = None,
             min_log_bf: float = 2.0) -> pl.DataFrame:
    """One row per (credible set, variant) for the components with real evidence."""
    features = fit["features"]
    tbl = fit["summary"].table
    cols = {c: features[c].to_list() for c in
            ("feature", "gene", "mutation", "who_grade", "who_name", "who_match",
             "n_carriers", "n_carriers_resistant", "tier", "n_collapsed")}
    name_to_row = {f: i for i, f in enumerate(cols["feature"])}

    rows = []
    for rec in tbl.iter_rows(named=True):
        if rec["ser_log_bf"] < min_log_bf:
            continue
        cs = rec.get("cs") or []
        pips = rec.get("cs_pip") or []
        betas = rec.get("cs_beta") or []
        for feat, pip, beta in zip(cs, pips, betas):
            i = name_to_row[feat]
            carriers = cols["n_carriers"][i]
            res = cols["n_carriers_resistant"][i]
            row = {
                "component": rec["component"],
                "ser_log_bf": rec["ser_log_bf"],
                "cs_size": rec["cs_size"],
                "purity": rec.get("purity"),
                "feature": feat,
                "gene": cols["gene"][i],
                "mutation": cols["mutation"][i],
                "pip": pip,
                "beta": beta,
                "tier": cols["tier"][i],
                "who_grade": cols["who_grade"][i],
                "who_name": cols["who_name"][i],
                "who_match": cols["who_match"][i],
                "carriers": carriers,
                "carriers_resistant": res,
                "pct_resistant": round(100.0 * res / carriers, 1) if carriers else None,
                "n_collapsed": cols["n_collapsed"][i],
            }
            if marginal is not None:
                row["marginal_beta"] = float(marginal["beta"][i])
                row["lrt_p"] = float(marginal["lrt_p"][i])
                row["fisher_p"] = float(marginal["fisher_p"][i])
                row["odds_ratio"] = float(marginal["odds_ratio"][i])
            rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    drugs = sys.argv[1:] or list(cfg["drugs"])
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_cols(25)
    pl.Config.set_tbl_rows(60)

    L = int(os.environ.get("L", cfg["fit"]["L"][-1] if isinstance(cfg["fit"]["L"], list)
                           else cfg["fit"]["L"]))
    for drug in drugs:
        if not (PROC / drug / "X.npz").exists():
            print(f"[{drug}] no design built, skipping")
            continue
        print(f"\n{'=' * 100}\n{drug}  (L={L})\n{'=' * 100}")
        fit = fit_one(drug, cfg, L=L, method=None, covariate_mode="joint")
        print(f"n={fit['n']} cases={fit['cases']} p={fit['p']} "
              f"fit {fit['seconds']:.1f}s")
        print(fit["summary"])

        print("  marginal tests per variant (LRT + Fisher) ...", flush=True)
        # The marginal comparator conditions on the same covariates. There is no joint
        # nuisance factor in a one-variant GLM, so it uses the frozen-offset route.
        marg_offset, _ = covariate_offset(fit["meta"], cfg)
        marg = marginal_tests(
            fit["X"], fit["meta"]["y"].to_numpy().astype(float), marg_offset)
        table = cs_table(fit, marginal=marg)
        out = RESULTS / drug
        out.mkdir(parents=True, exist_ok=True)
        if table.height:
            table.write_csv(out / "credible_sets.csv")
            print(table)
        else:
            print("  no component reached log BF 2")
        # expand_cs=True adds per-feature list columns; CSV cannot hold them, and
        # credible_sets.csv already carries that content one row per variant.
        flat = fit["summary"].table.drop(
            [c for c, dt in fit["summary"].table.schema.items() if dt == pl.List]
        )
        flat.write_csv(out / "components.csv")
        with open(out / "fit.pkl", "wb") as fh:
            pickle.dump({k: v for k, v in fit.items() if k not in ("state", "X")}, fh)


if __name__ == "__main__":
    main()
