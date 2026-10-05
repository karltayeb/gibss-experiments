"""Figures: PIPs and effect sizes per arm, relative to CAVI-Q2 or absolute, along design density.

Everything comes from the cached fits (alpha, mu, var per component and feature) and the
simulation's true b. The design X is not cached; it is regenerated from the replicate seed
(the design sampler is the first draw of `core.simulate`) for the linear-predictor error.

PIP figure (rows), PIPs from the DECLARED components (log BF >= 2), as the CS metrics: an
undeclared component spreads ~1/p over every feature, which is invisible per feature but adds
~1 to any sum over the 253 nulls, so the full-fit sum would just count undeclared components.
  pip_causal  mean PIP over the causal features           (ref: arm minus CAVI)
  pip_null    sum of PIPs over the null features           (ref: arm minus CAVI)
              = expected number of false inclusions per fit
Effect figure (rows), the first three conditional on detection, i.e. over declared components
whose 95% CS contains a causal, on that causal (the one with the largest alpha if several):
  mean_ratio  mu / b, the component's conditional posterior mean over the truth (log axis;
              ref: ratio of the arm's mean to CAVI's). Far above 1 at small theta for every
              arm: a column with 5 members is near separation and the prior variance is
              estimated (max 100), so the conditional posterior is wide and right-skewed.
  int_cov     coverage of the 95% conditional posterior interval mu +- 1.96 sigma for b
  sd          the component's conditional posterior sd sigma    (ref: ratio, log axis)
  lp_rmse     sqrt(mean_i (x_i'(bhat - b))^2), bhat = sum_l alpha_l mu_l the marginal
              posterior mean; forgives mass moved among correlated columns (ref: ratio)
Bands: 95% intervals from resampling replicates, jointly for both arms of a difference.

    uv run python analysis/logistic_laplace_simulations/posterior_figure.py [SC] [redraw]
writes figures/{pip,effect}_{vs_cavi,absolute}.{pdf,png} (ser_-prefixed for 022-laplace-ser)
and figures/posterior_<SC>.parquet, the per-fit summary frame.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _TG_ROOT)
sys.path.insert(0, _HERE)
import core  # noqa: E402
import cs_tables as C  # noqa: E402
import delta_figure as F  # noqa: E402
import experiments.loader as loader  # noqa: E402
import laplacelib as R  # noqa: E402
from simulations.design import markov  # noqa: E402

Z95 = 1.959963984540054

PIP_METRICS = ["pip_causal", "pip_null"]
EFFECT_METRICS = ["mean_ratio", "int_cov", "sd", "lp_rmse"]
# how a per-rep (x, n) pair aggregates within an arm, and how an arm compares with the reference
#   agg: "mean" = mean of x over reps; "ratio" = sum x / sum n (n = per-rep count)
#   post: applied after aggregation (sqrt turns mean squared error into RMSE)
#   cmp: "diff" = arm - ref; "ratio" = arm / ref
SPEC = {
    "pip_causal": dict(agg="mean", post=None, cmp="diff"),
    "pip_null": dict(agg="mean", post=None, cmp="diff"),
    "mean_ratio": dict(agg="ratio", post=None, cmp="ratio"),
    "int_cov": dict(agg="ratio", post=None, cmp="diff"),
    "sd": dict(agg="ratio", post=None, cmp="ratio"),
    "lp_rmse": dict(agg="mean", post=np.sqrt, cmp="ratio"),
}
REL_LABEL = {"pip_causal": "PIP on causals", "pip_null": "sum PIP on nulls",
             "mean_ratio": "posterior mean ratio", "int_cov": "95% interval coverage",
             "sd": "posterior sd ratio", "lp_rmse": "predictor RMSE ratio"}
ABS_LABEL = {"pip_causal": "PIP on causals", "pip_null": "sum PIP on nulls",
             "mean_ratio": "posterior mean / truth", "int_cov": "95% interval coverage",
             "sd": "posterior sd", "lp_rmse": "predictor RMSE"}
REL_REF = {"pip_causal": 0.0, "pip_null": 0.0, "mean_ratio": 1.0, "int_cov": 0.0, "sd": 1.0, "lp_rmse": 1.0}
ABS_REF = {"pip_causal": None, "pip_null": None, "mean_ratio": 1.0, "int_cov": 0.95, "sd": None, "lp_rmse": None}
_R = [0.5, 0.71, 1, 1.41, 2]
REL_LOG = {"mean_ratio": _R, "sd": [0.25, 0.35, 0.5, 0.71, 1, 1.41, 2], "lp_rmse": _R}
ABS_LOG = {"mean_ratio": [1, 2, 4, 8], "sd": [0.0625, 0.125, 0.25, 0.5, 1, 2, 4],
           "lp_rmse": [0.0625, 0.125, 0.25, 0.5, 1]}


def truth(spec: core.SimulationSpec, replicate: int) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate one replicate's design (dense) and true coefficient vector b by re-running
    the pipeline's deterministic simulation (seeded by spec + replicate), so no
    simulations.parquet needs to be on disk. Works for every design function."""
    import jax.experimental.sparse as _jsp

    class _Dense:
        """Stand-in for BCOO during re-simulation: the sparse conversion is pure layout (no
        RNG), and skipping JAX here makes the per-replicate re-simulation ~100x faster."""
        def __init__(self, a): self.a = np.asarray(a, dtype=float)
        @classmethod
        def fromdense(cls, a): return cls(a)
        def todense(self): return self.a
        def __array__(self, dtype=None, copy=None): return self.a if dtype is None else self.a.astype(dtype)
        def __matmul__(self, b): return self.a @ np.asarray(b)
        @property
        def shape(self): return self.a.shape
    _orig = _jsp.BCOO
    _jsp.BCOO = _Dense
    try:
        sim = core.simulate(spec, int(replicate))
    finally:
        _jsp.BCOO = _orig
    X = sim.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X, dtype=float)
    b = np.zeros(X.shape[1], dtype=float)
    b[np.asarray(sim.causal_indices, dtype=int)] = np.asarray(sim.causal_effects, dtype=float)
    return X, b


def _fit_rows(path: str, b_by_rep: dict[int, np.ndarray], X_by_rep: dict[int, np.ndarray]) -> list[dict]:
    df = pl.read_parquet(path, columns=["replicate", "single_effects", "credible_sets"])
    out = []
    for r in df.iter_rows(named=True):
        rep = int(r["replicate"])
        b = b_by_rep[rep]
        causal = np.flatnonzero(b != 0.0)
        null = np.flatnonzero(b == 0.0)
        se = r["single_effects"]
        alpha = np.array([e["alpha"] for e in se])          # L x p
        mu = np.array([e["mu"] for e in se])
        sd = np.sqrt(np.array([e["var"] for e in se]))
        lbf = np.array([e["ser_log_bf"] for e in se])
        decl = lbf >= R.MIN_LOG_BF
        pip = 1.0 - np.prod(1.0 - alpha[decl], axis=0) if decl.any() else np.zeros(alpha.shape[1])
        bhat = (alpha * mu).sum(0)
        resid = X_by_rep[rep] @ (bhat - b)
        row = {"replicate": rep,
               "pip_causal": float(pip[causal].mean()) if len(causal) else None,
               "pip_null": float(pip[null].sum()),
               "lp_rmse": float((resid ** 2).mean()),       # mean squared error; sqrt after aggregation
               "mean_ratio": 0.0, "int_cov": 0.0, "sd": 0.0, "n_cond": 0}
        for l, cs in enumerate(r["credible_sets"]):
            if lbf[l] < R.MIN_LOG_BF:
                continue
            hit = [j for j in cs["cs"] if b[j] != 0.0]
            if not hit:
                continue
            j = max(hit, key=lambda k: alpha[l, k])          # the causal this component is about
            row["mean_ratio"] += float(mu[l, j] / b[j])
            row["int_cov"] += float(abs(mu[l, j] - b[j]) <= Z95 * sd[l, j])
            row["sd"] += float(sd[l, j])
            row["n_cond"] += 1
        out.append(row)
    return out


def load_posterior(sc: str) -> pl.DataFrame:
    """One row per fit: cell meta, method, PIP summaries, linear-predictor MSE, and the sums
    over covering declared components of mu/b, interval cover and sd, with their count."""
    cfg = loader.load_config()
    cells: dict[str, dict] = {}
    for coll in loader.collection_method_pairs(cfg, sc).values():
        for bh, mh, mname, mcoord, scoord in coll["pairs"]:
            c = cells.setdefault(bh, {"scoord": scoord, "paths": {}})
            c["paths"].setdefault(R._method_key(mname), (f"{_TG_ROOT}/results/by_batch/{bh}/fits/{mh}/fits.parquet",
                                                         int((mcoord.get("kwargs") or {}).get("L", 1) or 1)))
    frames = []
    for bh, c in cells.items():
        meta = R.cell_meta(c["scoord"])
        paths = {k: v for k, v in c["paths"].items() if os.path.exists(v[0])}
        if meta["null"] or not paths:
            continue
        spec = loader.resolve_simulation_from_coord(c["scoord"])
        reps = sorted({int(r) for p_, _ in paths.values()
                       for r in pl.read_parquet(p_, columns=["replicate"])["replicate"].to_list()})
        X_by_rep, b_by_rep = {}, {}
        for rep in reps:
            X_by_rep[rep], b_by_rep[rep] = truth(spec, rep)
        for method, (path, fit_L) in paths.items():
            rows = _fit_rows(path, b_by_rep, X_by_rep)
            frames.append(pl.DataFrame(rows).with_columns(
                pl.lit(method).alias("method"), pl.lit(bh).alias("batch_hash"), pl.lit(fit_L).alias("fit_L"),
                **{k: (pl.lit(None, dtype=pl.Int64) if v is None else pl.lit(v)) for k, v in meta.items()}))
    return pl.concat(frames, how="diagonal_relaxed")


def metric_frame(P: pl.DataFrame, metrics: list[str], ref: str | None = "cavi",
                 n_boot: int = 2000, seed: int = 0, axis: str = "m",
                 panels: tuple[str, ...] = ("T", "gap")) -> pl.DataFrame:
    """Per (metric, panel keys, axis value, arm): arm vs ref (SPEC[metric]['cmp']) with a joint
    rep-resampled 95% CI, or with ref=None each arm's own value with its own CI. Same columns
    as delta_figure.delta_frame so delta_figure.draw renders it (`axis`/`panels` as there)."""
    rng = np.random.default_rng(seed)
    arms = [m for m in R.METHODS if m != ref and m in P["method"].unique().to_list()]
    rows = []
    keys = list(panels) + [axis]
    for combo in P.select(keys).unique().sort(keys).iter_rows():
        g = P.filter(*[C.eq(c, v) for c, v in zip(keys, combo)]).sort(["batch_hash", "replicate"])
        kv = dict(zip(keys, combo)); T, gap, m = kv.get("T"), kv.get("gap"), combo[-1]
        per = {}
        for mth in arms + ([ref] if ref else []):
            h = g.filter(pl.col("method") == mth)
            n_cond = h["n_cond"].to_numpy().astype(float)
            per[mth] = {k: (h[k].to_numpy().astype(float), n_cond if SPEC[k]["agg"] == "ratio" else None)
                        for k in metrics}
        n_rep = len(per[arms[0]][metrics[0]][0])
        idx = rng.integers(0, n_rep, (n_boot, n_rep))

        def est(a, post, i=None):
            x, n = a
            if n is None:
                v = x.mean() if i is None else x[i].mean(1)
            else:
                v = x.sum() / max(n.sum(), 1) if i is None else x[i].sum(1) / np.maximum(n[i].sum(1), 1)
            return post(v) if post else v

        for metric in metrics:
            sp = SPEC[metric]
            for mth in arms:
                a, ab = est(per[mth][metric], sp["post"]), est(per[mth][metric], sp["post"], idx)
                if ref is None:
                    dlt, bb = a, ab
                else:
                    r0, rb = est(per[ref][metric], sp["post"]), est(per[ref][metric], sp["post"], idx)
                    dlt, bb = (a / r0, ab / rb) if sp["cmp"] == "ratio" else (a - r0, ab - rb)
                lo, hi = np.nanpercentile(bb, [2.5, 97.5])
                rows.append({"metric": metric, "T": T, "gap": gap, "m": m,
                             "theta": (m / float(g["n"][0])) if axis == "m" else float(m),
                             "method": mth, "delta": float(dlt), "lo": float(lo), "hi": float(hi)})
    design = str(P["design"][0]) if "design" in P.columns else "binary"
    return pl.DataFrame(rows).with_columns(pl.lit(axis).alias("axis"), pl.lit(",".join(panels)).alias("panels"),
                                           pl.lit(design).alias("design"))


def draw(df: pl.DataFrame, metrics: list[str], *, ref: str | None = "cavi", **kw):
    rel = ref is not None
    ylabel = (f"PIP metrics, relative to {R.METHOD_LABEL[ref]}" if rel else "PIP metrics") \
        if metrics == PIP_METRICS else \
        (f"effect metrics, relative to {R.METHOD_LABEL[ref]}" if rel else "effect metrics")
    return F.draw(df, ref=ref, metrics=metrics, labels=REL_LABEL if rel else ABS_LABEL,
                  refs=REL_REF if rel else ABS_REF, log_ticks=REL_LOG if rel else ABS_LOG, ylabel=ylabel, **kw)


def main(sc: str = "022-laplace", mode: str = "load") -> None:
    """mode="redraw" reuses figures/posterior_<sc>.parquet from an earlier run (layout tweaks)."""
    out = os.path.join(_HERE, "figures")
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, f"posterior_{sc}.parquet")
    if mode == "redraw":
        P = pl.read_parquet(cache)
    else:
        P = load_posterior(sc)
        P.write_parquet(cache)
    prefix = "" if sc == "022-laplace" else f"{sc.removeprefix('022-laplace-')}_"
    for name, metrics in (("pip", PIP_METRICS), ("effect", EFFECT_METRICS)):
        for ref in ("cavi", None):
            df = metric_frame(P, metrics, ref=ref)
            stem = f"{prefix}{name}_{'vs_cavi' if ref else 'absolute'}"
            df.write_parquet(os.path.join(out, f"{stem}.parquet"))
            fig = draw(df, metrics, ref=ref)
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
            print(os.path.join(out, f"{stem}.png"))


if __name__ == "__main__":
    main(*sys.argv[1:])
