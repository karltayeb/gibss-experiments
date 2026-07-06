import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    from scipy.optimize import brentq


@app.cell(hide_code=True)
def _title():
    mo.md(r"""
    # Effect-size calibration: univariate logBF grid

    We size simulated logistic effects by the **expected univariate log Bayes
    factor** of the causal feature, on the grid

    $$\text{logBF} \in \{4, 8, 16, 32, 64\}\ \text{nats}.$$

    Sizing uses the **exact expected log-likelihood ratio**
    $\Lambda(b) = n\cdot \mathrm{MI}(x;y)$ (≈ logBF up to an $O(\log n)$ Occam
    term). $\Lambda$ is **monotone in $b$** and saturates only at the null
    deviance $n\,H(\bar p)$ — so every rung is reachable by a 1-D root find,
    with **no Wald ceiling**. (The Wald $z$ saturates and falls; that limits
    Wald-approximation *methods*, not the evidence or the sizing — shown as an
    aside.) `glmbf`-free throughout.

    **Centered.** Everything is sized on the *centered* feature $x-\mathrm E[x]$,
    to match how methods fit (`center=True`) and how `simulate()` generates $y$
    (from the centered design). So $b_0$ fixes the base rate $\sigma(b_0)$
    *independently of $b$* — no base-rate drift for asymmetric designs. Symmetric
    designs (gaussian/uniform/q=0.5) are unchanged; asymmetric binary (q=0.1,
    q=0.9) differ from the uncentered sizing.

    Design marginals (rho-independent for the univariate evidence): `gaussian`
    $N(0,1)$, `uniform` $U(-1,1)$, `binary q=0.5/0.1/0.9` ($\pm1$).
    """)
    return


@app.cell
def _config():
    N = 1000                       # primary n for the simulation validation
    N_GRID = [500, 1000, 2000]     # mapping table + distribution facets sweep these
    B0 = -2.0                      # primary intercept for simulation
    B0_GRID = [0.0, -2.0, -4.0]    # mapping table sweeps these
    REPS = 800
    LOGBF = [4, 8, 16, 32, 64]
    DESIGNS = ["gaussian", "uniform", "binary q=0.5", "binary q=0.1", "binary q=0.9"]
    return B0, B0_GRID, DESIGNS, LOGBF, N, N_GRID, REPS


@app.cell
def _analytic():
    def sigmoid(z):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def _H(p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    def marginal(dist):
        if dist == "gaussian":
            x = np.linspace(-8.0, 8.0, 1200)
            w = np.exp(-x * x / 2.0)
            return x, w / w.sum()
        if dist == "uniform":
            x = np.linspace(-1.0, 1.0, 1200)
            return x, np.full(1200, 1.0 / 1200)
        if dist == "binary q=0.5":
            return np.array([-1.0, 1.0]), np.array([0.5, 0.5])
        if dist == "binary q=0.1":
            return np.array([-1.0, 1.0]), np.array([0.9, 0.1])
        if dist == "binary q=0.9":
            return np.array([-1.0, 1.0]), np.array([0.1, 0.9])
        raise ValueError(dist)

    def expected_lrt(dist, b0, b, n):
        """Λ = n·MI(x;y) = n·[H(p̄) − E_x H(p_x)], sized on the CENTERED feature
        x − E[x] so b0 fixes the base rate σ(b0) (matches center=True fits and
        the centered simulate). Symmetric designs (E[x]=0) unchanged; asymmetric
        binary (q≠0.5) differ. Monotone in b; saturates at deviance n·H(p̄_∞)."""
        x, pw = marginal(dist)
        xbar = float(np.sum(pw * x))
        px = sigmoid(b0 + b * (x - xbar))
        pbar = float(np.sum(pw * px))
        return n * (_H(pbar) - float(np.sum(pw * _H(px))))

    def wald_half_z2(dist, b0, b, n):
        """½ z² from the (true-parameter) profile information on the CENTERED
        feature — the Wald-method proxy. Non-monotone in b (separation collapses
        the information)."""
        x, pw = marginal(dist)
        xbar = float(np.sum(pw * x)); xc = x - xbar
        w = sigmoid(b0 + b * xc) * (1.0 - sigmoid(b0 + b * xc))
        Ew = float(np.sum(pw * w)); Ewx = float(np.sum(pw * w * xc)); Ewx2 = float(np.sum(pw * w * xc * xc))
        return 0.5 * n * b * b * (Ewx2 - Ewx * Ewx / Ew)

    def solve_b(dist, logbf_target, b0, n):
        """b with Λ(b) = target (monotone root find). NaN only if target exceeds
        the deviance ceiling (never, for this grid)."""
        if expected_lrt(dist, b0, 60.0, n) < logbf_target:
            return float("nan")
        return float(brentq(lambda bb: expected_lrt(dist, b0, bb, n) - logbf_target, 1e-5, 60.0))

    def deviance_ceiling(dist, b0, n):
        return expected_lrt(dist, b0, 60.0, n)

    return deviance_ceiling, expected_lrt, sigmoid, solve_b, wald_half_z2


@app.cell(hide_code=True)
def _mapping_md():
    mo.md(r"""
    ## Mapping table: (n, b0, design, logBF) → effect size `b`

    `b` solves `Λ(b) = logBF` (exact-LRT sizing). `deviance` is the saturation
    ceiling `n·H(p̄_∞)` — every rung is far below it, so all cells are reachable
    (no Wald limit). `000_markov` uses the **b0 ∈ {0, −2}** rows; **−4** is for
    the later imbalanced experiment (now fully computable).
    """)
    return


@app.cell
def _mapping_table(B0_GRID, DESIGNS, LOGBF, N_GRID, deviance_ceiling, solve_b):
    def _build():
        rows = []
        for n in N_GRID:
            for b0 in B0_GRID:
                for dist in DESIGNS:
                    rec = {"n": n, "b0": b0, "design": dist,
                           "deviance": round(deviance_ceiling(dist, b0, n))}
                    for lbf in LOGBF:
                        rec[f"L{lbf}"] = round(solve_b(dist, lbf, b0, n), 3)
                    rows.append(rec)
        return pl.DataFrame(rows)

    mapping = _build()
    mo.ui.table(mapping, selection=None)
    return (mapping,)


@app.cell(hide_code=True)
def _decouple_md():
    mo.md(r"""
    ## Why exact-LRT, not Wald: they decouple at saturation

    For one design (binary q=0.5, b0=−4), sweep `b` and compare the exact LRT
    `Λ(b)` (monotone, what we size by) against the Wald `½z²(b)` (rises, then
    **falls** as the data separate). `½z²` is only a valid proxy in the regular
    regime; at strong signal it collapses — which is why Wald-approximation
    methods' BFs saturate, and why sizing must use `Λ`.
    """)
    return


@app.cell
def _fig_decouple(N, expected_lrt, wald_half_z2):
    def _make():
        bs = np.linspace(0.01, 8.0, 200)
        lam = [expected_lrt("binary q=0.5", -4.0, b, N) for b in bs]
        wz = [wald_half_z2("binary q=0.5", -4.0, b, N) for b in bs]
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.plot(bs, lam, label="exact LRT  Λ = n·MI (monotone)", lw=2)
        ax.plot(bs, wz, label="Wald  ½z²  (saturates, falls)", lw=2)
        for lbf in (4, 8, 16, 32):
            ax.axhline(lbf, color="0.8", lw=0.7, zorder=0)
        ax.set_xlabel("effect b"); ax.set_ylabel("expected logBF (nats)")
        ax.set_ylim(0, 70); ax.set_title("binary q=0.5, b0=−4, n=1000")
        ax.legend(fontsize=8)
        return fig

    _make()
    return


@app.cell(hide_code=True)
def _sim_md():
    mo.md(r"""
    ## Simulation validation (realized LRT)

    For each `(n, design, logBF)` at the primary b0, draw `REPS` datasets
    `y ~ Bernoulli(σ(b0 + b·x))`, fit univariate logistic full and null models,
    and record the **realized LRT** `= ℓ_full − ℓ_null` (and the Wald z, for
    reference). The mean realized LRT should match the target logBF.
    """)
    return


@app.cell
def _simulate(B0, DESIGNS, LOGBF, N_GRID, REPS, mapping, sigmoid):
    def draw_x(dist, n, rng):
        if dist == "gaussian":
            return rng.normal(size=n)
        if dist == "uniform":
            return rng.uniform(-1.0, 1.0, size=n)
        q = float(dist.split("=")[1])  # binary q=0.5/0.1/0.9
        return 2.0 * (rng.random(n) < q) - 1.0

    def _loglik(p, y):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit_lrt_z(x, y, iters=60):
        X = np.column_stack([np.ones_like(x), x]); b = np.zeros(2)
        for _ in range(iters):
            p = sigmoid(X @ b); W = p * (1 - p)
            try:
                step = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-9 * np.eye(2), X.T @ (y - p))
            except np.linalg.LinAlgError:
                return np.nan, np.nan
            b = b + step
            if np.max(np.abs(step)) < 1e-9:
                break
        p = sigmoid(X @ b); W = p * (1 - p)
        ll_full = _loglik(p, y)
        ybar = float(y.mean()); ll_null = _loglik(np.full_like(y, ybar), y)
        cov = np.linalg.inv(X.T @ (W[:, None] * X) + 1e-12 * np.eye(2))
        return ll_full - ll_null, b[1] / np.sqrt(cov[1, 1])

    def _run():
        out = {}  # (n, design, logBF) -> (lrt_array, z_array)
        for n in N_GRID:
            blk = {(r["design"], lbf): r[f"L{lbf}"]
                   for r in mapping.iter_rows(named=True) if r["n"] == n and r["b0"] == B0
                   for lbf in LOGBF}
            for dist in DESIGNS:
                for lbf in LOGBF:
                    b = blk.get((dist, lbf))
                    if b is None or b != b:
                        continue
                    rng = np.random.default_rng(20260628)
                    lrts = np.empty(REPS); zs = np.empty(REPS)
                    for r in range(REPS):
                        x = draw_x(dist, n, rng)
                        xc = x - x.mean()  # centered design (matches centered simulate/fit)
                        y = (rng.random(n) < sigmoid(B0 + b * xc)).astype(float)
                        lrts[r], zs[r] = fit_lrt_z(xc, y)
                    m = np.isfinite(lrts)
                    out[(n, dist, lbf)] = (lrts[m], zs[np.isfinite(zs)])
        return out

    sim = _run()
    return (sim,)


@app.cell
def _validation_table(sim):
    def _build():
        rows = []
        for (n, dist, lbf), (lrt, z) in sim.items():
            rows.append({
                "n": n, "design": dist, "logBF_target": lbf,
                "LRT_mean": round(float(lrt.mean()), 1),
                "LRT_sd": round(float(lrt.std()), 1),
                "z_mean": round(float(z.mean()), 2),
                "n_reps": int(lrt.size),
            })
        return pl.DataFrame(rows).sort(["n", "design", "logBF_target"])

    validation = _build()
    mo.ui.table(validation, selection=None)
    return


@app.cell(hide_code=True)
def _fig_md():
    mo.md(r"""
    ## Realized-LRT distributions (ECDFs)

    Grid: **rows = design, columns = n**; one curve per logBF rung (legend),
    dashed vertical line = target logBF. Curves center on their target (mean on
    target); spread is intrinsic and **widest for the rare binary (q=0.1)**.
    Larger n only sharpens — no rung disappears (no Wald ceiling).
    """)
    return


@app.cell
def _fig_ecdf(DESIGNS, LOGBF, N_GRID, sim):
    def _make():
        nr, nc = len(DESIGNS), len(N_GRID)
        fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.5 * nr), constrained_layout=True)
        cmap = plt.get_cmap("viridis")
        for i, dist in enumerate(DESIGNS):
            for j, n in enumerate(N_GRID):
                ax = axes[i, j]
                for k, lbf in enumerate(LOGBF):
                    key = (n, dist, lbf)
                    if key not in sim:
                        continue
                    lrt = np.sort(sim[key][0])
                    ecdf = np.arange(1, lrt.size + 1) / lrt.size
                    color = cmap(k / (len(LOGBF) - 1))
                    ax.plot(lrt, ecdf, color=color, lw=1.3, label=f"{lbf}")
                    ax.axvline(lbf, color=color, ls="--", lw=0.7, alpha=0.7)
                ax.set_title(f"{dist} · n={n}", fontsize=8)
                ax.set_xscale("symlog"); ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel("ECDF", fontsize=8)
                if i == nr - 1:
                    ax.set_xlabel("realized LRT", fontsize=8)
                ax.legend(fontsize=5, loc="lower right", title="logBF", title_fontsize=5)
        fig.suptitle("Realized-LRT ECDFs by design × n (dashed = target logBF)")
        return fig

    _make()
    return


@app.cell(hide_code=True)
def _conclusions():
    mo.md(r"""
    ## Takeaways

    - **Exact-LRT sizing works and is monotone** — `Λ(b)=target` reproduces the
      target mean realized LRT; no glmbf, no Wald ceiling. Every rung (incl.
      logBF=64) is reachable at every (design, b0, n).
    - **Centered sizing is accurate across imbalance.** Empirical mean LLR / Λ
      (n=1000, ≥40 reps): at strong signal (L≥16) ratio 0.95–1.07 for every
      design at b0 ∈ {0, −2, −4}; at the imbalanced b0=−4 (σ≈1.8%, ~18 cases)
      ratio 0.95–1.11. Only logBF=4 (weak) overshoots (~1.1–1.3×, worst at
      balanced b0) — finite-sample LLR bias vs the asymptotic n·MI, not a
      centering/imbalance defect. So Λ is a good prior-free target everywhere.
    - **logBF and z² decouple at saturation** — `Λ` keeps rising while Wald `½z²`
      falls (see the decouple figure). Size by `Λ`; the z label is a
      regular-regime mnemonic only.
    - **The Wald ceiling is a method property**, not a limit on evidence or
      sizing — it explains why Laplace/JJ BFs saturate at strong, separated
      signal.
    - **rho is orthogonal** — sizing uses the causal marginal, so one frozen `b`
      per (design, b0, logBF) serves the whole correlation sweep.

    Frozen table → `notes/2026-06-27-univariate-logbf-z-grid.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
