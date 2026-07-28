import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scipy.special import logsumexp


@app.cell(hide_code=True)
def title_cell():
    mo.md("""
    # Scale Mixture EM: Coarse-to-Fine Grid

    Fits `g = sum_k pi_k * N(0, sigma_k^2)` by EM on a log-spaced sigma grid.

    Strategy: solve on grid of size 2^4, warm-start the 2^5 solve, and so on up to 2^8.
    Warm init interpolates the previous solution with a small uniform component.
    Compare total EM iterations vs cold-start (uniform init) at each grid size.
    """)
    return


@app.cell
def data_controls_cell():
    pi0_true_sl = mo.ui.slider(
        0.5,
        0.99,
        step=0.01,
        value=0.85,
        label="True pi0 (null fraction)",
        show_value=True,
    )
    sigma_true_sl = mo.ui.slider(
        0.1,
        6.0,
        step=0.1,
        value=2.0,
        label="True sigma (non-null)",
        show_value=True,
    )
    n_obs_sl = mo.ui.slider(
        100, 5000, step=100, value=1000, label="n obs", show_value=True
    )
    seed_nb = mo.ui.number(start=0, stop=999, step=1, value=0, label="seed")
    mo.hstack([pi0_true_sl, sigma_true_sl, n_obs_sl, seed_nb], gap=2)
    return n_obs_sl, pi0_true_sl, seed_nb, sigma_true_sl


@app.cell
def data_cell(n_obs_sl, pi0_true_sl, seed_nb, sigma_true_sl):
    pi0_true = pi0_true_sl.value
    sigma_true = sigma_true_sl.value
    n_obs = int(n_obs_sl.value)
    seed = int(seed_nb.value)

    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 1.0 - pi0_true, size=n_obs)  # z=1 → non-null
    theta = np.where(z == 1, rng.normal(0.0, sigma_true, size=n_obs), 0.0)
    x = theta + rng.normal(0.0, 1.0, size=n_obs)  # se = 1
    return pi0_true, sigma_true, x


@app.cell
def grid_precompute_cell(x):
    n_full = 256  # 2^8
    sigma_max = 8.0
    sigma_grid = np.logspace(-2, np.log10(sigma_max), n_full)

    # L[i, k] = log N(x_i ; 0, sigma_k^2 + 1)  shape: (n_obs, n_full)
    L_full = norm.logpdf(x[:, None], 0.0, np.sqrt(sigma_grid[None, :] ** 2 + 1.0))
    return L_full, n_full, sigma_grid


@app.function
def solve_pi(L, pi_init, max_iter=2000, tol=1e-8):
    """
    EM for scale mixture.
    L : (n, K)  log-likelihoods  log f_k(x_i)
    Returns (pi, n_iter, loglik_history)
    """
    n, K = L.shape
    pi = np.asarray(pi_init, dtype=float)
    pi = pi / pi.sum()
    log_n = np.log(n)
    loglik_history = []
    for it in range(1, max_iter + 1):
        log_joint = L + np.log(pi)[None, :]  # (n, K)
        log_norm = logsumexp(log_joint, axis=1)  # (n,)
        loglik = float(log_norm.sum())
        loglik_history.append(loglik)
        if it > 1 and abs(loglik - loglik_history[-2]) < tol:
            return pi, it, loglik_history
        log_r = log_joint - log_norm[:, None]  # (n, K)
        pi = np.exp(logsumexp(log_r, axis=0) - log_n)
        pi = pi / pi.sum()
    return pi, max_iter, loglik_history


@app.cell
def solver_controls_cell():
    tol_sl = mo.ui.slider(
        -12, -2, step=1, value=-8, label="log10(tol)", show_value=True
    )
    max_iter_sl = mo.ui.slider(
        100, 5000, step=100, value=2000, label="max iter", show_value=True
    )
    alpha_sl = mo.ui.slider(
        0.0,
        0.2,
        step=0.005,
        value=0.01,
        label="uniform mix alpha",
        show_value=True,
    )
    mo.hstack([tol_sl, max_iter_sl, alpha_sl], gap=2)
    return alpha_sl, max_iter_sl, tol_sl


@app.cell
def comparison_cell(L_full, alpha_sl, max_iter_sl, n_full, tol_sl):
    tol = 10.0**tol_sl.value
    max_iter = int(max_iter_sl.value)
    alpha = float(alpha_sl.value)

    stages = [2**s for s in range(2, 7)]  # 16, 32, 64, 128, 256


    def _subgrid(K):
        stride = n_full // K
        return np.arange(0, n_full, stride)[:K]


    def _expand_pi(pi_prev, K_new):
        """Double the grid: split each weight evenly to its two children."""
        pi_exp = np.empty(K_new)
        pi_exp[0::2] = pi_prev / 2.0
        pi_exp[1::2] = pi_prev / 2.0
        return pi_exp


    # Cold start: uniform init at every stage
    cold = []
    for _K in stages:
        _L = L_full[:, _subgrid(_K)]
        _pi, _nit, _hist = solve_pi(
            _L, np.ones(_K) / _K, max_iter=max_iter, tol=tol
        )
        cold.append({"K": _K, "n_iter": _nit, "loglik": _hist[-1], "pi": _pi})

    # Coarse-to-fine: warm start from previous stage
    c2f = []
    _pi_prev = None
    for _K in stages:
        _L = L_full[:, _subgrid(_K)]
        if _pi_prev is None:
            _pi_init = np.ones(_K) / _K
        else:
            _pi_exp = _expand_pi(_pi_prev, _K)
            _pi_init = (1.0 - alpha) * _pi_exp + alpha / _K
            _pi_init /= _pi_init.sum()
        _pi, _nit, _hist = solve_pi(_L, _pi_init, max_iter=max_iter, tol=tol)
        _pi_prev = _pi
        c2f.append({"K": _K, "n_iter": _nit, "loglik": _hist[-1], "pi": _pi})
    return c2f, cold, stages


@app.cell
def plot_cell(c2f, cold, pi0_true, sigma_grid, sigma_true, stages):
    _ks = [r["K"] for r in cold]
    _cold_iters = [r["n_iter"] for r in cold]
    _c2f_iters = [r["n_iter"] for r in c2f]
    _cold_ll = [r["loglik"] for r in cold]
    _c2f_ll = [r["loglik"] for r in c2f]

    _fig, _axes = plt.subplots(1, 4, figsize=(18, 4))

    # Panel 1: iterations to convergence
    _ax = _axes[0]
    _x = np.arange(len(_ks))
    _w = 0.35
    _ax.bar(
        _x - _w / 2,
        _cold_iters,
        _w,
        label="cold start",
        color="steelblue",
        alpha=0.8,
    )
    _ax.bar(
        _x + _w / 2,
        _c2f_iters,
        _w,
        label="coarse-to-fine",
        color="tomato",
        alpha=0.8,
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"2^{int(np.log2(k))}" for k in _ks])
    _ax.set_xlabel("grid size K")
    _ax.set_ylabel("EM iterations")
    _ax.set_title("Iterations to convergence")
    _ax.legend(fontsize=8)

    # Panel 2: final log-likelihood (both should match)
    _ax = _axes[1]
    _ax.plot(_ks, _cold_ll, "o-", color="steelblue", label="cold start")
    _ax.plot(_ks, _c2f_ll, "s--", color="tomato", label="coarse-to-fine")
    _ax.set_xlabel("grid size K")
    _ax.set_ylabel("log-likelihood")
    _ax.set_title("Final log-likelihood")
    _ax.set_xscale("log", base=2)
    _ax.legend(fontsize=8)

    # Panel 3: estimated mixture density at finest grid
    _ax = _axes[2]
    _n_full = len(sigma_grid)
    _stride_last = _n_full // stages[-1]
    _idx_last = np.arange(0, _n_full, _stride_last)[: stages[-1]]
    _sg = sigma_grid[_idx_last]
    _ax.step(
        _sg,
        cold[-1]["pi"],
        where="mid",
        color="steelblue",
        alpha=0.8,
        label="cold start",
    )
    _ax.step(
        _sg,
        c2f[-1]["pi"],
        where="mid",
        color="tomato",
        alpha=0.8,
        linestyle="--",
        label="coarse-to-fine",
    )
    _ax.axvline(
        sigma_true,
        color="black",
        linewidth=1.2,
        linestyle=":",
        label=f"true sigma={sigma_true}",
    )
    _ax.set_xscale("log")
    _ax.set_xlabel("sigma")
    _ax.set_ylabel("pi_k")
    _ax.set_title("Estimated mixture weights (K=256)")
    _ax.legend(fontsize=8)

    # Panel 4: implied marginal density
    _ax = _axes[3]
    _x_eval = np.linspace(-6.0, 6.0, 400)
    _n_full = len(sigma_grid)
    _stride_last = _n_full // stages[-1]
    _idx_last = np.arange(0, _n_full, _stride_last)[: stages[-1]]
    _sg = sigma_grid[_idx_last]

    for _label, _res, _color, _ls in [
        ("cold start", cold[-1], "steelblue", "-"),
        ("coarse-to-fine", c2f[-1], "tomato", "--"),
    ]:
        # marginal: sum_k pi_k * N(x; 0, sigma_k^2 + 1)
        _marg_sd = np.sqrt(_sg[:, None] ** 2 + 1.0)  # (K, 1)
        _log_comp = norm.logpdf(_x_eval[None, :], 0.0, _marg_sd)  # (K, n_x)
        _log_pi = np.log(np.maximum(_res["pi"][:, None], 1e-300))
        _f_hat = np.exp(logsumexp(_log_pi + _log_comp, axis=0))
        _ax.plot(
            _x_eval, _f_hat, color=_color, linestyle=_ls, alpha=0.85, label=_label
        )

    # true marginal
    _f_true = pi0_true * norm.pdf(_x_eval, 0.0, 1.0) + (1.0 - pi0_true) * norm.pdf(
        _x_eval, 0.0, np.sqrt(sigma_true**2 + 1.0)
    )
    _ax.plot(
        _x_eval, _f_true, color="black", linewidth=1.2, linestyle=":", label="true"
    )
    _ax.set_xlabel("x")
    _ax.set_ylabel("density")
    _ax.set_title("Implied marginal density (K=256)")
    _ax.legend(fontsize=8)

    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
