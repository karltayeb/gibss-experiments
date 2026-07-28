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
    # Two-Group Log-Likelihood Explorer

    Model: `thetahat | theta ~ N(theta, 1)`, `theta ~ g = pi0*delta(0) + (1-pi0)*N(mu, var)`

    Each facet fixes `pi0` and shows the log-likelihood surface over `(mu, var)`.
    The shared color scale lets you compare likelihood mass across facets.
    The truth marker (★) appears in the facet matching the true `pi0`.
    """)
    return


@app.cell
def grid_cell():
    pi0_values = np.round(np.linspace(0.10, 0.90, 9), 2)
    mu_grid = np.linspace(-4.0, 4.0, 60)
    var_grid = np.linspace(0.1, 8.0, 60)
    return mu_grid, pi0_values, var_grid


@app.cell
def controls_cell(mu_grid, pi0_values, var_grid):
    _mu_step = float(mu_grid[1] - mu_grid[0])
    _var_step = float(var_grid[1] - var_grid[0])

    pi0_true_dd = mo.ui.dropdown(
        options={str(v): v for v in pi0_values},
        value=str(pi0_values[2]),
        label="True pi0",
    )
    mu_true_sl = mo.ui.slider(
        start=float(mu_grid[0]),
        stop=float(mu_grid[-1]),
        step=_mu_step,
        value=2.0,
        label="True mu",
        show_value=True,
    )
    var_true_sl = mo.ui.slider(
        start=float(var_grid[0]),
        stop=float(var_grid[-1]),
        step=_var_step,
        value=2.0,
        label="True var",
        show_value=True,
    )
    n_obs_sl = mo.ui.slider(
        start=100, stop=2000, step=100, value=500, label="n obs", show_value=True
    )
    seed_nb = mo.ui.number(start=0, stop=999, step=1, value=0, label="seed")

    mo.hstack([pi0_true_dd, mu_true_sl, var_true_sl, n_obs_sl, seed_nb], gap=2)
    return mu_true_sl, n_obs_sl, pi0_true_dd, seed_nb, var_true_sl


@app.cell
def true_params_cell(mu_true_sl, n_obs_sl, pi0_true_dd, seed_nb, var_true_sl):
    pi0_true = float(pi0_true_dd.value)
    mu_true = float(mu_true_sl.value)
    var_true = float(var_true_sl.value)
    n_obs = int(n_obs_sl.value)
    seed = int(seed_nb.value)
    return mu_true, n_obs, pi0_true, seed, var_true


@app.cell
def data_cell(mu_true, n_obs, pi0_true, seed, var_true):
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 1.0 - pi0_true, size=n_obs)
    theta_nonnull = rng.normal(mu_true, np.sqrt(var_true), size=n_obs)
    theta = np.where(z == 1, theta_nonnull, 0.0)
    x = theta + rng.normal(0.0, 1.0, size=n_obs)
    return (x,)


@app.cell
def loglik_cell(mu_grid, pi0_values, var_grid, x):
    # (n_obs,) log density under null
    _log_f0 = norm.logpdf(x, 0.0, 1.0)

    # (n_obs, n_mu, n_var) log density under non-null for each (mu, var)
    _log_f1 = norm.logpdf(
        x[:, None, None],
        mu_grid[None, :, None],
        np.sqrt(var_grid[None, None, :] + 1.0),
    )

    loglik = np.empty((len(pi0_values), len(mu_grid), len(var_grid)))
    for _j, _pi0 in enumerate(pi0_values):
        _lp0 = np.log(_pi0) + _log_f0[:, None, None]
        _lp1 = np.log(1.0 - _pi0) + _log_f1
        loglik[_j] = logsumexp(
            np.stack([np.broadcast_to(_lp0, _lp1.shape), _lp1], axis=-1), axis=-1
        ).sum(axis=0)
    return (loglik,)


@app.cell
def plot_cell(
    loglik,
    mu_grid,
    mu_true,
    pi0_true,
    pi0_values,
    var_grid,
    var_true,
):
    import matplotlib.gridspec as gridspec

    _vmin = float(loglik.min())
    _vmax = float(loglik.max())
    _levels = np.linspace(_vmin, _vmax, 40)
    _n_cols = 3
    _n_rows = int(np.ceil(len(pi0_values) / _n_cols))
    _cell = 3.2  # inches per square facet

    _fig = plt.figure(figsize=(_n_cols * _cell + 0.8, _n_rows * _cell + 0.4))
    _gs = gridspec.GridSpec(
        _n_rows, _n_cols + 1,
        figure=_fig,
        width_ratios=[*([1] * _n_cols), 0.06],
        hspace=0.35, wspace=0.25,
    )
    _axes_flat = [_fig.add_subplot(_gs[_r, _c]) for _r in range(_n_rows) for _c in range(_n_cols)]
    _cax = _fig.add_subplot(_gs[:, _n_cols])

    _cf = None
    for _j, (_ax, _pi0) in enumerate(zip(_axes_flat, pi0_values)):
        _cf = _ax.contourf(
            mu_grid, var_grid, loglik[_j].T,
            levels=_levels, vmin=_vmin, vmax=_vmax, cmap="viridis",
        )
        _ax.contour(
            mu_grid, var_grid, loglik[_j].T,
            levels=_levels[::5], colors="white", linewidths=0.5, alpha=0.4,
        )
        _ax.set_box_aspect(1)
        _ax.set_title(f"pi0 = {_pi0:.2f}", fontsize=9)
        _ax.set_xlabel("mu", fontsize=8)
        _ax.set_ylabel("var", fontsize=8)
        _ax.tick_params(labelsize=7)

        if np.isclose(_pi0, pi0_true):
            _ax.scatter(
                [mu_true], [var_true],
                color="red", marker="*", s=250, zorder=5,
                edgecolors="white", linewidths=0.5,
            )

    for _ax in _axes_flat[len(pi0_values):]:
        _ax.axis("off")

    _fig.colorbar(_cf, cax=_cax, label="log-likelihood")
    plt.suptitle("Log-Likelihood Surface (fixed pi0 per facet)", fontsize=11)
    _fig
    return


if __name__ == "__main__":
    app.run()
