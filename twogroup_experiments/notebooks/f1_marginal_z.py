import sys

if sys.path:
    sys.path.pop(0)

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np


@app.cell(hide_code=True)
def title_cell(mo):
    mo.md(
        """
        # Marginal z-score Explorer

        Explore the implied marginal null and non-null z-score distributions for
        a normal non-null prior `f1 = N(mu0, sigma0)`.
        """
    )
    return


@app.cell
def parameter_state_cell(mo):
    mu0, set_mu0 = mo.state(1.5)
    sigma0, set_sigma0 = mo.state(0.1)
    se, set_se = mo.state(1.0)
    t_max, set_t_max = mo.state(5.0)
    n_grid, set_n_grid = mo.state(400)
    return mu0, n_grid, se, set_mu0, set_n_grid, set_se, set_sigma0, set_t_max, sigma0, t_max


@app.cell(hide_code=True)
def controls_cell(
    mo,
    mu0,
    n_grid,
    se,
    set_mu0,
    set_n_grid,
    set_se,
    set_sigma0,
    set_t_max,
    sigma0,
    t_max,
):
    mu0_input = mo.ui.number(
        value=float(mu0()),
        start=-5.0,
        stop=5.0,
        step=0.1,
        label="mu0",
        on_change=lambda value: set_mu0(float(value)),
    )
    sigma0_input = mo.ui.number(
        value=float(sigma0()),
        start=0.0,
        stop=5.0,
        step=0.01,
        label="sigma0",
        on_change=lambda value: set_sigma0(float(value)),
    )
    se_input = mo.ui.number(
        value=float(se()),
        start=0.1,
        stop=5.0,
        step=0.1,
        label="observation se",
        on_change=lambda value: set_se(float(value)),
    )
    t_max_input = mo.ui.number(
        value=float(t_max()),
        start=1.0,
        stop=10.0,
        step=0.5,
        label="max t",
        on_change=lambda value: set_t_max(float(value)),
    )
    n_grid_input = mo.ui.number(
        value=int(n_grid()),
        start=100,
        stop=2000,
        step=50,
        label="grid points",
        on_change=lambda value: set_n_grid(int(value)),
    )

    sigma_preset_btn = mo.ui.button(
        label="Set sigma0 = 0.01",
        on_click=lambda _: set_sigma0(0.01),
    )
    mu_preset_btn = mo.ui.button(
        label="Set mu0 = 0",
        on_click=lambda _: set_mu0(0.0),
    )

    mo.vstack(
        [
            mo.hstack([mu0_input, sigma0_input, se_input]),
            mo.hstack([t_max_input, n_grid_input]),
            mo.hstack([sigma_preset_btn, mu_preset_btn]),
        ]
    )
    return


@app.cell
def distribution_helpers_cell(math, np):
    def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
        centered = (x - mean) / sd
        return np.exp(-0.5 * centered**2) / (sd * np.sqrt(2.0 * np.pi))

    def normal_cdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
        standardized = (x - mean) / (sd * np.sqrt(2.0))
        erf_vec = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_vec(standardized))

    return normal_cdf, normal_pdf


@app.cell
def summary_cell(mu0, np, se, sigma0):
    null_sd = float(se())
    nonnull_sd = float(np.sqrt(se() ** 2 + sigma0() ** 2))
    return nonnull_sd, null_sd


@app.cell(hide_code=True)
def summary_display_cell(mo, mu0, nonnull_sd, null_sd, se, sigma0):
    mo.md(
        f"""
        **Current `f1`:** `N({mu0():.2f}, {sigma0():.2f})`  
        **Null marginal:** `Z | null ~ N(0, {null_sd:.3f}^2)`  
        **Non-null marginal:** `Z | non-null ~ N({mu0():.2f}, {nonnull_sd:.3f}^2)`  
        **Observation noise:** `se = {se():.2f}`
        """
    )
    return


@app.cell
def density_data_cell(mu0, n_grid, nonnull_sd, np, t_max):
    x_max = max(6.0, abs(mu0()) + 4.0 * nonnull_sd, float(t_max()) + 1.0)
    x = np.linspace(-x_max, x_max, int(n_grid()))
    return x, x_max


@app.cell
def density_plot_cell(mu0, normal_pdf, null_sd, plt, x, nonnull_sd):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.plot(x, normal_pdf(x, 0.0, null_sd), label="Null", linewidth=2.5)
    _ax.plot(x, normal_pdf(x, mu0(), nonnull_sd), label="Non-null", linewidth=2.5)
    _ax.set_title("Overlapping marginal z-score densities")
    _ax.set_xlabel("z")
    _ax.set_ylabel("Density")
    _ax.legend()
    _ax.grid(alpha=0.25)
    _fig.tight_layout()
    _fig
    return


@app.cell
def tail_data_cell(mu0, n_grid, normal_cdf, np, null_sd, t_max, nonnull_sd):
    t = np.linspace(0.0, float(t_max()), int(n_grid()))
    null_tail = 2.0 * (1.0 - normal_cdf(t, 0.0, null_sd))
    nonnull_tail = 1.0 - normal_cdf(t, mu0(), nonnull_sd) + normal_cdf(-t, mu0(), nonnull_sd)
    return nonnull_tail, null_tail, t


@app.cell
def tail_plot_cell(nonnull_tail, null_tail, plt, t):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.plot(t, null_tail, label="Null", linewidth=2.5)
    _ax.plot(t, nonnull_tail, label="Non-null", linewidth=2.5)
    _ax.set_title(r"$P(|Z| > t)$ as a function of $t$")
    _ax.set_xlabel("t")
    _ax.set_ylabel(r"$P(|Z| > t)$")
    _ax.set_ylim(0.0, 1.0)
    _ax.legend()
    _ax.grid(alpha=0.25)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def tail_summary_cell(mo, nonnull_tail, null_tail, np, t):
    t_ref = 2.0
    idx = int(np.argmin(np.abs(t - t_ref)))
    mo.md(
        f"""
        At `t = {t[idx]:.2f}`:

        - `P(|Z| > t | null) = {null_tail[idx]:.3f}`
        - `P(|Z| > t | non-null) = {nonnull_tail[idx]:.3f}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
