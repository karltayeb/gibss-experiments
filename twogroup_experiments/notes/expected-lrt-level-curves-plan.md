# Plan: expected-LRT detectability level curves (gene-set size vs. per-gene signal vs. enrichment)

**Status:** ready to implement. This is a self-contained handoff for a fresh agent.
**Owner of this note:** written 2026-07-21 as a design handoff; no implementation done yet.

---

## 1. The idea / what we want to communicate

For gene-set enrichment under the two-group model there are three knobs we control
(location scenario): the **gene-set size** $m$, the **enrichment log-odds** $b$, and
the **per-gene location** $\mu$ of the effect distribution $f_1$. The background
log-odds is fixed at $b_0=-2$.

Define

$$ f(m, b, \mu) \;=\; \mathbb{E}\big[\text{LRT statistic for enrichment of the causal set}\big], $$

the *expected* likelihood-ratio-test statistic for detecting that the set is
enriched. We want to **plot the $(b,\mu)$ level curves of $f$ for small, medium, and
large gene sets**. The message: a larger set reaches the same detectability
(same iso-LRT curve) at a **smaller** $b$ and $\mu$ - i.e. large sets can be
detected even when the per-gene effect ($f_1$ vs $f_0$ gap) is small.

Then repeat the exercise for the **scale scenario**: $f_1 = \mathcal N(0,\sigma^2)$
(variance inflation, no location shift), giving $f(m, b, \sigma)$, and plot its
$(b,\sigma)$ level curves for small/medium/large sets.

This supersedes an earlier abandoned attempt (a full-pipeline "cost of discretizing
at $\mathbb E[z^2]=3$" second run) - see §7. That attempt aimed the per-gene signal
the wrong way (it weakened *small* sets, which are the ones that need a *strong*
per-gene signal). This landscape plot is the right way to show the size/signal
tradeoff, and it is far lighter: no simulation of the C2 design, no SuSiE fits, no
Snakemake. It is pure NumPy/SciPy.

---

## 2. The model and the statistic (precise)

Two-group model, per gene $i$, with a single gene-set indicator $x_i\in\{0,1\}$
(1 iff gene $i$ is in the causal set $S$, $|S|=m$):

- Latent membership $\gamma_i \sim \mathrm{Bernoulli}(\pi_i)$, with
  $\pi_i = \sigma(b_0 + b\,x_i)$, $\sigma(t)=1/(1+e^{-t})$, $b_0=-2$ fixed.
- Effect $\theta_i \mid \gamma_i{=}0 \sim \delta_0$;
  $\theta_i \mid \gamma_i{=}1 \sim f_1$.
  - **loc scenario:** $f_1 = \mathcal N(\mu, \tau_0^2)$ with $\tau_0 = 0.1$
    (matches the `loc_*` signals in `experiments/library.yaml`).
  - **scale scenario:** $f_1 = \mathcal N(0, \sigma^2)$ (matches `scale_*`).
- Observed z-score $z_i = \theta_i + \varepsilon_i$, $\varepsilon_i\sim\mathcal N(0,1)$.

So the **observed-data mixture components** (which the oracle knows) are:

- $g_0(z) = \mathcal N(z;\,0,\,1)$  (a null gene's z).
- **loc:** $g_1(z) = \mathcal N(z;\,\mu,\,\tau_0^2 + 1)$.
- **scale:** $g_1(z) = \mathcal N(z;\,0,\,\sigma^2 + 1)$.

The per-gene marginal likelihood at log-odds $\eta$ is
$\ell_i(\eta) = \log\!\big[(1-\sigma(\eta))\,g_0(z_i) + \sigma(\eta)\,g_1(z_i)\big]$.

**LRT statistic** (test $H_0: b=0$ vs $H_1: b$ free, with $b_0=-2$ fixed in both):

$$ \mathrm{LRT}(z_{1:m}) = 2\Big[\max_{b}\ \sum_{i} \ell_i(b_0+b\,x_i)\ -\ \sum_{i}\ell_i(b_0)\Big]. $$

$f(m,b_{\text{true}},\mu)$ (or $\sigma$) is the **Monte-Carlo mean** of this LRT over
replicate draws of $z_{1:m}$ generated at the *true* parameters.

### 2.1 Key simplification: only the in-set genes matter

Because $b_0$ is **fixed** (not profiled), every out-of-set gene has $x_i=0$, so its
term $\ell_i(b_0)$ is identical under $H_0$ and $H_1$ and cancels in the LRT. The
statistic therefore depends **only on the $m$ genes in the set**. Consequences:

- No design matrix, no total gene count $n$, no competitor gene sets. Just draw $m$
  z-scores for the in-set genes and optimize a scalar $b$.
- Data generation for one replicate: draw $\gamma_i \sim \mathrm{Bernoulli}(\sigma(b_0+b_{\text{true}}))$
  for $i=1..m$; $z_i \sim g_0$ if $\gamma_i{=}0$ else $g_1$ (sampling $\theta$ from
  $f_1$ then adding $\mathcal N(0,1)$ is equivalent to drawing from $g_1$ directly).

> Note: this is a deliberate modeling choice matching "fix $b_0=-2$". If instead you
> profile $b_0$ (re-estimate it under $H_0$ and $H_1$), the out-set genes re-enter
> weakly through $\hat b_0$ and you reintroduce an $n$ dependence. **Recommend fixing
> $b_0$** for clarity, speed, and to keep $(m,b,\mu)$ as the only knobs. Mention the
> profiled variant only as an optional robustness check.

### 2.2 Validated reference implementation

This was run and checked during planning (loc scenario):

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

def expected_lrt_loc(m, b_true, mu, tau0=0.1, n_rep=2000, seed=0):
    rng = np.random.default_rng(seed)
    sig = lambda t: 1.0 / (1.0 + np.exp(-t))
    b0 = -2.0
    sd1 = np.sqrt(tau0**2 + 1.0)                     # scale: use sd1 = sqrt(sigma**2 + 1), g1 mean 0
    g0 = lambda z: norm.pdf(z, 0.0, 1.0)
    g1 = lambda z: norm.pdf(z, mu, sd1)              # scale: norm.pdf(z, 0.0, sd1)
    p_true = sig(b0 + b_true)
    out = np.empty(n_rep)
    for r in range(n_rep):
        gam = rng.binomial(1, p_true, m)
        z = np.where(gam == 1, rng.normal(mu, sd1, m), rng.normal(0.0, 1.0, m))
        d0, d1 = g0(z), g1(z)
        ll0 = np.sum(np.log((1 - sig(b0)) * d0 + sig(b0) * d1))
        negll = lambda b: -np.sum(np.log((1 - sig(b0 + b)) * d0 + sig(b0 + b) * d1))
        res = minimize_scalar(negll, bounds=(-2.0, 12.0), method="bounded")
        out[r] = 2.0 * (-res.fun - ll0)
    return out.mean()

# sanity: expected_lrt_loc(m=45, b_true=3.0, mu=2.0) ~ 46.5
#         (original C2 design targeted LRT ~ 26 / oracle causal-set logBF ~ 10)
```

Runtime: ~1 s for `n_rep=500`, `m=45`. The whole landscape is seconds-to-minutes.
Vectorize the inner max over a $b$-grid (evaluate $\ell$ for all replicates ×
b-grid at once, take the max per replicate) if you want a finer grid; the naive
`minimize_scalar` loop above is already fast enough for a 25-40 point grid.

---

## 3. Deliverable

A new Quarto notebook: **`twogroup_experiments/analysis/detectability_level_curves.qmd`**
(+ its rendered `.html`), following the same knitr+reticulate scaffold as
`analysis/c2_discretization_example.qmd` (copy the `setup` R chunk verbatim - it
resolves `proj_root` and the repo-root `.venv`). All analysis chunks are Python.
No `results/`, no manifest, no gibss imports are needed - just `numpy`, `scipy`,
`matplotlib`.

Notebook structure:

1. **Overview** - the model (§2), the message (§1), and that $f$ is an expected LRT
   over the in-set genes with $b_0=-2$ fixed. Typeset equations as LaTeX
   (`$...$` / `$$...$$`), not code fences.
2. **Location scenario** - compute $f(m,b,\mu)$ on a grid for three set sizes and
   produce the figures in §5.
3. **Scale scenario** - same for $f(m,b,\sigma)$.
4. **Takeaways** - read the iso-LRT curve shift off the plots.

Keep a small module-level cache (write the computed grids to
`analysis/.cache_landscape/*.npz` keyed by scenario+grid params, mtime-guarded)
so re-rendering prose is instant, mirroring the `cached_parquet` pattern in the
existing notebook.

---

## 4. Grids and set sizes (starting values, all tunable)

- **Set sizes** (one panel each): `small = 50`, `medium = 150`, `large = 300`.
  (Matches the 010 design's small 40-50 / large 250-300; medium is new.)
- **loc grid:** $b \in [0, 4]$ (≈30 points), $\mu \in [0, 3]$ (≈30 points).
- **scale grid:** $b \in [0, 4]$, $\sigma \in [0.5, 5]$ (sd of $f_1$).
- **Monte-Carlo reps:** `n_rep = 2000` (bump to 4000 for the final render if the
  contours look ragged). Use a fixed seed **per grid point** derived from its
  indices (no `Date.now`/global RNG) for reproducibility.

---

## 5. Figures (per scenario)

**(A) The money plot - iso-LRT overlay.** One panel. Draw the **LRT = 26** contour
(the original difficulty target; also draw 10 and 50 as lighter references) for
each of small/medium/large on the same $(b,\mu)$ (or $(b,\sigma)$) axes, one colour
per set size. The three curves should nest toward the origin as $m$ grows - that is
the whole point. Label each curve with its $m$.

**(B) Filled-contour panels.** A 1×3 row (small/medium/large), each a filled contour
of $f$ over the grid with contour lines labelled, and the LRT=26 line highlighted.
Shared colour scale across the three so the leftward/downward shift is visible.

Use `matplotlib.pyplot.contour` / `contourf`. For the scale scenario the $y$-axis is
$\sigma$; consider annotating the equivalent $\mathbb E[z^2]=\sigma^2+1$ on a second
axis if helpful. Use the project's Okabe-Ito-style palette for the three set-size
curves if convenient, but any clearly distinct three colours are fine (this notebook
has no method-palette obligation since there are no GSEA methods here).

Give every reported table/figure a one-line legend defining the axes and the
statistic (expected LRT, $b_0=-2$ fixed, in-set genes only).

---

## 6. Conventions (this repo)

- Run everything with **`uv run python` / `uv run quarto ...`**, never bare python.
- Quarto render uses the **knitr engine + reticulate → repo-root `.venv`**; the venv
  is one directory up from `twogroup_experiments/`. Copy the `setup` chunk from
  `analysis/c2_discretization_example.qmd`. Render with the project's usual
  `quarto render` invocation (see `project_quarto_render_gsea` memory / the existing
  notebook's header for the QUARTO_R / pandoc-sandbox caveats).
- One-off exploratory analysis of this kind lives in `twogroup_experiments/analysis/`
  (the `analysis/` folder convention), not `scripts/`.
- Equations as LaTeX math, not backticked code.

---

## 7. Working-tree state

The abandoned $\mathbb E[z^2]=3$ attempt this session was **fully reverted** - the
`twogroup_experiments/` source tree is back to its pre-session state (only the
pre-existing, unrelated `viz_utils.py` / `.gitignore` modifications remain). There
is nothing to reuse or discard: this plan is standalone (§2.2 has the complete
reference primitive; it needs only `numpy` / `scipy` / `matplotlib`).

Two things to be aware of, neither introduced by this plan:
- **Pre-existing 010 drift.** `experiments/010_c2_cost_of_discretizing.yaml` references
  an enrichment `cod_null` and methods `linear_z` / `cox_binned` / ... that no longer
  exist in `experiments/library.yaml`. The 010 supercollection therefore only works
  off the **stale `results/manifest_cache.json`** (gitignored); a manifest *rebuild*
  would crash. This predates the session and is unrelated to this task - do not let it
  confuse you, and do not touch a `experiments/*.yaml` mtime unless you intend to deal
  with it.
- **No new infrastructure is required.** You do **not** need `sized_single_effect`, the
  C2 design, gibss/SuSiE fits, or Snakemake. Everything is in §2.2.

---

## 8. Sanity checks before trusting the plots

1. **Monotonicity:** $f$ increases in $b$, in $\mu$ (loc) / $\sigma$ (scale), and in
   $m$. Any non-monotonicity beyond Monte-Carlo noise is a bug.
2. **Reference point:** $f(m{=}45, b{=}3, \mu{=}2)\approx 46$ (loc). The original C2
   design's difficulty target was LRT $\approx 26$ ⇔ oracle causal-set logBF
   $\approx 10$; use the LRT=26 contour as the "difficulty-matched" locus.
3. **Size shift:** the LRT=26 contour for `large` should sit clearly inside (closer
   to the origin than) the `small` contour. If it does not, revisit the data-gen or
   the in-set reduction.
4. **Scale is harder:** for equal $m$ and $b$, the scale scenario needs a larger
   per-gene signal to reach the same LRT than loc does (variance inflation is weaker,
   symmetric evidence). Expect the scale contours to sit higher on their $y$-axis.

---

## 9. Optional extensions (only if asked)

- A 3-D or faceted view adding $m$ as a continuous third axis (iso-surfaces of $f$).
- Overlay the *discretized* detectability (LRT for a hard-threshold / rank recoding
  of $z$) on the same axes to connect back to the "cost of discretizing" story.
- Profiled-$b_0$ variant (§2.1) as a robustness panel.
