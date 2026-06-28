# Univariate logBF grid and effect-size calibration

**Date:** 2026-06-27 (rev. 2026-06-28: exact-LRT sizing; Wald-z demoted to an aside)
**Companion notebook:** `notebooks/effect_size_calibration.py` (marimo)

## What we size by

We size simulated effects by the **expected univariate log Bayes factor** of the
causal feature — design-agnostic, monotone in signal, not gameable by credible-set
size (unlike coverage), not capped by correlation (unlike causal PIP: rho affects
*localization*, not the causal's own marginal evidence). Grid:

```
expected univariate logBF ∈ {4, 8, 16, 32, 64}   (nats)
```

(Dropped logBF=2 — redundant with the null sims. Kept 64 — see "reachability".)

## Sizing: the exact expected LRT (`Λ = n·MI`), glmbf-free

The asymptotic Bayes factor is `logBF ≈ Λ − ½log n + O(1)`, where the dominant
term is the expected log-likelihood ratio

```
Λ(b) = n · MI(x; y) = n · [ H(p̄) − E_x H(p_x) ],   p_x = σ(b0 + b·x),  p̄ = E_x[p_x]
```

(mutual information × n; `H` = binary entropy). Expectations are over the causal
feature's **marginal** (binary = 2 atoms; gaussian/uniform = 1-D quadrature),
so sizing is **rho-independent**. `Λ(b)` is **monotone increasing in b**,
saturating only at the **null deviance** `n·H(p̄_∞)` (hundreds of nats — see
table), so every grid rung is reachable by a simple 1-D monotone root find
`Λ(b) = target`. No glmbf, no pilot needed for the point estimate; the notebook
**validates by simulation** (realized LRT mean hits target).

Why the LRT and not the Wald statistic: the Wald `z` is **non-monotone** in `b`
(it saturates and falls — see aside), so `½z²` is *not* a valid sizing target at
strong signal. The LRT is the actual evidence and stays monotone.

### Relation to z (regular-regime heuristic only)

In the regular regime `Λ ≈ ½z²`, so `z ≈ √(2·logBF)` is a convenient label
(logBF 4/8/16/32/64 → z ≈ 2.83/4/5.66/8/11.31). But **logBF and z² decouple at
saturation** (the LRT keeps rising while z falls); treat z as a mnemonic, not a
target.

### Multiplicity (SER detection)

A single-effect regression over `p` features pays an ~`log p` penalty:
`SER logBF ≈ uni_logBF − log p`. At p=500 (`log p ≈ 6.2`) detection (SER logBF >
`min_log_bf`=2) needs `uni_logBF ≳ 8`, so the lower rungs {4, 8} probe the
low-/no-power floor and {16, 32, 64} the detected regime — but detection is only
one outcome; localization and CS calibration are measured throughout.

## Frozen effect sizes `b` (exact-LRT sizing, n=1000)

| design | b0 | L4 | L8 | L16 | L32 | L64 | deviance ceiling |
|---|---|---|---|---|---|---|---|
| gaussian | 0 | 0.180 | 0.256 | 0.366 | 0.531 | 0.789 | ~671 |
| uniform | 0 | 0.311 | 0.441 | 0.628 | 0.902 | 1.316 | ~665 |
| binary q=0.5 | 0 | 0.179 | 0.254 | 0.361 | 0.514 | 0.740 | ~693 |
| binary q=0.1 | 0 | 0.299 | 0.425 | 0.607 | 0.874 | 1.286 | ~325 |
| gaussian | −2 | 0.275 | 0.387 | 0.545 | 0.766 | 1.091 | ~671 |
| uniform | −2 | 0.477 | 0.675 | 0.953 | 1.347 | 1.910 | ~665 |
| binary q=0.5 | −2 | 0.277 | 0.392 | 0.556 | 0.793 | 1.137 | ~693 |
| binary q=0.1 | −2 | 0.440 | 0.613 | 0.851 | 1.180 | 1.649 | ~325 |
| gaussian | −4 | 0.621 | 0.830 | 1.086 | 1.405 | 1.829 | ~670 |
| uniform | −4 | 1.124 | 1.543 | 2.077 | 2.735 | 3.537 | ~663 |
| binary q=0.5 | −4 | 0.674 | 0.951 | 1.337 | 1.853 | 2.502 | ~693 |
| binary q=0.1 | −4 | 0.974 | 1.313 | 1.752 | 2.316 | 3.039 | ~325 |

**000_markov uses the b0 ∈ {0, −2} blocks; −4 is included for the later
imbalanced experiment** (now fully computable — no ragged cells). All rungs are
far below every deviance ceiling, so the grid is reachable at every (design, b0),
and — because the ceiling is the deviance, not a Wald limit — **also at n=500**.
The notebook regenerates this table for any n / b0.

## Aside: the Wald ceiling (a property of Wald-approximation methods)

The expected **Wald** statistic `E[z] ≈ b·√(n·I_W(b))` with `I_W` the *observed*
(true-parameter) information is **non-monotone**: as `b→∞` the data separate, the
weights `w_i = p_i(1−p_i)` collapse, `I_W` falls, the SE inflates faster than
`b̂`, and `z` rises then **falls**. So `z` has a finite maximum `z_max`, and the
`½z²` ABF of **Wald-approximation methods (Laplace / JJ)** saturates with it. For
binary q=0.5, b0=−4, n=1000: `Λ` climbs 8.8 → 38 → 214 → 648 (monotone), while
`½z²` peaks ~9.6 at b≈2 then decays to 0.

This is **not** a limit on the exact (quadrature) BF and **not** a sizing
constraint — it is why Laplace/JJ BFs under-state evidence at strong, separated
signal. (Earlier drafts mistook this for a logBF ceiling.)

### Cell-balance and the q=0.1 > q=0.5 Wald inversion

For ±1 binary, `I_W = 4AB/(A+B)`, `A=(1−q)w₋`, `B=q·w₊` (cell mass × weight) —
twice the harmonic mean, bottlenecked by the smaller cell, maximal at `A≈B`. At
b0=−4 a positive effect makes the treated cell high-weight while the control cell
is low-weight, so balance needs more *mass* on the control cell ⇒ **small q**:
the Wald `z_max` peaks near q≈0.2 (q=0.1 beats q=0.5). Symmetric at b0=0 (peak
q=0.5). A curiosity of the Wald statistic, not of the evidence.

## Empirical spread

Around the target, the realized LRT (and Wald z, in the regular regime) varies by
~±1 in z-units; spread is **wider for the rare binary** (q=0.1) and its low tail
can reach ~0 (occasionally undetectable even when the mean is on target). The
*target is the mean*; replicates vary. ECDFs/histograms faceted by design × n are
in the notebook.

## Implications for 000_markov

- Effect size = solve `b` per **(design-marginal, b0, target_logBF)** via the
  monotone `Λ(b)=target` root find; **frozen** in the experiment yaml (table
  above), reused across the rho sweep.
- Grid `logBF ∈ {4, 8, 16, 32, 64}`, b0 ∈ {0, −2}, n=1000, p=500.
- Expect intrinsic per-replicate spread (~±1 in z); the target is the mean.
