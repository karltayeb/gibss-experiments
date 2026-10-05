# Changepoints with GLM SuSiE — coal-mining disasters (Poisson) + lambda phage GC (binomial)

Brief for Claude Code. Goal: show that (generalized) SuSiE / gibss gives credible
intervals for changepoint locations in non-Gaussian sequences, benchmarked against an
**exact** Bayesian changepoint posterior. Companion to `../cryptic_tb/` and
`../ames_mutagenicity/`.

Repo conventions (`../AGENTS.md`): use `uv`; treat `changepoints/` as the analysis root.
Reuse gibss + `../logistic_susie_experiments/fits/` where possible.

## Setup common to both examples

- Ordered observations y_1..y_n. Design X = step basis: column j (j = 1..n-1) is
  x_tj = 1(t > j). A nonzero b_j is a jump in the linear predictor (log-rate or log-odds)
  between positions j and j+1. Intercept = level of the first segment.
- Each single effect's credible set = an **interval of plausible changepoint locations**
  (adjacent step columns are nearly collinear).
- Structure: X b = cumsum-type op, X^T r = reverse cumsum, so everything is O(n). Dense X
  is fine for n <= ~1000; for large n implement matvecs via cumsums (check whether gibss
  accepts a linear-operator X; if not, keep n small by binning).
- **Standardization**: do NOT standardize the step basis by default (it changes the
  implicit prior on jump size near the edges); report standardized as a sensitivity.
- **Family support**: check that gibss has a Poisson SER (and binomial with trials > 1).
  If Poisson is missing, implement a Poisson SER following the existing logistic
  IRLS/Laplace module (same interface: prep_data / initialize_state / default_schedule)
  and add a unit test against brute-force numerical integration for a single feature.

## Gold standard: exact Bayesian multiple-changepoint posterior

Both examples are conjugate (Poisson–Gamma; Binomial–Beta), so compute the exact
posterior over segmentations with the product-partition / forward-backward recursions of
**Fearnhead (2006), "Exact and efficient Bayesian inference for multiple changepoint
problems", Stat. Comput.** — O(n^2), trivial for n <= ~1000.
Outputs to compare against SuSiE:
- marginal posterior probability of a changepoint at each position (vs SuSiE PIP /
  per-effect alpha);
- posterior on the number of changepoints k (vs number of non-null SuSiE effects /
  credible sets);
- posterior mean rate profile (vs SuSiE fitted step function).
Use a geometric/negative-binomial prior on segment lengths; report sensitivity to it.
Note the models differ (SuSiE: normal prior on jumps in the linear predictor, L
effects; exact: independent segment rates) — agreement on locations is the point, not
identical numbers.

## Example 1 — British coal-mining disasters (Poisson) — core

- Data: dates of explosions killing >= 10, 15 Mar 1851 – 22 Mar 1962, ~191 events
  (Jarrett 1979, Biometrika, corrected series). Available in R as `boot::coal` (decimal
  dates); cross-check count/year totals against another copy (e.g. statsmodels or
  pymc examples). Record source + checks in `docs/data_notes.md`.
- Primary analysis: yearly counts 1851–1962 (n = 112), Poisson, step basis, L in {1,3,5}.
- Classic references to reproduce/compare:
  - single changepoint (Carlin, Gelfand & Smith 1992; Raftery & Akman 1986) — posterior
    mode around 1890 (after 1880s mine-safety legislation);
  - unknown number of changepoints via reversible-jump MCMC (Green 1995, Biometrika),
    which used the event times and found support for >1 change. Optionally implement
    a small RJMCMC as a second reference; the exact recursion is the main one.
- Extensions: (a) finer resolution — monthly (n ~ 1335) or daily counts (n ~ 40k, needs
  cumsum matvecs) to test scale and show CS width in calendar time is stable; (b) L=1
  vs exact single-changepoint posterior (should be nearly identical — good sanity check).
- Expected figure: counts over time + fitted step function + shaded CS intervals, with
  exact posterior changepoint probabilities underneath.

## Example 2 — lambda phage GC content (binomial) — extension

- Data: Enterobacteria phage lambda complete genome, NCBI RefSeq **NC_001416**
  (GenBank J02459), 48,502 bp. Download FASTA + GenBank annotation (Entrez / NCBI
  datasets). y_t = 1 if G/C else 0.
- Primary analysis: bin into windows of 100 bp (also 50, 250, 500, 1000) -> binomial
  counts (k GC out of m bases), n ~ 485 at 100 bp. Logistic link, step basis,
  L in {5, 10, 20}.
- Optional: single-base Bernoulli (n = 48,502) with cumsum matvecs.
- Comparisons:
  - exact Beta–Binomial changepoint posterior (Fearnhead 2006) on the same bins;
  - 2-state HMM posterior state probabilities (Churchill 1989 style); optionally
    published segmentations (Churchill 1989; Boys, Henderson & Wilkinson 2000, JRSS C;
    Braun & Müller 1998, Stat. Sci.) — qualitative comparison only;
  - gene annotation overlay: do CS intervals fall near boundaries between functional
    gene blocks (head/tail structural genes vs regulation/recombination/replication)?
- Known issues to report, not hide:
  - scale dependence: more segments appear at finer bins — show how #CSs changes with
    bin size and L;
  - local dependence (codon periodicity, dinucleotide effects) and overdispersion
    relative to binomial -> CS intervals likely too narrow; check with a
    beta-binomial/quasi-likelihood dispersion estimate on residuals, and/or by
    block-resampling the sequence;
  - truth is fuzzy (no exact "correct" boundaries).

## Calibration (cheap, do for both)

Simulate from the fitted piecewise-constant model (coal: Poisson; lambda: binomial at
100 bp), many replicates; check (i) coverage of true changepoints by 95% CSs,
(ii) false CSs (no true changepoint inside), (iii) CS width vs jump size and distance
from the sequence ends. For lambda also simulate with overdispersion/autocorrelation to
show the narrowing.

## Sensitivity

L; estimated vs fixed prior variance; standardized vs raw step basis; impl
(irls/Laplace vs quadrature where available); bin size (lambda); segment-length prior
for the exact method.

## Deliverables

- `docs/data_notes.md` — sources, checks, counts.
- `scripts/` — numbered steps: 00_download, 01_build (counts/bins, step basis),
  02_exact_posterior, 03_fit_susie, 04_simulate_calibration, 05_summarize.
- `results/coal/`, `results/lambda/` — fits, CS tables (interval start/end in years or
  bp, PIP mass, jump size), exact-posterior comparisons, figures.
- `results/summary.md` — agreement with exact posterior and classic analyses, number of
  changepoints, calibration results, issues, and an honest verdict for each example.
