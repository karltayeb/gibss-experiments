# eqtl: negative binomial SuSiE for cis-eQTL fine-mapping

Does fitting the count model directly (gibss `family="negative_binomial"`, local branch
`nb-augmentations` at 9799751) fine-map cis-eQTLs better than the transform-then-linear
pipeline, how much does getting the overdispersion right matter, and which of gibss's three
NB routes (logistic cumulant, Polya-Gamma, Poisson-Gamma) to use? Isolated uv project; run
everything from this directory.

    uv sync
    uv run pytest
    uv run snakemake --cores 6 results/collections/pilot/fit.parquet
    uv run snakemake --profile <midway profile> results/collections/001-main/fit.parquet

## Simulation

Genotypes: stdpopsim HomSap / OutOfAfrica_3G09 / CEU, chr22 30.0-30.5 Mb on the
HapMapII_GRCh38 map, 20k diploids, simulated once and cached under `results/populations/`
(the GWAS design note's recipe, `notes/gwas_simulation_design.md`). Each replicate samples
n individuals, simplifies, and keeps biallelic sites with in-sample MAF >= `maf_min`
(0.05, jaxQTL's filter). At n = 300 that is ~670 SNPs.

Expression: one gene per replicate, the jaxQTL pseudobulk model without the single-cell
layer,

    y_i ~ NB(mu_i, r),   log mu_i = log l_i + log mu0 + sum_j beta_j g_ij,   r = 1 / phi,

with log l_i ~ N(0, libsize_sd^2) a known library-size factor, mu0 the mean count of a
reference-library individual, phi the DESeq2 / edgeR dispersion, and k causal sites drawn
from the common-MAF band with pairwise |r| <= 0.5, random signs. Effect magnitudes follow
the alpha model, |beta_j| = beta (2 p_j (1 - p_j))^(alpha / 2), alpha = -1 by default (the
GCTA / standardized-genotype convention): every causal explains beta^2 of log-scale variance
and the expected marginal chi-square at the causal is n beta^2 / (1 / mu0 + phi) whatever
site was drawn (`ExpressionSpec.expected_chi2`). `alpha: 0` gives jaxQTL's
frequency-independent log fold change per allele. `src/eqtl/simulate.py`.

Effect ladder (2026-10-23): `beta` in {0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4}, a factor 2 in
chi-square per step. With p ~ 670 a set is declared from chi-square ~25, so each (mu0, phi)
cell at n = 300 has 2-3 rungs on the slope of its power curve. The first pilot used per-allele
|beta| in {0.15, 0.6}, which put every cell at power 0.2 or 1.0; its numbers below are from
that grid. For reference, jaxQTL's own simulation is a single-SNP association power study
(g ~ Binomial(2, MAF) without LD, beta_g ~ N(0, 0.05) per allele at MAF 0.2, alpha = 0.3,
N = 1000, baseline expression swept to vary sample coverage) and never fine-maps.

## Arms (`src/eqtl/fits.py`)

| arm | input | model |
|---|---|---|
| `linear_int` | library-normalized counts, rank inverse-normal | `gibss.linear`, residual variance estimated |
| `linear_log` | log(normalized count + 1) | `gibss.linear`, residual variance estimated |
| `nb_true` | counts, offset log l, dispersion = simulated r | NB gIBSS (`fit_glm_susie`) |
| `nb_est` | counts, offset log l, dispersion from `negbin.estimate_null_dispersion` | NB gIBSS |
| `nb_score_rss` | NB score-test z at the estimated-r null + GLM-weighted in-sample LD | SuSiE-RSS (`src/eqtl/rss.py`) |
| `nb_score_rss_true` | the same summary statistics at the true r (null intercept refit at that r) | SuSiE-RSS |
| `nb_mle_rss` | one NB regression per SNP at the estimated r (Fisher scoring), Wald z = bhat / se, plain in-sample genotype LD | SuSiE-RSS on sufficient statistics (`rss.susie_rss`); the DESeq2-style effect-and-SE path |
| `nb_cavi` | counts, offset log l, estimated dispersion | CAVI in Q2, spectral kernel (`cf_cavi_spectral`), null init, EB prior variance, 50-sweep cap (no prior-variance freeze: see notes) |
| `nb_cavi_true` | the same at the true r (wired, not in the collections) | CAVI-Q2 spectral |
| `nb_polya_gamma` | counts, offset log l, estimated dispersion | NB by its local Polya-Gamma augmentation (`method="polya_gamma"`, localjj on the NB base; Gaussian base problem), gIBSS |
| `nb_poisson_gamma` | counts, offset log l, estimated dispersion | NB by its local Poisson-Gamma augmentation (`method="poisson_gamma"`, Gamma exposure per row and feature; Poisson base problem), gIBSS. Posterior variances deflate by ~1/(1 - mu/(mu + r)); CSs close |
| `poisson` | counts, offset log l, no dispersion | Poisson gIBSS (`family="poisson"`); the phi = 0 companion to `linear_log` |
| `poisson_cavi` | counts, offset log l, no dispersion | Poisson exact CAVI-Q2 (`cf_cavi`; the spectral kernel is logistic-only), same settings as `nb_cavi` |

The three NB routes are named by the base regression problem each reduces to. The logistic
cumulant at eta - log r is exact (NB = Binomial with y + r trials), so `nb_cavi` (CAVI-Q2 on
it) maximizes the exact NB ELBO over Gaussian q and is the accuracy reference; the two
augmentations optimize certified lower bounds on that ELBO in closed form. `fit.elbo` is the
exact ELBO of every count-arm state, so the routes are scored on one scale
(`analysis/nb_routes.py`).

All arms center the design, estimate the intercept and the prior variance (EB). The
inverse-normal arm is the GTEx / jaxQTL-linear baseline (with one gene the TMM factor is
the library size). The RSS arms are jaxQTL's calling + fine-mapping path: G_hat =
G - X(X'WX)^-1 X'WG, z_j = g_hat_j' s / sqrt(D_j), R_hat = D^-1/2 G_hat' W G_hat D^-1/2,
then SuSiE-RSS z ~ N(R_hat b, R_hat). It runs through `gibss.linear` on a synthetic design
with Gram matrix R_hat and cross-product z (exact identity, `tests/test_rss.py`), so there
is no susieR dependency. Wald z from per-SNP fits is not in the row space of the LD, so
`nb_mle_rss` runs the same Gaussian IBSS on the sufficient statistics (R, z) directly
(`rss.susie_rss`, susieR's `susie_rss` with lambda = 0); on score statistics it reproduces
the gibss route effect for effect and sweep for sweep (`tests/test_rss.py`). The score arms
were called `nb_rss` / `nb_rss_true` before 2026-10-27.

## Pipeline (`Snakefile`, `config.yaml`)

Cells are content-addressed on (population spec, expression spec); collections are
Cartesian grids with a replicate count, L values and arms. Shared cells are simulated and
fitted once.

    results/populations/<pop>.trees, .sites.parquet
    results/cells/<cell>/sims/reps_<start>_<stop>.npz                      # one batch of replicates
    results/cells/<cell>/fits/<arm>_L<L>/reps_<start>_<stop>/{pip,cs,fit}.parquet
    results/collections/<name>/{pip,cs,fit}.parquet     # concatenated + cell coordinates

`pip`: one row per feature (pip, posterior_mean, causal, beta_true, r2_causal, maf).
`cs`: one row per component (ser_log_bf, cs_size, purity, beta, covered, cs_r2_max,
members). `fit`: one row per fit (seconds, sweeps, converged, r_true, r_used, null-model
fit, prior variances). Aggregations in `src/eqtl/evaluate.py`; the notebook in `analysis/`.

## Pilot (2026-10-21, 16 cells x 5 reps, n = 300)

- All six arms agree at L = 1; the NB arms carry higher log BF at the causal (9-10 vs 6 nats
  on a weak cell) and the null-model r is 0.2-0.8 log2 units under the truth (conservative).
- At L = 5 on the high-information cells (mu0 = 1000, phi = 0.05, |beta| = 0.6) the NB gIBSS
  arms declare sets with no causal in them (0.4-0.6 per rep) that the linear and RSS arms do
  not. Dissected replicate: the rare causal (MAF 0.10, |z| 15) sits in LD with a common SNP
  that tags both causals (|z| 27); gIBSS's support {610, 591, 633} is 2.3 nats below the
  truth in NB log-likelihood, so fixed-L IBSS from the null init walks into a near-tied
  split. Greedy forward selection (`L: auto`, supported in the pipeline but not in the
  collections) recovers the truth with log BFs 236 / 526 vs 99 / 87 / 108. The main grid will
  say how often this happens.
- Spectral CAVI-Q2 (`nb_cavi`, 022 settings) converges in 5-43 sweeps (8-93 s/rep at n = 300;
  dosage is 0/1/2 so it takes the dense O(n p ntau) path) and lands in the SAME split
  configuration as gIBSS on the dissected replicate: the split is the variational fixed point
  reached from the null init, not the gIBSS plug-in. Warm-starting CAVI from the gIBSS fit
  converges in 2-3 sweeps to the same sets.
- `analysis/relative_cs.py`: the 021/025 relative-CS figure (coverage / power deltas, size
  ratio, paired bootstrap bands) against a reference arm, default `linear_int`.

## NB routes and the Poisson companion (2026-10-22, pin 9799751)

Pilot, L = 5, every count arm scored by the exact NB ELBO against `nb_cavi` (80 reps):

| arm | ELBO - ref (median / min, nats) | max PIP diff (median) | identical declared sets | sd of top component / ref | seconds / ref |
|---|---|---|---|---|---|
| `nb_est` (gIBSS quad) | 0.01 / -0.01 | 0.0004 | 75% | 1.00 | 0.19 |
| `nb_polya_gamma` | -0.6 / -16.5 | 0.001 | 63% | 0.65 | 0.17 |
| `nb_poisson_gamma` | -1.6 / -97.8 | 0.004 (max 0.5) | 50% | 0.42 | 0.20 |

- gIBSS on the logistic cumulant IS the exact CAVI-Q2 fit (same ELBO to a few hundredths of
  a nat, same PIPs, same posterior sd) at a fifth of the time. The two augmentations land on
  the same PIPs most of the time but understate the posterior sd (Polya-Gamma by a third,
  Poisson-Gamma by more than half, growing with s = mu0 / (mu0 + r) as its docstring
  predicts), and Poisson-Gamma loses tens of nats and whole credible sets on the
  high-count, high-overdispersion cells. Neither earns a place in the collections.
- Dosage level collapse (5292a59): `nb_cavi` L = 5 median 46 s / max 85 s per rep at n = 300,
  down from 28 / 261 on 1fa286a. It still runs to the 50-sweep cap on 85% of reps, so gIBSS
  (13 s, 100-sweep cap) remains the cheaper route; the 7-9 s CAVI timings seen with the 022
  freeze protocol were the freeze stopping CAVI early, not convergence.
- `freeze_prior_variance=0.01` (the 022 protocol) is harmful on count data: on 003-poisson at
  mu0 = 1000 it froze early-sweep components and CAVI "converged" in 5 sweeps 187 nats below
  gIBSS with four false single-SNP sets (log BF up to 1378). Both CAVI arms now run plain EB
  with only the sweep cap; `cf_cavi` then reproduces the gIBSS fit to the cent.
- `003-poisson` (Poisson truth, 8 cells x 5 reps): `nb_est` does not recover the Poisson model
  when a causal is present. The null-model r_hat is ~100 at |beta| 0.15 and 6-14 at |beta|
  0.6 (never at the 1e4 cap with a causal; a null gene hits it), because the intercept +
  offset null reads the eQTL variance as overdispersion. Cost: at mu0 = 10, |beta| 0.6 the
  nb_est set is 3x the Poisson one (median 32 vs 9 SNPs); at mu0 = 1000 LD bounds the set
  and the arms coincide. Poisson vs log-linear: Poisson sets are smaller at low counts (9 vs
  13 at |beta| 0.6, 36 vs 54 at |beta| 0.15 with two causals), identical at mu0 = 1000.
  `poisson` and `poisson_cavi` agree (ELBO within 0.02 nats median) except one rep where
  CAVI at the cap sits 19 nats below gIBSS.

## Ladder pilot (2026-10-27, 56 cells x 5 reps, standardized beta ladder, 7 arms)

`analysis/_site/results_pilot.html` and `results_003-poisson.html`. The augmentation arms
were dropped from the pilot collection (scored above; their fits stay under `results/cells`).

- The ladder has a slope now. At L = 5, mean over the eight (mu0, phi, k) corners, power
  runs 0.10 / 0.28 / 0.40 / 0.55-0.69 / 0.75-0.84 / 0.90-0.96 / 0.80-0.90 over beta 0.05
  to 0.4 for every arm; the top rung is the two-causal cells with |r| up to 0.5 between
  causals. Coverage sits at 0.8-1.0 everywhere.
- The individual-level NB arms (`nb_true`, `nb_est`, `nb_cavi`) lead the inverse-normal
  baseline by 0.05-0.1 power on the 0.14-0.28 rungs (0.64-0.69 vs 0.55 at 0.14) with the
  same set sizes; with 5 reps per cell that is 2-4 causals out of 40. Nothing separates the
  arms below 0.1 or at 0.4. `nb_cavi` == `nb_est` on every rung (same power, size, false
  sets).
- Estimated vs true dispersion: the intercept-only null reads the eQTL as overdispersion,
  median log2(r_hat / r) = -0.45 at beta 0.1, -1.2 at 0.2, -2.4 at 0.4 when phi = 0.05;
  -0.45 at 0.4 when phi = 0.5. It never hits the cap. The cost is invisible in the
  aggregate (`nb_est` and `nb_true` match within rep noise); it shows in single weak cells
  (mu0 = 10, phi = 0.5, k = 1, beta 0.14: power 0.4 vs 0.8).
- RSS: `nb_score_rss` (null-model r) tracks `nb_est`. `nb_score_rss_true` breaks at the top rung:
  coverage 0.70 vs 0.87-0.91, 0.6 false sets per rep vs 0.2, median declared log BF 78 vs
  36-38 (max 1614 vs 183), smallest sets (median 9). The larger true r inflates the score
  z and the in-sample LD no longer matches them (the usual SuSiE-RSS failure); the null
  model's deflated r is what keeps the jaxQTL path honest. Both RSS arms run to the sweep
  cap on half the reps at L = 5 with unchanged sets.
- False sets at L = 5 are 0.1-0.2 per rep for every arm from rung 0.1 up, linear included,
  so the split-set behaviour of the first pilot is not NB-specific at matched signal.
- `003-poisson` (mu0 1 / 10 / 100, same ladder): `poisson` beats `linear_log` by 0.03-0.05
  power at the top rungs (0.95 vs 0.92 at 0.4, 0.88 vs 0.83 at 0.28) with the same sizes,
  and `poisson` == `poisson_cavi`. `nb_est` pays for its r_hat on Poisson counts with wider
  sets at the top rungs (median 23 vs 15 at 0.28, 15 vs 11 at 0.4) at equal power.
- The effect-recovery panel now shows only the individual-level NB arms: the RSS
  coefficient is the z-scale noncentrality, not a log fold change.

## Notes

- gibss pin: local clone at 9799751 (`nb-augmentations`; stacks 1fa286a NB family, 5292a59
  spectral level-collapse of 0/1/2 dosages, 9799751 augmentation routes; none on origin yet);
  switch the url in `pyproject.toml` once pushed, keep the rev explicit.
- The null-model dispersion is intercept + offset only (no covariates in v1); it
  underestimates r by the causal variance share, which is conservative.
- Laptop timing at n = 300, p ~ 670: NB gIBSS L=5 ~8 s/rep, L=1 ~0.5 s; linear and RSS
  arms < 1 s. `nb_true` and the RSS arms can run to the 100-sweep cap at L=5 (ELBO / alpha
  stopping rules creep while null components decay) with the same credible sets as the
  converged arms; `fit.parquet` records sweeps and convergence.
