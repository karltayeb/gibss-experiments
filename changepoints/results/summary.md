# Changepoints with GLM SuSiE: results

Pinned gibss-mono rev e2fd039 (local main on 2026-10-05, not yet pushed; it adds the
Binomial family the lambda example uses). All numbers below are
reproduced by the numbered scripts; see `README.md`.

## Coal-mining disasters (Poisson)

Data: 191 explosions, 1851-1962 (`docs/data_notes.md`). Yearly counts, n = 112, step
basis with 111 columns (column j = jump in log rate between year j and year j+1), no
standardization, shared estimated intercept, columns centered, estimated prior variance.
The gibss method is the default Poisson SER (free-form quadrature over the effect, plug-in
offsets). A CS is "declared" when its SER log Bayes factor is at least 2 nats.

Exact benchmark: Poisson-Gamma product-partition model (Fearnhead 2006), Gamma(shape a,
rate a / mean count) on segment rates, every year boundary a changepoint independently
with probability p (geometric segment lengths). Defaults a = 1, p = 0.02 (2.2 expected
changepoints a priori). The forward-backward sums are checked against brute-force
enumeration in `tests/test_exact.py`.

### Single changepoint: SuSiE L=1 against the exact posterior

Figure `results/coal/figures/coal_year_L1_vs_exact_single.png`.

| quantity | SuSiE L=1 | exact (one changepoint, uniform location) |
|---|---|---|
| posterior mode (first year of the new regime) | 1892 | 1892 |
| 95% set (first years of the new regime) | 1886-1894 and 1897 (9 years) | the same 9 years |
| exact posterior mass inside the SuSiE CS | 0.958 | |
| log BF, changepoint vs none | 31.2 | 29.4 |
| max abs difference in location probabilities | 0.007 | |
| total variation distance between the two location posteriors | 0.011 | |
| rate ratio after / before | 0.30 (log jump -1.19, sd 0.13) | |

The SuSiE L=1 credible set is the exact 95% HPD set, element for element, even though the
models differ (normal prior on the log-rate jump vs independent Gamma rates). The classic
location (Raftery & Akman 1986; Carlin, Gelfand & Smith 1992: 1889-1892) is reproduced.
Column definitions: "95% set" is the smallest set of year boundaries whose posterior mass
reaches 0.95 (SuSiE: alpha of effect 1; exact: the single-changepoint location posterior);
"log BF" is SuSiE's SER log Bayes factor and the exact marginal-likelihood ratio.

### Multiple changepoints: L = 3, 5 against the exact multi-changepoint posterior

Figure `results/coal/figures/coal_year_main.png` (fit, CS hulls, exact mean rate profile
and marginal boundary probabilities).

SuSiE, yearly, default method (identical at L=3 and L=5; the extra effects are null with
prior variance ~0.004):

| effect | log BF | CS size | CS hull (first year of new regime) | top | top alpha | log jump | rate ratio |
|---|---|---|---|---|---|---|---|
| 1 | 22.9 | 10 | 1886-1897 | 1892 | 0.18 | -1.03 | 0.36 |
| 2 | 1.8 | 38 | 1883-1961 (0.93 of its mass in 1936-1959, 0.91 in 1940-1955) | 1948 | 0.34 | -1.18 | 0.31 |

Column definitions: "log BF" is the SER log Bayes factor of the effect against b = 0;
"CS size" the number of step columns in the 95% credible set; "CS hull" the span from the
smallest to the largest column in the set; "top" the column with the largest alpha and
"top alpha" that alpha; "log jump" the posterior mean of the top column's coefficient
(log rate ratio after vs before) and "rate ratio" its exponential.

Exact posterior, yearly, Gamma shape 1 (p is the prior per-boundary changepoint
probability; "mass" is the summed marginal boundary probability in the window, the
expected number of changepoints there; "P(cp in window)" is from 4000 exact posterior
samples of the segmentation):

| p | prior E[k] | E[k \| y] | mode k | P(k=1) | P(k=2) | P(k>=3) | P(cp in 1886-1897) | P(cp in 1940-1955) | P(cp in 1925-1935) |
|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.56 | 1.67 | 1 | 0.46 | 0.43 | 0.11 | 0.98 | 0.47 | 0.04 |
| 0.01 | 1.11 | 2.13 | 2 | 0.25 | 0.47 | 0.28 | 0.98 | 0.66 | 0.12 |
| 0.02 | 2.22 | 2.92 | 2 | 0.09 | 0.33 | 0.58 | 0.97 | 0.83 | 0.25 |
| 0.05 | 5.55 | 5.20 | 5 | 0.01 | 0.06 | 0.94 | 0.97 | 0.96 | 0.60 |
| 0.10 | 11.1 | 9.16 | 9 | 0.00 | 0.00 | 1.00 | | | |

Gamma shape 0.5 and 2 move E[k | y] by at most 0.15 at fixed p (`results/coal/year/exact/grid_summary.csv`).

Reading:

- Locations agree. SuSiE effect 1 alpha tracks the exact marginal boundary probability
  around 1892 (peak 0.18 vs 0.20), and effect 2 sits where the exact posterior puts its
  second mode, a drop from about 1.1 to 0.5 disasters per year in 1948 (Green 1995 found
  the same second change with reversible-jump MCMC on the event times).
- The number of changepoints is where the methods differ, and the exact answer depends on
  the prior as much as on the data. Under the exact model P(k >= 2) runs from 0.54
  (p = 0.005) to 0.91 (p = 0.02). SuSiE's second effect has log BF 1.8, below the 2-nat
  declaration threshold: it reports one declared changepoint plus one sub-threshold
  candidate. The exact posterior's 1930 bump (an upward blip in 1930-1947) gets 0.25 at
  p = 0.02 and SuSiE does not pick it up at all.
- The second effect's CS is a poor interval. Its hull is 1883-1961, but 93% of its alpha
  lies in one block, 1936-1959. The 95% set collects a tail of tiny alphas scattered across
  the series (a known weakness of coverage-defined CSs on weak effects); the contiguous
  block is the honest interval. Report both.
- The fitted step function and the exact posterior mean rate profile agree except that the
  exact profile has the 1930-1947 bump SuSiE smooths over, and SuSiE's own second step
  (1948) is softened because that effect is sub-threshold.

### Resolution: yearly vs monthly (`results/coal/figures/coal_resolution.png`)

| resolution | n | effect | log BF | CS size | CS in calendar time | width (years) | top |
|---|---|---|---|---|---|---|---|
| year | 112 | 1 | 22.9 | 10 | 1886-1897 | 12 | 1892 |
| month | 1344 | 1 | 23.6 | 104 | 1886.75-1897.0 | 10.3 | 1890.25 |
| year | 112 | 2 | 1.8 | 38 | 1883-1961 | 79 | 1948 |
| month | 1344 | 2 | 1.6 | 518 | 1851.1-1962.9 | 112 | 1947.75 |

The CS width in calendar time and the log BFs are stable from 112 to 1344 positions
(the monthly CS has 104 members because the monthly alpha is spread over 12x more
columns, but it covers the same decade). The dense 1344 x 1343 design fits in 2-4 s;
a cumsum linear operator was not needed. The monthly top for effect 1 (Mar 1890) differs
from the yearly top (1892) because the yearly binning cannot see within-year timing; both
sit inside each other's CS.

### Sensitivity (yearly, effect 1 CS; `results/coal/year/cs_table.csv`)

| variant | L | log BF | CS size | CS hull | top | effect 2 log BF |
|---|---|---|---|---|---|---|
| default (raw basis, EB prior var) | 1 / 3 / 5 | 31.2 / 23.2 / 22.9 | 9 / 10 / 10 | 1887-1897 / 1886-1897 | 1892 | - / 1.75 / 1.77 |
| standardized step basis | 1 / 3 / 5 | 31.2 / 23.1 / 22.9 | 9 / 10 / 10 | same | 1892 | - / 1.76 / 1.76 |
| fixed prior variance 0.25 | 1 / 3 / 5 | 29.4 / 20.3 / 17.3 | 10 / 11 / 12 | 1887-1897 / 1886-1897 / 1883-1897 | 1892 | - / <0.5 / <0.5 |
| fixed prior variance 1.0 | 1 / 3 / 5 | 31.1 / 23.0 / 21.2 | 9 / 10 / 11 | same as default | 1892 | - / 1.19 / <0.5 |
| cf_cavi (exact CAVI, Gaussian q) | 1 / 3 / 5 | 31.7 / 24.4 / 24.1 | 9 / 10 / 10 | same as default | 1892 | - / 1.57 / 1.59 |
| irls (plug-in IRLS) | 1 / 3 / 5 | 31.3 / 31.2 / 31.1 | 10 / 10 / 10 | 1888-1898 / 1887-1897 | 1892 | - / <0.5 / <0.5 |

Standardizing the step basis changes nothing here: the estimated prior variance rescales
(1.4 -> 0.32) and the CS is identical. The step columns' standard deviations only vary
2.5-fold across the interior, so the implicit prior on edge jumps barely moves; on a
series with a changepoint near an end this would matter more. Fixing the prior variance
small (0.25) widens the CS and kills the second effect. The IRLS plug-in gives the same
CS with log BF 31, matching the default at L=1. At L=3 and L=5 it stays at 31 while the
default drops to 23; IRLS also finds no second effect. Under the earlier
2079de2 pin IRLS reported 36-37 nats here; gibss 6646e02 (null intercept scored tight)
fixed that. The exact-CAVI fit (cf_cavi) matches the default within 1.5 nats.

### Calibration by simulation (`results/coal/year/calibration/`)

Design A: Poisson counts simulated from a two-changepoint step function fit to the data
(new regimes from 1892 and 1948; rates 3.10, 1.11, 0.47 per year), 200 replicates, n = 112.
Figure `calibration_design_a.png`.

| L | declared CSs | CS coverage | false CSs | median CS size | median hull width | mean declared per rep | mean false per rep | recall 1892 | recall 1948 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 200 | 0.945 | 11 | 6 | 7 | 1.00 | 0.055 | 0.945 | 0.02 |
| 3 | 337 | 0.955 | 15 | 9 | 9 | 1.69 | 0.075 | 0.98 | 0.63 |
| 5 | 336 | 0.955 | 15 | 9 | 9 | 1.68 | 0.075 | 0.975 | 0.63 |

Column definitions: "CS coverage" is the fraction of declared CSs (log BF >= 2) that
contain a true changepoint column; "false CSs" the count that contain none; "recall" the
fraction of replicates in which some declared CS contains that true changepoint. The exact
posterior mean number of changepoints on the same replicates (a = 1, p = 0.02) is 2.85
against a truth of 2.

Declared CSs cover a true changepoint at the nominal 95% rate at every L, and L=3 and L=5
behave identically. The 1948 drop (1.1 -> 0.47 over the last 15 years) is a weak signal:
SuSiE declares it in 63% of replicates, and the exact model with p = 0.02 answers the
"how many" question with 2.85, over the truth, because its prior expects 2.2 changepoints
in a series that has 2. The false CSs are near misses, not spurious locations: at L=1
the gap from the CS hull to the nearest true column is 1-3 columns in 7 of 11 cases
(the single step is pulled a little late by the second drop it cannot model: mean top
offset +1.3 years at L=1 vs +0.06 at L=3), and at L=3/5 it is 1-5 columns in 11 of 15,
mostly CSs for the weak 1948 drop that stop short of column 96. In 3-4 cases per L the
hull contains the true column but the set does not (the scattered-tail issue above).

Design B: one changepoint at step column c (new regime from year 1852 + c), base rate 3
per year, rate ratio r after the change, n = 112, 100 replicates per cell. SuSiE L=1
against the exact single-changepoint 95% HPD set on the same replicate; L=3 for false
sets. Figure `calibration_design_b.png`; full table `design_b_summary.csv`.

| new regime from | years to nearest end | r | L=1 power | L=1 coverage | exact HPD coverage | L=1 median set size | exact median HPD size | exact mass in L=1 CS | L=3 false per fit |
|---|---|---|---|---|---|---|---|---|---|
| 1857 | 6 | 0.25 | 0.66 | 0.99 | 0.98 | 12.5 | 13 | 0.95 | 0.01 |
| 1867 | 16 | 0.25 | 0.99 | 0.95 | 0.95 | 6 | 6 | 0.96 | 0.06 |
| 1882 | 31 | 0.25 | 1.00 | 0.97 | 0.96 | 5 | 5 | 0.96 | 0.04 |
| 1907 | 56 | 0.25 | 1.00 | 0.96 | 0.96 | 4.5 | 4.5 | 0.97 | 0.07 |
| 1932 | 31 | 0.25 | 1.00 | 0.98 | 0.98 | 5 | 5 | 0.97 | 0.03 |
| 1947 | 16 | 0.25 | 0.99 | 0.95 | 0.96 | 5 | 5 | 0.97 | 0.05 |
| 1957 | 6 | 0.25 | 0.53 | 0.98 | 0.98 | 10 | 10.5 | 0.96 | 0.02 |
| 1857 | 6 | 0.33 | 0.49 | 0.98 | 0.98 | 33.5 | 37 | 0.96 | 0.03 |
| 1867 | 16 | 0.33 | 0.97 | 0.96 | 0.96 | 9 | 9 | 0.96 | 0.05 |
| 1907 | 56 | 0.33 | 1.00 | 0.98 | 0.98 | 6 | 6 | 0.96 | 0.04 |
| 1947 | 16 | 0.33 | 0.99 | 0.93 | 0.93 | 8 | 8 | 0.96 | 0.07 |
| 1957 | 6 | 0.33 | 0.26 | 0.99 | 0.99 | 54 | 42 | 0.96 | 0.01 |
| 1857 | 6 | 0.5 | 0.15 | 0.99 | 0.99 | 91.5 | 82 | 0.97 | 0.00 |
| 1867 | 16 | 0.5 | 0.60 | 0.98 | 0.96 | 32 | 31 | 0.96 | 0.02 |
| 1907 | 56 | 0.5 | 0.98 | 0.96 | 0.96 | 14.5 | 14 | 0.96 | 0.04 |
| 1947 | 16 | 0.5 | 0.53 | 1.00 | 0.98 | 35 | 27.5 | 0.96 | 0.02 |
| 1957 | 6 | 0.5 | 0.09 | 0.99 | 0.99 | 101.5 | 84 | 0.97 | 0.02 |

Column definitions: "power" is the fraction of replicates with log BF >= 2; "coverage"
the fraction of replicates whose 95% set (SuSiE L=1 CS, or the exact HPD set of the
single-changepoint location posterior) contains the true column, computed over all
replicates; "set size" the number of columns in that set; "exact mass in L=1 CS" the
exact location posterior's mass inside the SuSiE CS (0.95 means the two sets agree);
"L=3 false per fit" the mean number of declared L=3 CSs that miss the true column.

Coverage is at or above 0.95 in every cell for both methods, and SuSiE's set matches the
exact HPD set in size and in the mass it captures (0.95-0.97) wherever the signal is
detectable. Near the ends the two diverge in the expected direction: with 6 years to an
end and a mild jump, SuSiE's set is wider than the exact one (54 vs 42, 101 vs 84 columns)
because the normal prior on the jump is centered at zero and the few edge observations
cannot pin a jump size, so alpha spreads over the whole series; coverage stays nominal
only because the set is nearly everything. Power is the real edge cost: 0.53-0.66 at
ratio 0.25 and 0.09-0.15 at ratio 0.5 with 6 years to an end, against 0.98-1.0 in the
middle. L=3 adds 0.00-0.07 false declared CSs per fit, all near misses of the kind seen
in design A.

### Verdict for the coal example

Where the truth is one changepoint, Poisson SuSiE with L=1 reproduces the exact Bayesian
changepoint posterior: same mode, same 95% set, location probabilities within 0.007. With
L=5 it finds the same two locations the exact multi-changepoint posterior finds, with
calibrated 95% intervals (coverage 0.945-0.955 in design A, 0.93-1.00 in every design B
cell, matching the exact HPD set size wherever the jump is detectable) and few false sets,
all near misses. Power collapses within about 6 years of either end of the series. What it
does not do is answer "how many changepoints" the way the exact model does: SuSiE's second
effect stays sub-threshold at log BF 1.8 while the exact posterior gives P(k >= 2) between
0.54 and 0.91 depending on its segment-length prior, and the exact posterior also entertains
a third change around 1930 that SuSiE ignores. SuSiE is the more conservative counter, and
its count is not prior-tunable in the same way. The CS of a weak effect is a hull plus
scattered tail members; the contiguous block carrying most of the alpha is the interval to
report. Known issue to carry forward to the lambda example: coverage-defined CSs on
sub-threshold effects are not intervals.

## Lambda phage GC content (binomial)

Data: NC_001416.1, 48,502 bp, GC 0.499 (`docs/data_notes.md`). y_t = 1(G or C), binned
into non-overlapping windows; binomial counts (k of m), logit link, step basis, no
standardization, centered columns, shared intercept. The fit is gibss's own Binomial
family, `fit_glm_susie(X, k, trials=m)` (added in e2fd039). `tests/test_binomial.py`
checks its single-feature log BF against numerical integration and its fit against the
Bernoulli fit on the expanded rows. An earlier local Binomial family gave bitwise
identical fits on these data and was removed.

Exact benchmark: Beta(1, 1)-Binomial product-partition model, per-bin changepoint
probability p scaled with bin width so the expected number of changepoints per kb is
constant (p = 0.01 per 100 bp bin, 4.8 expected a priori). Second benchmark: a 2-state
HMM with binomial emissions (Churchill 1989 style), EM-fitted, posterior switch
probabilities. Figures `results/lambda/figures/lambda_main.png` (100 bp, L=20) and
`lambda_scale.png` (CS hulls by bin width).

### Declared credible sets at 100 bp, L=20 (`results/lambda/w100/declared_cs_annotated.csv`)

| CS (bp, first base of new regime) | top | log BF | size (bins) | GC before -> after | nearest block boundary | in CS? | exact mass in hull | HMM switch mass in hull |
|---|---|---|---|---|---|---|---|---|
| 201-3600 | 2501 | 3.6 | 27 | 0.56 -> 0.56 (small) | head start 191 (5.5 kb away) | no | 1.06 | 0.88 |
| 20601-21700 | 21101 | 103 | 10 | 0.54 -> 0.52 | tail end 22557 (1.5 kb) | no | 0.90 | 0.06 |
| 22501-22700 | 22501 | 513 | 2 | 0.50 -> 0.41 | tail end 22557 / b2 start 22686 (56 bp) | yes | 1.00 | 0.67 |
| 27801-28000 | 27901 | 294 | 2 | 0.41 -> 0.47 | int start 27812 (89 bp) | yes | 0.85 | 0.00 |
| 33001-33900 | 33201 | 34 | 9 | 0.46 -> 0.44 | cIII start 33299 (98 bp) | yes | 0.98 | 0.99 |
| 38301-39400 | 39201 | 57 | 9 | 0.46 -> 0.49 | cro start 38041 (1.2 kb) | no | 0.99 | 1.12 |
| 46401-46900 | 46401 | 27 | 5 | 0.49 -> 0.44 | lysis end 46427 (26 bp) | yes | 0.97 | 0.94 |

Column definitions: "CS" is the hull of the 95% credible set in bp; "top" the column
with the largest alpha; "log BF" the SER log Bayes factor; "GC before -> after" the
fitted GC fraction on either side of the top column; "nearest block boundary" the closest
boundary between the functional blocks in `config.yaml` to the top column and its
distance; "in CS?" whether that boundary falls inside the CS hull (one bin of slack);
"exact mass in hull" the exact posterior's expected number of changepoints inside the
hull; "HMM switch mass in hull" the same for the 2-state HMM.

Four of the seven CSs sit on a functional block boundary to within 100 bp: the end of
the tail genes / start of the b2 region (22.5 kb, the largest jump, GC 0.50 -> 0.41), the
int gene (27.9 kb, 0.41 -> 0.47), the exo-bet-gam / cIII boundary (33.2 kb) and the end
of the lysis genes (46.4 kb). The 39.2 kb CS is inside the O gene, 1.2 kb from the
cro/cII start, and the exact posterior agrees it is a changepoint (mass 0.99). The 21.1
kb CS is inside the tail-fiber region (orf-401 / orf-314, a region of lower GC that ends
before the b2 drop); the exact posterior also places 0.90 of a changepoint there while
the HMM, with only two GC levels, does not resolve it. The 2.5 kb CS is a weak, wide
head-region effect (log BF 3.6, 27 bins) with no gene boundary.

### Where SuSiE and the exact posterior disagree

- The exact posterior (p = 0.01) has k = 9-12 changepoints (mean 10.7, 90% interval 9-12)
  against 7 declared SuSiE CSs. The missing ones are a short pulse inside b2 (GC 0.33 ->
  0.42 -> 0.37 over 24.1-25.0 kb, exact boundary probabilities 0.99 and 0.79), a 22.6 kb
  boundary (0.74) adjacent to the 22.5 kb one, and the ea-region bump at ~31 kb. All are
  short segments or small jumps.
- The 24.1 kb step is not a SuSiE blind spot but an ARD casualty. With the prior variance
  fixed (0.05 or 0.25) instead of estimated, L=20 declares 8 CSs including 24001-24100 at
  log BF 87-118 and a wide tail-region effect (16.6-21.5 kb, log BF 12); with the estimated
  prior variance the 24.1 kb effect's variance collapses to 0 and the 13 spare effects
  are pruned. L=10 with the estimated prior variance declares 8 CSs (it keeps a 31.2 kb
  effect, log BF 66, that L=20 drops, and puts 128 nats on 33.1 kb where L=20 puts 34
  on 33.2 kb): the EB fits at L=10 and L=20 sit in different local optima. The locations
  with log BF above 20 are the same in every variant.
- The 2-state HMM is a different question (which of two GC levels) and answers it with
  9-19 switches depending on bin width; it fires on the 2.5 kb head effect and on the Nin
  region (40-45 kb) where SuSiE and the exact model see nothing.

### Scale dependence (`results/lambda/scale_table.csv`)

| bin width | bins | declared CSs L=5 / 10 / 20 | median CS width (bp), L=20 | max CS width (bp), L=20 | exact E[k] (90% interval) | HMM expected switches |
|---|---|---|---|---|---|---|
| 50 | 970 | 5 / 8 / 7 | 850 | 3350 | 10.7 (9-13) | 19.0 |
| 100 | 485 | 5 / 8 / 7 | 900 | 3400 | 10.7 (9-12) | 16.6 |
| 250 | 194 | 5 / 8 / 7 | 1000 | 3500 | 9.4 (8-11) | 14.0 |
| 500 | 97 | 5 / 8 / 7 | 1000 | 3000 | 9.0 (8-10) | 9.2 |
| 1000 | 49 | 5 / 6 / 6 | 1500 | 3000 | 10.0 (9-12) | 8.7 |

The number and the locations of declared CSs do not change from 50 to 500 bp (the
`lambda_scale.png` hulls line up), and the CS width in bp is roughly constant rather than
proportional to the bin width. The expected scale dependence ("more segments at finer
bins") does not show up for SuSiE here; it does show up for the HMM. At 1000 bp two CSs
merge. Fit time at 50 bp (970 x 969 dense design, L=20) is 82 s.

### Overdispersion and dependence

The GC indicator has lag-1 autocorrelation 0.029 (lags 2, 3, 6: 0.038, 0.024, 0.040),
and the Pearson dispersion of the 100 bp counts around the fitted step function is
1.07 (chi-square over degrees of freedom; 1 is binomial). Both inflate the binomial variance a little, so the CSs are
somewhat too narrow. The calibration below quantifies it.

### Calibration (`results/lambda/calibration/summary.csv`)

Truth = the 7 declared changepoints of the 100 bp, L=20 fit with segment GC fractions
0.36 / 0.40 / 0.43 / 0.48 / 0.50 / 0.50 / 0.53 / 0.58 (so one of the seven, 2.5 kb,
is a jump of 0.03 and barely detectable); 40 replicates per generative model; fits at
L=5 and L=10 (L=10 declares the same CSs as L=20 on the data and is 3x faster).

| generative model | L | declared CSs per fit | false per fit | CS coverage | recall of the 7 | median CS size (bins) | median hull (bins) | median Pearson dispersion of residuals |
|---|---|---|---|---|---|---|---|---|
| binomial (independent bins) | 5 | 5.0 | 0.15 | 0.97 | 0.69 | 4 | 4 | 1.09 |
| binomial | 10 | 6.6 | 0.28 | 0.96 | 0.90 | 5 | 5 | 1.01 |
| beta-binomial, phi = 1.07 | 5 | 5.0 | 0.28 | 0.95 | 0.68 | 4 | 5 | 1.18 |
| beta-binomial | 10 | 6.6 | 0.40 | 0.94 | 0.88 | 5.5 | 6 | 1.08 |
| per-base Markov, lag-1 rho = 0.029 | 5 | 5.0 | 0.15 | 0.97 | 0.69 | 4 | 5 | 1.14 |
| per-base Markov | 10 | 6.7 | 0.28 | 0.96 | 0.91 | 5.5 | 6 | 1.05 |

Column definitions: "declared CSs per fit" is the mean number of components with log
BF >= 2; "false per fit" the mean number of those containing no true changepoint
column; "CS coverage" the fraction of declared CSs containing a true column; "recall"
the fraction of the 7 true changepoints inside some declared CS; "median Pearson
dispersion" the chi-square/df of the simulated counts around the fitted step function
(1 for a binomial truth). The beta-binomial replicates use the dispersion estimated from
the real data (1.07); the Markov replicates match the real lag-1 autocorrelation.

Coverage is nominal under the binomial truth (0.96-0.97) and slips to 0.94-0.95 under
the beta-binomial truth, with 0.1 more false CSs per fit; the real sequence's
autocorrelation (0.03) is too weak to matter. The narrowing is real but small at this
dispersion. Of the 61 false CSs across all cells, 50 have their top within 5 bins (500 bp)
of a true changepoint, 10 are within 6-19 bins, and 1 is far away (77 bins): the "false"
sets are near misses on short segments (two true changepoints 14 bins apart at 21.0 and
22.4 kb) and on the weak 2.5 kb jump, not fabricated locations. Recall at L=5 is capped
at 5/7 by L itself; at L=10 it is 0.88-0.91, the misses being the 0.03 jump.

### Verdict for the lambda example

Binomial SuSiE on 100 bp bins finds 7 changepoints in lambda's GC profile, four of them
within 100 bp of a boundary between functional gene blocks (tail/b2, int, exo-bet-gam/cIII,
end of lysis) and the other three agreed on by the exact Beta-Binomial posterior. The CS
locations and widths are stable from 50 to 500 bp bins (no scale dependence in the
number of SuSiE CSs; the HMM does drift from 9 to 19 switches), and the simulation
calibration is nominal under the binomial model and only slightly short (0.94) under the
measured overdispersion of 1.07. Two honest problems. First, SuSiE under-counts relative
to the exact posterior (7 against 9-12): short pulses and small jumps are missed, and
one of them (24.1 kb, log BF ~100 when the prior variance is fixed) is dropped only
because the empirical-Bayes prior variance collapsed it, so L=10 and L=20 land in
different local optima and the count moves by one between them. Second, the 95% CS of a
weak effect is again a hull plus scattered tail members (the 2.5 kb set has 27 bins).
Report the contiguous block carrying the alpha mass, and treat the number of declared
CSs as a lower bound that depends on L and the prior-variance treatment, not as a
posterior on k. Where SuSiE is strong, it is strong: the 22.5 kb and 27.9 kb jumps have
2-bin (200 bp) credible sets with log BF in the hundreds, and the exact posterior
places 0.85-1.00 of a changepoint inside each.
