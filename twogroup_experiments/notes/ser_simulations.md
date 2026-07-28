
## Simulation design

There are a few choices to make in the two group simulation. The design matrix, enrichment model, signal, and error combined define a simulation scenario. 

enrichment: the enrichment model decides how each unit is assigned in the twogroup model, for example `ser_b0_{b0}_b_{b}` is an ser enrichment mode with intercept b0 and effect b. if b=0 is a null simulation, in the sense that no covariate in design influences n. 
signal: `loc_{loc}`, `scale_{scale}`
error: `gaussian`, `t_df_{df}`

design, enrichment, signal, error are implemented is functions. we can always implement new functions for new design matrices, enrichment models, signal. the simulations gets hashed to a unique simulation hash for reproducibility.


### Simulation Spec:

This represents the universe of simultions defined by the simulation config.

design: 
- `c4`, 
- `hallmark`, 
- `gaussian_rho_{rho}_n_{n}_p_{p}`, with grid over $\rho=0.01, \dots, 0.99$
- `uniform_rho_{rho}_n_{n}_p_{p}` with grid over $\rho=0.01, \dots, 0.99$

enrich
- `ser_b0_{b0}_b_{b}` we have a grid over $b_{0}=-3, -2, -1, 0$, and $b=-2, -1, 0, 1, 2$. 
signal:
- `loc_{loc}` with loc varying over {grid}
- `scale_{scale}` with varying over {grid}
error:
- `normal`
- `t_df_{df}` t distribution with degrees of freedom. as df -> infty this approaches the normal distribution.

the human readable name is `design=c4__enrichment=ser_b0_-2.00_b_2.00__signal=loc_1.00__error=gaussian`

### Methods

Method specs determine how a model is fit to the data, and what data is available to the model. Oracle models have access to data not available in typical data analysis, e.g. twogroup oracle has oracle f1, logistic_oracle has oracle group membership, etc.

twogroup: oracle, estimate_loc_scale, estimate_loc, estimate_scale
logistic: oracle, threshold_{z_score_threshold},
cox_light: threshold_{z_score_threshold}, threshold is used to right censor data
cox_heavy
linear: estimate_residual_variance, fixed_residual_variance

## Comparisons

Can we estimate `f1`? 


### SER Evaluation

Setting aside the issue of calibration.
### PIP Based analysis

All SER and SuSiE method admit this PIP based analysis. We evaluate the power and For each variable in each simulation we compute the posterior inclusion probability. Variables are ranked by PIP (pooled across simulations) and we show the relationship between power (fraction of all casual variables detected at a PIP threshold) and false discover proportion (fraction of variables exceeding the PIP threshold that are false positive). `pip_power_fdp_plot`

We assess the calibration of the PIPs. PIPs are sorted into 20 bins and we compare the frequency of causal variables in the bin to the average posterior inclusion probability within the bin. `pip_calibration`. 

Question: can we support with a quantitative measure of calibration?

### CS Based analysis

When we simulate data under and SER and fit an SER, a large enough credible set always contains the causal variable. The performance of different methods can be assessed in terms of the resolution and calibration of the credible sets. These two quantities are in tension, small credible sets are not useful if they do not contain the causal variable, and calibration can always be achieved by including a large enough fraction of the variables uniformly at random. Good methods have small, calibrated, credible sets.

#### Unconditional analysis 

These analyses are restricted to simulations where data are simulated under an SER and fit ans SER. 

First we compare CS size across methods. First, without respect to calibration of the CS we plot CS size as a function of coverage. at nominal $\beta$-credible set. We also report the coverage of these credible sets (the fraction of $\%95$ credible sets containing a causal variable). `cs_size_coverage_dot` (this is a new plot, similar to power-coverage-size plots, but here power and coverage coincide). 

Calibration: we plot nominal coverage level on the x axis and empricial coverage on the y axis. `cs_calibration`, `agg_cs_calibration`. 

To disentangle calibration and resolution, for a collection of simulations, we find the size $\hat \beta$ that obtains frequentist coverage and compare the credible set sizes. On the x axis we plot the empirical coverage, on the y axis we plot the average size of the credible sets. Good methods get coverage at small size. `cs_coverage_size` and `agg_cs_coverage_size`. 

% In a similar vein, we plot coverage as a function of credible set size, both measured as the number of variables in the CS and the posterior mass of the CS. On the x axis we plot "size" (number of variables, nominal cs size), on the y axis we plot coverage.

#### Conditional analysis

In practice, signals are filtered before inspecting the credible sets. We consider filtering on the Bayes factor for the SER. Where we need to threshold based on BF we use $\exp(2)$. For this analysis we consider null and non-null simulations. 

First we assess the ability of the BF to discriminate between null and non-null simulations. We show the ROC curves and the empirical CDFs of $\text{BF}_{01}$. 

Among credible sets passing the BF threshold, we assess power, coverage, and size of the selected credible sets as a function of beta. In a three panel plot we have beta on the x axis and power (total fraction of unique causal variables detected in beta-CSs), size (mean fraction of variables included in beta-CSs), and coverage (fraction of reported CSs containing a causal variable). `agg_cs_power_size_coverage_trace` 

Next, 

We can also consider the power of 95% credible sets (now, we count discoveries that do not include the causal variable as a false positive). This distinguishes the ability of the SER to reject the global null versus its ability to identify which signal is causal. 

Nominal levels. `cs_power_size_coverage_dot` we report the power (fraction of causal variables identified in a credible set), coverage, and size of credible sets, conditional on passing threshold. 

We also repeat the calibration analysis above. Here, there is the caveat that not all reported effects contain a causal variable (e.g. if it is a null simulation). The requested coverage may not be achieved if the Bayes factor threshold for inclusion is too permissive. If we want 95% empirical coverage but 5% of simulations are null, we are forced to obtain 100% coverage on the non-null simulations. 

At a given Bayes factor threshold the posterior can trade off resolution and coverage. Ideally we find that the model can identify causal variables with high resolution. 

TODO: descibe all the plots and what information they give use,.


### Comparisons

Here is a key for understanding simulations. 
A = DESIGN, B=ENRICHMENT, C=F1, D=ERROR
uppercase = vary, lowercase=fixed

**abcD:** Error mispecification

*t distributed  t-distributed simulation scenarios. d = 2, 5, 10, 50 degrees of freedom (pick d range to demonstrate impact)*

$D = (2, 5, 10, 50)$

for all settings in product:
$a \in \{  \text{C4}, \text{Hallmark}, \text{Gaussian-AR1}\}$
$b \in \{ (-2, 2) \}$
$c \in \{  N(\mu, 0.1), N(0, \sigma^2)\}$  $\mu, \sigma = 2$ for approximately equal $\mathbb E z^2$ 

label, e.g. 
`000__design=uniform_rho_0.90_n_500_p_100__enrichment=ser_enrich__signal=scale_2.00`


Abcd: n_samples

Increase sample size may exacerbate partial likelihood asymmetry. We investigate that here. We consider the Uniform AR1 with p = 200, 
$A = \text{ Uniform AR-1}$ with $\rho = 0.9$, $p=200$, and $n \in \{ 200, 500, 1000, 2000 \}$
$b = (-2, 2)$
$c \in \{  N(\mu, 0.1), N(0, \sigma^2)\}$ need to pick a value for $\mu=2$ , $\sigma = 2$ 
$d$ = $N(0, 1)$ 

label, e.g. 
`001__design=uniform_rho_0.90_n_500_p_200__enrichment=ser_enrich__signal=scale_2.00`



**abCd:** Signal distribution

*location driven SNR*
*scale driven SNR*

**aBcd:** Enrichment

*none for now*

**Abcd:** Design matrix

*Gaussian markov increase correlation*
*Gaussian markov increase p*
*Gaussian markov increase n*


Options for B:
- Hallmark
- C4
- Gaussian (rho, n, p)


for now, we don't care about this. 

000-A-B-C: supercollection with a t distributed, heteroskedastic, and gaussian collections. error distributions should be chosen to be illustrative.

001-A-B-D: location, increase snr

002-A-B-D: scale, increase snr

003-A-B-D: location, increase sample size

004-A-B-D: scale, increase sample size




- Error model
	- t degrees of freedom
	- heteroskedasticity 
	- gaussian 
- SNR
	- location
	- scale
- 
- location signal (hallmark, c4, uniform_rho_0.9_n_500_p_100, gaussian_rho_0.9_n_500_p_100)
- scale signal (hallmark, c4, uniform_rho_0.9_n_500_p_100, gaussian_rho_0.9_n_500_p_100)
- location versus scale (low, medium, high signal)

- p features
- n samples
- markov correlation (uniform marginals, gaussian marginals)
- equicorrelation (gaussian marginals, uniform marginals)

In this section we highlight the different comparisons we make and summarize what we have learned about inference in the SER model using different data resolutions. 

#### Model mispecification

We consider the impact of model mispecification. There are two types of model mispecification that we consider. First, we consider if the error model is incorrect. We $t$-distributed error with varying degrees of freedom. As the degrees of freedom increase, the error model becomes approximately normal and the model mispecfication is modest. We expect that the performance of the twogroup model, which assumes a homoskedastic normal error model, will be effected, even with oracle knowledge of $f_1$. 

#### Number of features

In the gaussian and uniform correlated variable simulations, we vary the number of features. 

#### Number of samples

In the gaussian and uniform correlated variable simulations, we vary the number of samples. We are particularly interested in the relative performance of different rank based methods. 

#### Location versus scale 

The location models are well approximate by a linear model, the scale models are not. The linear model performs well in the location simulations (in what sense?) and poorly in misspecified setting. Importantly, across many simulation settings we find that cox-heavy has smaller mean cs size at calibration than  



**Cox-light versus logistic**

The advantage of cox-light over logistic is that it retains information about the relative ordering of observations that pass the significance threshold. When the threshold is set too low, the signal is diluted in logistic regression, while cox can still detect association in the ranks. When the threshold is set high, such that all observations passing the threshold are high confidence true positives, there is little difference. Under the generative model considered here the rankings are uniformly sampled permutations.

Overall, cox-light often performs comparably to logistic when the threshold is set high, and is less sensitive to the choice of threshold, offering an attractive alternative to the binarized analysis. 

**Cox-heavy versus cox-light**

Cox heavy and cox-light differ in that they rank the data in ascending and descending order of significance respectively. Cox-heavy does not require the selection of a threshold, but it does require full computation of the partial likelihood over all samples, which may computationally expensive. Cox-heavy can be right-censored. This reduces the computational burden by reducing the number of terms in the partial likelihood. 



---
when signal is loc, linear model of z scores often performs better than the two group model, even though data are generated under twogroup. If signal is scale driven the linear model on z-score fails completely, as expected.

for gaussian design matrix: logistic, cox-light, and cox-heavy all perform similarly. 
 
for uniform design: logistic often needs bigger credible sets to achieve calibration. cox-heavy often has similar or smaller cs size at calibration. this indicates that the 

gene set design matrix: 



when 

002-gaussian-n-features-loc CSs are extremely well resolved. recommendation: decrease mu=sigma=1.0

