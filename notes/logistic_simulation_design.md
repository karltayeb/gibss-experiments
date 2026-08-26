## Description 

We generate simulations to compare the performance of different methods for logistic SuSiE. 

We simulate $\text{AR}_1(n, p, rho)$ and $\text{Bin-AR}_1(n, p, q, rho)$. Where $n$ is the number of observations, $p$ is the number of covariates. $q$ is the density of the matrix, and $\rho$ is the correlation of adjacent features. 


For each design matrix distribution, we tune the effect size so that the marginal univariate regression has target signal strength $T = \mathbb{E}[LRT]$. 

For the Gaussian design we fit n=500, p=256, rho=0.9. For the Binary we fit $\text{Bin-AR}(1000, 256, 0.5, 0.8)$ and $\text{Bin-AR}(10000, 256, 0.05, 0.8)$. For Bin-AR, $\rho$ is the adjacent-column correlation on the observed (binary) scale, mapped internally to the latent Gaussian correlation.

We generate single effect simulations $L^*=1$. For each design matrix, $b_0 = -3, -2, -1$, $T=4, 8, 16, 32$

We also run multi-effect simulations $L^* = 2, 3, 5$, with $b_0 = -2$, sweeping over $T=4, 8, 16, 32$. 
We control the correlation via the gap between causal variables $g$. 
For example $g=10$, $L^*=3$ places causal variables at $10, 20, 30$. 
We set the gap $g = 2, 5, 10, 20$. Because the AR(1) correlation compounds on the latent Gaussian scale, these gaps give causal-to-causal correlations of roughly $\{0.81, 0.59, 0.35, 0.12\}$ (Gaussian) and $\{0.72, 0.57, 0.42, 0.25\}$ (Bin-AR), matched around $g=5$ and spreading apart at $g=20$. At $g=2$ the causals are only marginally resolvable, so treat $L^*=5$ there as a stress case rather than a per-causal coverage measurement.

For each simulation scenario, we also fit a matched null simulation where $\mathbf{b} = 0$.

We run each simulation scenario for 50 replicates. 

## Methods

We compare CAVI, gIBSS, and JJ SuSiE. In all cases, we perform variational inference in $\mathcal{Q}_2$. We center the design matrix. A shared intercept is fit as an independent factor with $q(b_o) in \mathcal{N}$. The prior variance for each single effect is estimated.

For the single effect simulations, we fit each method with $L=1$.
For single effect and multi effect simulations, we fit each method with $L=10$.
For the null simulations we fit each method with $L=1, 10$.


## Details

This section provides clarification to the agent. Not for humans.

+ Write this into one experiment file, you can specify multiple super-collections. 
+ JJ is method="globaljj"
+ you will have to rework the fitting harness. ideally we map closely to fit_glm_susie arguments. 
+ it would be helpful to record the fit time for each fit. 
+ make sure the entire model state is saved for reproducibility.
+ make sure that sparsity is being exploited where it can be. 
+ when it comes to running on midway, make sure that the wall times are tailored to the method-- CAVI is much slower than gIBSS, for example.
+ JJ is automatically Q2. 
+ max_prior_variance = 100.0
+ you should estimate the intercept. specifically, you should use a guassia flat Gaussian prior N(0, 100). and because we are in Q2, i want q(b0) gaussian. mostly important for intercept integration in CAVI. 
+

## Deliverables

+ twogroup_experiments/analysis/logistic_susie_simulations/design.qmd 
    + generate supplementary figure that shows T for different design matrix (rows) and intercepts (columns)).
    + generate a supplementary table that reports the effect sizes used at each (design, intercept) setting. 
    + include two tables listing all the simulation settings. 
+ generate the corresponding simulation experiment. 
