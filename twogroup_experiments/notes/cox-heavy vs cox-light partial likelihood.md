The behavior of these two models can be quite different because the partial likelihood is not symmetric
$$
\frac{\exp(x_{(t)}^T \beta)}{\sum_{i \ge t} \exp(x_{(i)}^T \beta)}.
$$

**Gradients of the partial likelihood**

$$
L(\beta) = \sum_{t} x_{(t)}^T\beta - \log \sum_{i \ge t} \exp(x_{{(i)}}^T\beta)
$$

With $\eta_{i} = \exp(x_{i}^T \beta)$ and $S_{t} = \sum_{i \ge t} \eta_{(i)}$, $\frac{\partial S_{t}}{\partial \eta_{i}} = \eta_{i} \mathbb 1 \{ i > t \}$, $\frac{{\partial \eta_{i}}}{\partial \beta} = x_{i}$ 

$$
L(\beta) = \sum_{t} \log \eta_{(t)} - \log S_{t}
$$
$$
\nabla_{\beta} L = \sum_{t} x_{(t)} - \sum_{i \ge t} \frac{ \eta_{(i)}}{S_{t}} x_{(i)}
$$

$$
\nabla^2_{\beta} L = -\sum_{t} \sum_{i \ge t} \frac{ \eta_{(i)}}{S_{t}^2} x_{(i)} x_{(i)}^T
$$

Enrichment in cox-heavy results in a negative $\beta_j$ for the enriched feature, causing observations in the enriched class to be less likely to be selected as the ranking is generated. On the one hand, these observations are in the risk set for a greater amount of time. On the other hand, the partial likelihood is somewhat insensitive to perturbations in $-\beta$, as their contribution to the denominator is small.  

An alternative analysis would rank observations by their signed test statistics. Whether using cox-light or cox-heavy it becomes advisable to retain the full ranking. 


uv run snakemake all_plots --rerun-incomplete --until materialize_twogroup_experiment_batch --forcerun materialize_twogroup_experiment_batch -c12


It may be desirable to model the ranks of test statistics rather than modelling the test statistics directly. The cox proportional hazards model depends only on the ranks of the data. We examine the suitability of applying cox ph model to ranking (e.g. of wald statistics). We compare the model that ranks wald statistics from smallest to largest and largest to smallest.


Consider the case where there are two groups: a background group with arrival times that are distributed $\text{Exp}(\lambda_{1})$ and an enriched group that arrives at $\text{Exp}(\lambda_{2})$. Suppose we have $n = n_1 + n_2$ samples,  $n_1$ of the first type and $n_2$ of the second. Then the waiting time to the first even is $n_1 \lambda_{1} + n_{2}\lambda_{2}$. Let $\delta_{i}$ be an indicator for the event that the $i$-th arrival is from the enriched group.
$$
	\mathbb P(\delta_{1} = 1) = \frac{n_{2}\lambda_{2}}{n_{1}\lambda_{1} + n_{2}\lambda_{2}}
$$

This is a two group model. Let $X \sim \text{Exp}(\lambda_1)$ and $Y \sim \text{Exp}(\lambda_2)$,
$$
F_{X}(X) \sim B(1, 1) \quad F_{X}(Y) = B(1, \lambda_{2} / \lambda_{1})
$$

*Proof* TODO

$f_X(x) \propto (1-x)^{\lambda_{1} - 1}$
$$
\mathbb P( X < Y) = \int P( X< y) p(y) dy = 
$$

Claim: Any two group model that satisfies the proportional hazards assumption has a ranking distribution identical to the exponential ranking distribution. 

### Gaussian two group model 

Throughout let $Z$ denote a standard gaussian random variable, the marginal distribution of $z$-scores under the null. $X = \sigma Z + \mu$ is the marginal distribution of $z$-scores under the alternative.

**Ranking by $z$-score satisfies the proportional hazards assumption** 



[[Sensitivity of the cox model to early and late arrivals]]

[[Misspecification of the Gaussian two group model]]