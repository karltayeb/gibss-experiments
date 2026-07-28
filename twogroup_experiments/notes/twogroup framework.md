
## Motivation

Method for gene set analysis often require binary or ranked list of genes. These are often derived from richer gene level test statistics. Reducing the data to their ranks or binary membership causes a loss of information, but also simplifies analysis and makes inference more robust. Primarily, one no longer needs to model, and risk mispecification, of the distribution of gene level test statistics.

Here we present a simple but flexible model for gene level effects, the covariate moderated two group model. True effects are drawn from a two component mixture representing null and non-null effects with distribution $f_0$ and $f_1$ respectively. We observe the true effects corrupted with gaussian noise. We consider the covariate moderated two group model where the prior mixture weights depend on covariates or side information $X$. In particular we consider the setting where there are $p$ candidate covariates, and we assume that there is a single covariate that drives enrichment.

Using this model we explore the tradeoffs between robustness and information loss. 
1. Two group model: correct model but we need to estimate $f_1$. Even if a great deal can be learned about the marginal distribution of $\tilde{f}$, it may still be difficult to separate $f_1$. 
2. Rank based models (cox regression): we consider ranking variables from most to least significant, but also least to most significant, which we call cox-light and cox-heavy respectively. For cox-light we also consider right-censoring the data, that is only observing the ranks for genes exceeding a threshold. 
3. Binary model (logistic regression): We can also binarize the data. Compared to cox-light, this can be seen as interval censored data wherein genes above the threshold are tied and below the threshold are unknown. The censored interpretation is consistent with the idea that $\mathbb 1  \{ z^2 < \tau \}$ only reflects that we failed to reject the null.
4. Additionally, we consider fitting a linear model to the $z$-scores.

Overall our findings are that
- cox-heavy is often a god
- depending on the simulation scenario and choice of threshold, cox-light performs similarly or better than logistic. cox-light is less sensitive to the choice of threshold, a
- logistic regression and the cox methods make a different tradeoff between power and resolution. calibrated logistic SER often reports larger credible sets and obtains higher power, while cox reports smaller credible sets at lower power.
- estimation of the enrichment model with the two group error model can be unreliable. Even with oracle $f_1$. performance is further degraded by estimating $f_1$. 
- 

## Setup

$$
\begin{align}
\hat{\beta}_{i} &= \beta_{i} + \epsilon_{i}, & \beta_{i} | x_{i} \sim  (1 - \pi(x_{i})) \delta_{0} + \pi(x_{i}) f_{1}(\cdot) \\
\end{align}
$$


When you do logistic regression on the binarized $y = \mathbb 1(z^2 \ge \tau$), the estimand changes. Regression coefficients are now interpreted as the log odds of the $z^2$ exceeding threshold $\tau$, rather than. When $f_1$ and $f_0$ are well separated, and the threshold is chosen so that $\mathbb P(z^2\le \tau | H_0) \approx \mathbb P(z^2 \ge \tau | H_{1}) \approx 1$. 


