

**Setup** Consider the simple univariate logistic regression with an intercept
$$
y \sim \text{Bernoulli}(p_{i}), \quad \log \frac{p_i}{1 - p_{i}} = b_{0} + bx
$$
to completely specify our model we need a prior $g(b, b_{0}) = g(b) g(b_{0} | b)$. Deferring the specification of $g(b)$, we let
$$
	g(b_{0} | b) \propto |I_{00}(b_{0}, b)|^{1/2}, \quad I_{00} = -\mathbb{E}_{X, Y} \left[  \frac{\partial^2}{\partial b_{0}^2} \log p(Y | X, b_{0}, b) \right]
$$
While we do not know the population distribution $P_{X, Y}$ we do have a sample of $n$ observations with empirical distribution $\hat{P}_{X, Y}$. The observed information i.e. the negative hessian is a good approximation. Let $H(b_{0}, b) = \nabla^2\ell(b_{0}, b)$.  Suppressing dependence on $b_0$ and $b$, $H_{00} = -\sum p_{i}(1- p_{i})$. And $I_{00} = -\frac{1}{n}H_{00}(1 + O(n^{-1/2}))$. 

Let's marginalize over the intercept. We approximate twice. First by using a Laplace approximation, and second by substituting the observed information for the Fisher information. Alternatively, we could think of using the data-dependent prior, $\hat{g}(b_{0}| b) \propto |H_{00}|^{1/2}$. We assume $P_{X,Y}$ satisfy the regularity conditions so that the relative error statements hold with high probability. 

Let $b_0(b) = \arg\max_{b_{0}} p(b_{0}, b)$ then:

$$
|I_{00}|^{1/2} \cdot |- H_{00}|^{-1/2} = \frac{1}{\sqrt{ n }} (1 + O(n^{-1/2}))
$$
And
$$
\begin{align}
	p(y | X, b) &= \int p(y | X, b, b_{0}) g(b_{0} | b) db_{0} \\
	& = \left[ p(Y | X, b, b_{0}(b)) \cdot g(b_{0}(b) | b) \cdot |-H_{00}(b_{0}(b), b)|^{-1/2}  \cdot \sqrt{ 2\pi } \right] (1 + O(n^{-1})) \\
	& = \left[ p(Y | X, b, b_{0}(b)) \sqrt{ \frac{2\pi}{n} } (1 + O(n^{-1/2})) \right] (1 + O(n^{-1})) \\
	& = \left[ p(Y | X, b, b_{0}(b)) \cdot \sqrt{ \frac{2\pi}{n}} \right] (1 + O(n^{-1/2})).
\end{align}
$$

We have that the marginal likelihood is well approximated (up to a the $\frac{1}{\sqrt{ n }}$ factor, which does not depend on $b$) by the profile likelihood $p(Y | X, b, b_{0}(b))$. 

**Iteratively reweighted least squares with centering** Let $\ell(b_{0}, b) = \log p(Y | X, b_{0}, b)$, 
$$
	J(b_{0}, b) = \ell(b_{0}, b) + \log g(b), \quad f(b) = \ell(b_{0}(b), b) + \log g(b)
$$
Our goal is to find 
$$
	b^* = \arg\max_{b} f(b)
$$

I claim that recentering the covariates at each stage of iteratively reweighted least squares and maximizing over $b$, converges to $b^*$. For simplicity, we will assume that $g(b) = N(b; 0, \sigma^2)$, although I believe the argument goes through for priors with convex, twice continuously differentiable densities. 

At a given stage of IRLS we have $H_{01} = \sum w_{i} x_{i}$ with $w_i = p_i(1-p_i)$. Applying the centering $x_{c} := x - \sum w_{i} x_{i} / \sum w_{i}$ renders the intercept estimate independent of the effect estimate. We have the usual reparameterization of the intercept:
$$
	\beta_{0} + bx_{c} = \beta_{0}  - b\bar{x}_{w}+ bx \implies b_{0} = \beta_{0} - b\bar{x}_{w}
$$
With a slight abuse of notation, let ${J}$ denote the joint log likelihood under the reparameterization. Let $\hat{J}$ denote its Taylor expansion about $\bar{\beta}_{0}, \bar{b}$. Similarly, $\log \hat{g}$ is the Taylor expansion of the prior. 
$$
	\hat{J}(\beta, b) := \ell(\bar{\beta}_{0}, \bar{b}) + \langle \nabla \ell(\bar{\beta}_{0}, \bar{b}),  \Delta \rangle + \frac{1}{2} \langle \Delta,  H \Delta \rangle + \log\hat{g}(b), \quad \Delta = \begin{bmatrix}
	\beta_{0} - \bar{\beta}_{0} \\
	b - \bar{b}
	\end{bmatrix}.
$$
Notice that because the Hessian is diagonal, $\hat{J}$ is separable in $\beta, b$.  The intercept is maximized at
$$
	\hat{\beta}_{0} = \frac{\sum w_{i} z_{i}}{\sum w_{i}},\;  
$$
$\hat{J}$ is quadratic in $b$. The posterior is Gaussian, $\hat{J}$ is maximized at the mode/posterior mean:
$$
{b} | x, z \sim N\left( \tilde{b}, \frac{s^2 \sigma^2}{s^2 + \sigma^2}\right), \quad \tilde{b} = \frac{\sigma^2}{\sigma^2 + s^2} \hat{b}, \quad  \hat{b} = \frac{\sum w_{i}z_{i} x_{i}}{\sum w_{i} x_{i}^2}, \quad s^2 = \frac{1}{\sum w_{i} x_{i}^2}.
$$

At the fixed point, $\tilde{b}$ is the MAP and $b_{0}(\tilde{ b}) = \hat\beta_0 + \tilde{b} \bar{x}_{w}$ is the maximum likelihood estimate of the intercept at the MAP estimate of $b$.  So, centering the data at each stage of IRLS produces the MAP estimate under the profile likelihood
$$
\tilde{b} = \arg\max_{b} \ell(b_{0}(b), b) + \log g(b).
$$

TODO: what if $\log g(b)$ is convex?

**Adaptive quadrature**

We've located the mode of $f$, to integrate over $b$ we use adaptive Gauss-Hermite quadrature. We pick our quadrature rule so that it produces the Laplace approximation when $m=1$. Taking a quadratic approximation of $f$ about $\tilde{b}$ we have: 
$$
\hat{f}(b)= f(\tilde{b}) + \frac{1}{2} f''(\tilde{b}) (b - \tilde{b}), \quad f''(b) =  - \frac{1}{\sigma^2_{MAP}} := -\left( \frac{1}{s^2} + \frac{1}{\sigma^2} \right) .
$$

The quadrature rule looks like:
$$
	\int \exp f(b) = \int {\exp (f(b) - \hat{f}(b))} \cdot \frac{\exp \hat{f}(b)}{\sqrt{  2 \sigma^2_{\text{MAP}} }} db \approx \frac{\exp f(\tilde{ b})}{\sqrt{ 2 \sigma^2_{\text{MAP}} }} \cdot \sum w_{i} \exp(f(b_{i}) - \hat{f}(b_{i})).
$$

Note: to evaluate $f(b_i)$ you need to optimize the intercept. Up to first order:
$$
	b_{0}(b_{i}) \approx \hat{\beta}_{0} - b_{i} \bar{x}_{w}.
$$

This actually corresponds with taking a single newton step for $b_0$. To see this, notice that $\hat{\beta}_{0}$ is a stationary point of $\hat{J}$. A newton step in the $\beta$ parameterization yields $\beta_{0i} = \hat{\beta}_{0}$. Mapping back to the intercept 
$$
	b_{0i} = \hat{\beta}_{0} - b_{i} \bar{x}_{w}.
$$


**Logistic SER** We've explained how to compute the marginal likelihood. Applying this to each variable separately, and to the null model gives you everything you need for inference in the SER.

**Orthogonalization** I should note this seems to be a special case of the parameter orthogonalization described by Cox and Reid in 1987 [https://doi.org/10.1111/j.2517-6161.1987.tb01422.x](https://doi.org/10.1111/j.2517-6161.1987.tb01422.x). This is also the same orthogonalization argument in by Wakefield to justify leaving the nuisance parameters out of the the formulation of the the ABF. I don't understand all the details of this paper, but I think Cox and Reid would have a full understanding of the problem here, so perhaps a better understanding of their work would help me better understand mine.

**Variational approximation** Suppose we wanted to make the variational approximation $q(b_{0}, \mathbf{ b}) = q(b_{0}) q(\mathbf{b})$. This extends more naturally to GIBSS. The weighted centering seems like a reasonable choice. For $X{\bf b}$ near $X\bar{\mathbf{b}}$ we have approximate orthogonality, making it a natural match to the approximation. You could apply this centering once (e.g. using the weights at the null model, which gives "null orthogonality"), or iteratively, which resembles a "global" version of the IRLS, where the likelihood approximation is shared.

**Linear SuSiE** naively applying linear SuSiE to binary response data can be quite disastrous. The main thing to do to avoid disaster is to not estimate the residual variance. Conservatively you can set the residual variance to $1/4$, which is the maximum variance of an observation under the logistic model. Or you can set the residual variance to $\bar{y}(1 - \bar{y})$. At this point you might as well provide the adjusted response. Now that is just one iteration of IRLS from the null model. I think it is the most reasonable proposal for linearizing logistic SuSiE. It does a good job of detecting if there is a signal (to logBF SER gets big), but it does not do a good job of localizing the signal (the credible sets are large).

**Problem?** In simulations, the iteratively re-centered/profiled SER suddenly becomes much more uncertain. How good is the Laplace approximation, and how much error do we introduce by using $H$ in place of $I$? Profile and Or, is there just some bug that become apparent. 

Put another way, using the fixed intercept causes the credible sets to get much smaller, seemingly without hurting coverage too much. The mechanics of this seem straight forward: when you profile out the intercept, the standard error is 
$$
	s_{p} = \left( h_{11} - \frac{h_{10}^2}{h_{00}} \right)^{-1/2}
$$
In contrast, with a fixed intercept, the standard error is simply 
$$
	s_{f} = h_{11}^{-1/2}
$$
When $x$ is highly correlated with the ones vector, $s_p$ becomes large. Cox and Reid say a consequence of orthogonality is "the asymptotic standard error for estimating the parameter of interest is the same whether the nuisance parameter is treated as known or unknown". So it seems that with centering, we (correctly) inflate the standard error, in line with the fact that we've marginalized over the nuisance parameter. This inflation *should* be more sever when $x$ is correlated with the 1s vector. Yet, we find that the the variational approximation, without centering, does not pay much price at all for it's overconfidence. Why?

