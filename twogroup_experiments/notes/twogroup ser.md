
**SuSiE Prior**

We say that $\mathbf{b} \sim \text{SER}(\pi, \sigma^2)$
$$
\mathbf{b} = b \gamma, \quad \gamma \sim \text{Multinomial}(1, \pi), \quad b \sim N(0, \sigma^2)
$$
We say that $\mathbf{b} \sim \text{SuSiE}(\{\pi_{l}\},  \{ \sigma^2_{l} \})$ when 
$$
	\mathbf{b} = \sum_{l=1}^L \mathbf{b}_{l}, \quad \mathbf{b}_{l} \sim \text{SER}(\pi_{l}, \sigma^2_{l}).
$$
That is, each single effect vector $\mathbf{b}_{l}$ has one non-zero coordinate, whose value is normally distributed. The sum of $L$ single effect vectors has at most $L$ non-zero coordinates. The SuSiE prior is useful in Bayesian variable selection analysis. When the number of coordinates is large compared to $L$, so that the probability of collisions is small, the SuSiE prior is similar to the 
