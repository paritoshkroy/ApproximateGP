# Introduction

## Gaussian processes
A Gaussian process is a random function $\{z(\mathbf{s}): \mathbf{s} \in \mathcal{D}\}$ defined over a $d$--dimensional surface (domain) $\mathcal{D} \subset \mathbb{R}^d$, any finite number of which have a multivariate normal distribution. Therefore, a Gaussian process can be completely specified by a mean function $\mu(\mathbf{s}) = \mathbb{E}\left[z(\mathbf{s})\right]$ and a covariance function $C(\mathbf{s},\mathbf{s}') = \text{Cov}\left[z(\mathbf{s}),z(\mathbf{s}^\prime)\right]$. The covariance function is commonly specified as the product of a variance parameter and a correlation function, a function of Euclidean distance between locations in $\mathcal{D}$, that is, $C(\mathbf{s},\mathbf{s}') = \sigma^2 \rho(\mathbf{s}, \mathbf{s}^\prime)$, where $\sigma$ is the marginal standard deviation and $\rho(\mathbf{s}, \mathbf{s}^\prime)$ is a valid correlation function that depends on the Euclidean distance between $\mathbf{s}$ and $\mathbf{s}^\prime$ in $\mathcal{D}$. The resulting Gaussian process is stationary and isotropic \citep{cressie1993statistics, banerjee2015hierarchical}, and is also called homogeneous Gaussian process \citep{banerjee2015hierarchical, schmidt2020flexible}.

A common example of a valid covariance function is the Mat\'ern family of isotropic covariance functions, which is given by
$$
\begin{align}
\label{Chap2:Maternfunction}
\rho(r, \nu, \ell) = \dfrac{2^{1-\nu}}{\Gamma (\nu)}\left( \sqrt{2\nu}\, \dfrac{r}{\ell} \right)^{\nu} \, K_{\nu} \left(\sqrt{2\nu} \, \dfrac{r}{\ell}\right),
\end{align}
$$
where $r = ||\mathbf{s}-\mathbf{s}^\prime||$ is the distance between any two locations $\mathbf{s}$ and $\mathbf{s}^\prime$ in $\mathcal{D}$. The parameter $\nu>0$ controls the differentiability of the process, and $\ell$ is called lengthscale, which measures the distance at which the process's fluctuations begin to repeat. A smaller lengthscale results in a more oscillatory function with rapid changes, capturing fine details in the data, whereas a larger lengthscale produces a smoother function with gradual changes, averaging out smaller fluctuations. In this sense, the lengthscale of a Gaussian process also serves as a measure of the smoothness or roughness of the functions it generates, impacting how it captures patterns and variations in the data. The component $K_{\nu}(\cdot)$ in the covariance function is a modified Bessel function of the second kind of order $\nu$. The process becomes very rough for a small value of $\nu$ (say, $\nu = 1/2$), whereas $\nu = 3/2$ and $\nu = 5/2$ are appealing cases for which the processes are once and twice differentiable, respectively, and for $\nu > 7/2$ the process is very smooth. Also, note that for $\nu \in \{1/2, 3/2, 5/2\}$ the resultant covariance function has a computationally simple form; however, for $\nu \notin \{1/2, 3/2, 5/2\}$ it is necessary to compute $K_{\nu}(\cdot)$, which is computationally expensive. In practice, the analyst may prefer to fix the parameter $\nu$ based on subject knowledge.

The lengthscale parameter is closely related to the practical range in spatial statistics, indicating the distance over which spatial dependence between observations remains effective. Beyond this range, observations are considered nearly independent, with correlations diminishing to negligible levels. The practical range is crucial for determining appropriate spatial scales for analysis and modeling, ensuring accurate representation of spatial processes. For instance, with $\nu = 3/2$, the Mat\'ern 3/2 correlation function is 
$$
\begin{align}
C_{3/2}(r, \ell) = (1 + \dfrac{\sqrt{3}\, r}{\ell}) \exp\left(-\dfrac{\sqrt{3}\,r}{\ell}\right),
\end{align}
$$
indicating how spatial correlation decreases with distance, and for distances $\ell/2$, $\ell$, $2\ell$, $2.75\ell$, and $4\ell$, the correlations are approximately 0.78, 0.48, 0.14, 0.05, and 0.008, respectively.

By definition, for any finite set of locations $\{\mathbf{s}_1,\ldots,\mathbf{s}_n\} \in \mathcal{D}$ the joint distribution of an $n$--dimensional vector of possible realizations $\mathbf{z}^\prime = (z(\mathbf{s}_1),\ldots,z(\mathbf{s}_n))$ from a Gaussian process follows a multivariate normal distribution, that is,
$$
\begin{align}
f(\mathbf{z} \mid \boldsymbol{\theta}) \propto \dfrac{1}{\sqrt{|\sigma^2\mathbf{B}|}} \exp\left\{-\dfrac{1}{2\sigma^2} (\mathbf{z} - \boldsymbol{\mu})^\prime \mathbf{B}^{-1}(\mathbf{z} - \boldsymbol{\mu})\right\},
\end{align}
$$
where $\boldsymbol{\mu}^\prime = (\mu(\mathbf{s}_1),\ldots,\mu(\mathbf{s}_n))$ is a $n$--dimensional vector of means, and $\mathbf{B}$ is a $n$--dimensional correlation matrix with $(i,j)$th element is $\rho(\mathbf{s}_i,\mathbf{s}_j)$.

