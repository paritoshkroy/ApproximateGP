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

## Point-referenced spatial data
Point-referenced spatial data refers to observations where each data point is associated with a precise location defined by coordinates. The coordinates typically include latitude and longitude for global positioning, easting and northing for local projections, or $(x,y)$--coordinates of a surface. Analyzing point reference data aims to capture variability and correlation in observed phenomena, predict values at unobserved locations, and assess uncertainty. Its applications are widespread in environmental monitoring and geophysical studies. For example, weather stations record temperature, humidity, and air quality at some fixed monitoring sites across a geographical area, and data analysis often aims to obtain a predicted surface of the phenomena by estimating values at unobserved locations.

Statistical modeling of point reference data often assumes that the measurement variable is, theoretically, defined at locations that vary continuously across the domain. Thus, it necessitates specifying a random surface \citep{gelfand2016spatial}. A widely adopted approach involves modeling the surface as a realization of a stochastic process. Gaussian processes provide a practical framework for such modeling, offering a versatile tool for representing the spatial processes that vary continuously across space. The Gaussian processes facilitate straightforward inference and prediction by capturing spatial correlations, interpolating data, modeling variability, and enabling probabilistic inference. In the following, we will explore the application of Gaussian processes in analyzing point-referenced spatial data.

## Modeling point-referenced spatial data using GP
Within this framework, observations over a finite set of locations in a spatial domain are assumed to be partial realizations of a spatial Gaussian process $\{y(\mathbf{s}): \mathbf{s} \in \mathcal{D}\}$ that defined on spatial index $\mathbf{s}$ varying continuously throughout $d$--dimensional domain $\mathcal{D} \in \mathbb{R}^{d}$. Therefore, the joint distribution of measurements at any finite set of locations is assumed to be multivariate normal, and the properties of the multivariate normal distribution ensure closed-form marginal and conditional distributions, leading to straightforward computation for model fitting and prediction.

It assumes that the measurement $y(\mathbf{s})$, at location $\mathbf{s}$, be a generated as

$$
\begin{align}
y(\mathbf{s}) = \mathbf{x}(\mathbf{s})^\prime \boldsymbol{\theta} + z(\mathbf{s}) + \epsilon(\mathbf{s}),
\end{align}
$$

where $\boldsymbol{\theta}$ is a $p$--dimensional vector of coefficient associated with $p$--dimensional vector of covariates, $\mathbf{x}(\mathbf{s})$, including intercept. The component $z(\mathbf{s})$ is assumed to be a zero mean isotropic Gaussian process with marginal variance $\sigma^2$ and correlation function $\rho(\cdot)$ capturing the spatial correlation and ensures that $y(\mathbf{s})$ is defined at every location $\mathbf{s} \in \mathcal{D}$, and $\epsilon(\mathbf{s})$ is assumed to be an independent random measurement error which follows a normal distribution with mean zero and variance $\tau^2$.


### Inference Procedure

Let $\mathbf{y} = (y(\mathbf{s}_1),\ldots, y(\mathbf{s}_n))^\prime$ be the $n$--dimensional vector of data over $n$ locations $\mathbf{s}_1,\ldots,\mathbf{s}_n$ in $d$--dimensional domain $\mathcal{D} \in \mathbb{R}^{d}$. Using marginalization with respect to $\mathbf{z}^\prime = (z(\mathbf{s}_1),\ldots,z(s_n))$, it can be shown that $\mathbf{y}$ is distributed as multivariate normal with

$$
\begin{align}
\mathbb{E}[\mathbf{y}] = \mathbf{X}\boldsymbol{\theta} \qquad \text{and} \qquad \text{Var}[\mathbf{y}] = \sigma^2 \mathbf{B} + \tau^2\mathbf{I},
\end{align}
$$
where $\mathbf{X}$ is a $n \times p$ design matrix based on the vector of covariates $\mathbf{x}(\mathbf{s}_i)$ and $\mathbf{B}$ is the $n$--dimensional correlation matrix whose $(i,j)$th elements is $\mathbf{B}_{ij} = \rho(||\mathbf{s}_i-\mathbf{s}_j||)$.

Under the Bayesian paradigm, model specification is complete after assigning a prior distribution for $\boldsymbol{\beta}$, $\sigma$, $\ell$ and $\tau$. Then, following Bayes' theorem, the joint posterior distribution of $\boldsymbol{\Phi} = \{\boldsymbol{\theta}, \sigma, \ell, \tau\}$ is proportional to

$$
\begin{align}
\pi(\boldsymbol{\Phi} \mid \mathbf{y}) \propto \mathcal{N}\left(\mathbf{y} \mid \mathbf{X}\boldsymbol{\theta}, \mathbf{V}\right) \; \pi(\boldsymbol{\Phi}),
\end{align}
$$

where $\mathbf{V} = \sigma^2 \mathbf{B} + \tau^2\mathbf{I}$ and $\pi(\boldsymbol{\Phi})$ denotes the prior distribution assigned to $\boldsymbol{\Phi}$. In general, the distribution $\pi(\boldsymbol{\Phi} \mid \mathbf{y})$ does not have a closed-form, and Markov chain Monte Carlo (MCMC) sampling methods are commonly employed to approximate this distribution. These methods are straightforward to implement using modern statistical computing platforms such as `BUGS`, `JAGS`, `NIMBLE`, and `Stan`. MCMC methods provide samples from the posterior distribution, which can be used to estimate various summary statistics. Once samples from the posterior distribution are available, predictions to unobserved locations follow straightforwardly.

The described inference procedure utilizes a model that is marginalized with respect to the latent GP by integrating out the latent variable $\mathbf{z}$. This approach allows the model to directly predict the observed responses, which is why it is referred to as a response GP or marginal GP.

## Response GP in Stan


```
data {
  int<lower=0> n;
  int<lower=0> p;
  vector[n] y;
  matrix[n,p] X;
  array[n] vector[2] coords;
  
  vector<lower=0>[p] scale_theta;
  real<lower=0> scale_sigma;
  real<lower=0> scale_tau;
  
  real<lower = 0> a; // Shape parameters in the prior for ell
  real<lower = 0> b; // Scale parameters in the prior for ell
  
}

transformed data{
  
}

parameters {
  vector[p] theta_std;
  real<lower=0> ell;
  real<lower=0> sigma_std;
  real<lower=0> tau_std;
}

transformed parameters{
  vector[p] theta = scale_theta .* theta_std;
  real sigma = scale_sigma * sigma_std;
  real tau = scale_sigma * tau_std;
}

model {
  theta_std ~ std_normal();
  ell ~ inv_gamma(a,b);
  sigma_std ~ std_normal();
  tau_std ~ std_normal();
  vector[n] mu = X*theta;
  matrix[n,n] Sigma = gp_matern32_cov(coords, sigma, ell);
  matrix[n,n] L = cholesky_decompose(add_diag(Sigma, square(tau)));
  y ~ multi_normal_cholesky(mu, L);
}
```

### Spatial Interpolation

One main interest in point-referenced spatial data analysis is obtaining a predicted surface for the process through pointwise prediction. Let $\{\mathbf{s}_1^\star, \ldots, \mathbf{s}_{n^\star}^\star\}$ be a set of $n^\star$ high resoluted grid locations covering $\mathcal{D}$ and suppose that vector of $p$ covariates values $\mathbf{x}(\mathbf{s}^\star)$ at each site is available. To obtain predicted surface along with reporting uncertainty measures, we need posterior predicted distribution of $\mathbf{y}^\star = (y(\mathbf{s}_1^\star),\ldots, y(\mathbf{s}_{n^\star}^\star))^\prime$, conditional on the observed data $\mathbf{y}$. For this purpose, consider the joint vector $(\mathbf{y}^{\star\prime},\mathbf{y}^\prime)$, under a Gaussian process assumption whose distribution is $(n^\star + n)$--dimensional multivariate normal. Consequently, the conditional distribution $\mathbf{y}^\star$ given $\mathbf{y}$ is $n^\star$--dimensional multivariate normal with conditional mean and variance, respectively, given by 

$$
\begin{align}
\mathbb{E}[\mathbf{y}^\star \mid \mathbf{y}] = 
\mathbf{X}^\star\boldsymbol{\theta} + \mathbf{V}^{\text{pred-to-obs}} \mathbf{V}^{-1} (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})\\
\text{and} \text{Var}[\mathbf{y}^\star \mid \mathbf{y}] = \mathbf{V}^\star - \mathbf{V}^{\text{pred-to-obs}} \mathbf{V}^{-1} \mathbf{V}^{\text{obs-to-pred}},
\end{align}
$$

which is used to perform the prediction, where $\mathbf{X}^\star$ is the $n^\star \times p$ design matrix of covariates at prediction locations. The covariance matrix $\mathbf{V}^\star$ is equal to $\sigma^2 \mathbf{B}^\star + \tau^2 \mathbf{I}$, where $\mathbf{B}^\star$ denotes the $n^\star$--dimensional spatial correlation matrix among the prediction locations. The component $\mathbf{V}^{\text{pred-to-obs}}$ is equal to $\sigma^2 \mathbf{B}^{\text{pred-to-obs}}$, where $\mathbf{B}^\text{pred-to-obs}$ denotes the $n^\star \times n$ spatial correlation matrix between prediction and observed locations.

However, the above joint prediction is computationally expensive as the conditional distribution of a multivariate normal distribution of dimension $(n^\star+n)$, computing conditional mean $\mu_{y(\mathbf{s}^\star) \mid \mathbf{y}}$ and variance $\sigma^2_{y(\mathbf{s}^\star) \mid \mathbf{y}}$ involves expensive matrix calculations. In practice, this can be avoided by performing predictions for each unobserved location separately. In that case, at a generic prediction location $\mathbf{s}^\star \in \mathcal{D}$, the posterior predictive distribution of $y(\mathbf{s})$ at $\mathbf{s}^\star$ is given by

$$
\begin{align}
\pi(y(\mathbf{s}^\star) \mid \mathbf{y}) = \int_{\boldsymbol{\Phi}} \mathcal{N}(y(\mathbf{s}^\star) \mid \mathbf{X}\boldsymbol{\theta}, \mathbf{V}) \pi(\boldsymbol{\Phi} \mid \mathbf{y}) \mathrm{d}\boldsymbol{\Phi}
\end{align}
$$

This procedure is known as a univariate prediction, each step of which involves calculating the matrix inversion of order $n$ and is still expensive if $n$ is large.

### Recovery of the Latent Component
One might be interested in the posterior distribution of the latent spatial component $z(\mathbf{s})$. The inference using joint posterior distribution in equation ignores the estimation of the latent vector $\mathbf{z}^\prime = (z(\mathbf{s}_1), \ldots, z(\mathbf{s}_n))$ during model fitting. Nevertheless, we can recover the distribution of vector $\mathbf{z}$ components via composition sampling once samples from the posterior distribution of the parameters are available. Note that the joint posterior distribution of $\mathbf{z}$ is

$$
\begin{align}
\pi(\mathbf{z} \mid \mathbf{y}) &= \int \pi(\boldsymbol{\Phi}, \mathbf{z} \mid \mathbf{y}) \; \mathrm{d} \boldsymbol{\Phi}\\
&= \int \pi(\mathbf{z} \mid \boldsymbol{\Phi}, \mathbf{y}) \; \pi(\boldsymbol{\Phi} \mid \mathbf{y}) \; \mathrm{d} \boldsymbol{\Phi},
\end{align}
$$

and

$$
\begin{align}
\pi(\mathbf{z} \mid \boldsymbol{\Phi}, \mathbf{y})
&\propto \mathcal{N}(\mathbf{z} \mid \mathbf{0}, \sigma^2 \mathbf{B}) \; \mathcal{N}\left(\mathbf{y} \mid \mathbf{X}\boldsymbol{\theta} + \mathbf{z}, \tau^2\mathbf{I}\right)\\
&\propto \exp\left\{-\frac{1}{2\sigma^2} \mathbf{z}^\prime \mathbf{B}^{-1} \mathbf{z}\right\} \; \exp\left\{-\frac{1}{2\tau^2} (\mathbf{y} - \mathbf{X}\boldsymbol{\theta} -\mathbf{z})^\prime (\mathbf{y} - \mathbf{X}\boldsymbol{\theta} - \mathbf{z})\right\} \nonumber\\
&\propto \exp\left\{-\frac{1}{2}\mathbf{z}^\prime \left(\frac{1}{\tau^2} \mathbf{I} + \frac{1}{\sigma^2}\mathbf{B}^{-1} \right) \mathbf{z} - \mathbf{z}^\prime \left(\frac{1}{\tau^2} \mathbf{I}\right) (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})\right\},
\end{align}
$$
which is the kernel of the multivariate normal distribution with mean and covariance, 
$$
\begin{align}
\mathbb{E}[\mathbf{z} \mid \mathbf{y}] &= \left(\dfrac{1}{\tau^2} \mathbf{I} + \dfrac{1}{\sigma^2}\mathbf{B}^{-1} \right)^{-1} \left(\dfrac{1}{\tau^2} \mathbf{I}\right) (\mathbf{y} - \mathbf{X}\boldsymbol{\theta}),\\
\text{and}\; \text{Var}[\mathbf{z} \mid \mathbf{y}] &= \left(\dfrac{1}{\tau^2} \mathbf{I} + \dfrac{1}{\sigma^2}\mathbf{B}^{-1} \right)^{-1},
\end{align}
$$

respectively. Therefore, posterior samples for $\mathbf{z}$ can be obtained by drawing samples from  $\pi(\mathbf{z} \mid \boldsymbol{\Phi}, \mathbf{y})$ one-for-one for each posterior sample of $\boldsymbol{\Phi}$. These are post-MCMC calculations; hence, sampling is not very expensive. Given the posterior samples for $\mathbf{z}$ associated with observed locations and $\boldsymbol{\Phi}$, it is also possible to obtain samples of the distribution of $n^\star$--dimensional vector $\mathbf{z}^\star$ of the values of $z(\mathbf{s})$ at unobserved locations $\mathbf{s}_{1}^\star, \ldots, \mathbf{s}_{n^\star}^\star$ via composition sampling. The procedure involves assuming joint vectors $(\mathbf{z}^{\star\prime}, \mathbf{z}^\prime)$ which follows $(n^\star + n)$--dimensional multivariate normal distribution and conditional distribution of $\mathbf{z}^\star$ given $\mathbf{z}$ is used to draw samples for $\mathbf{z}^\star$. The conditional distribution is $n^\star$--dimensional multivariate normal with mean
$$
\begin{align}
\mathbf{E}[\mathbf{z}^\star \mid \mathbf{z}] = \mathbf{B}^{\text{pred-to-obs}} \mathbf{B}^{-1} \mathbf{z}
\end{align}
$$
and variance 
$$
\begin{align}
\text{Var}[\mathbf{z}^\star \mid \mathbf{z}] = \sigma^2 (\mathbf{B}^\star - \mathbf{B}^{\text{pred-to-obs}} \mathbf{B}^{-1} \mathbf{B}^{\text{obs-to-pred}}).
\end{align}
$$


## Hierarchical representation of the above model
Note that the model above specification is referred to as the marginal or response Gaussian model, and the inference and prediction procedures are outlined based on it. However, this model can be represented hierarchically as follows:

$$
\begin{align}
\mathbf{y} \mid \boldsymbol{\theta}, \mathbf{z} \sim \mathcal{N} \left(\mathbf{X}\boldsymbol{\theta} + \mathbf{z}, \tau^2\mathbf{I}\right),
\end{align}
$$

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{B}).$$

In practice, the response Gaussian process model is often preferred for efficient parameter estimation, as it circumvents the need to estimate the latent vector $\mathbf{z}$ directly. Instead, in a Bayesian analysis, once posterior samples for the parameters are obtained, estimates for $\mathbf{z}$ can be recovered through composition sampling techniques.

## Latent GP in Stan


```
data {
  int<lower=0> n;
  int<lower=0> p;
  vector[n] y;
  matrix[n,p] X;
  array[n] vector[2] coords;
  
  vector<lower=0>[p] scale_theta;
  real<lower=0> scale_sigma;
  real<lower=0> scale_tau;
  
  real<lower=0> a;
  real<lower=0> b;
}

transformed data{
  
}

parameters {
  vector[p] theta_std;
  real<lower=0> ell;
  real<lower=0> sigma_std;
  real<lower=0> tau_std;
  vector[n] noise;
}

transformed parameters{
  vector[p] theta = scale_theta .* theta_std;
  real sigma = scale_sigma * sigma_std;
  real tau = scale_sigma * tau_std;
  //vector[n] z = cholesky_decompose(add_diag(gp_matern32_cov(coords, sigma, ell), 1e-8)) * noise;
  vector[n] z = cholesky_decompose(add_diag(gp_exponential_cov(coords, sigma, ell), 1e-8)) * noise;
  
  //matrix[n,n] Sigma = gp_exponential_cov(coords, sigma, phi);
  //matrix[n,n] V = add_diag(V, 1e-8);
  //matrix[n,n] L = cholesky_decompose(V);
  //vector[n] z = L * noise;
}

model {
  theta_std ~ std_normal();
  ell ~ inv_gamma(a,b);
  sigma_std ~ std_normal();
  tau_std ~ std_normal();
  noise ~ std_normal();
  vector[n] mu = X*theta;
  
  y ~ normal(mu + z, tau);
}

generated quantities{
  
}
```


## Computational complexity in analysing large datasets

The evaluation of the likelihood of a GP-based model requires the inversion of an $n \times n$ covariance matrix, which has a computational complexity of $\mathcal{O}(n^3)$, making it impractical for analyzing large datasets.

