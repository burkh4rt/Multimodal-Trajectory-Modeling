## Multimodal Trajectory Modeling

This repository provides code and anonymized data to accompany our paper
"AI-guided early dementia prediction using unsupervised multimodal modeling of
brain health trajectories" [^1]. In it, we propose and validate a mixture of
state space models to perform unsupervised clustering of short trajectories.
Within the state space framework, we let expensive-to-gather biomarkers
correspond to hidden states and readily obtainable cognitive metrics correspond
to measurements. Upon training with expectation maximization, we find that our
clusters stratify persons according to clinical outcome. Furthermore, we can
effectively predict on held-out trajectories using cognitive metrics alone. Our
approach accommodates missing data through model marginalization and
generalizes across research and clinical cohorts.

#### Data format

We consider a training dataset

$$
\mathcal{D} = \{(x_{1:T}^{i}, z_{1:T}^{i}) \}_{1\leq i \leq n_d}
$$

consisting of $`n_d`$ sequences of states and observations paired in time. We
denote the states $`z_{1:T}^{i} = (z_1^i, z_2^i, \dotsc, z_T^i)`$ where
$`z_t^i \in \mathbb{R}^d`$ corresponds to the state at time $`t`$ for the
$`i`$th instance and measurements
$`x_{1:T}^{i} = (x_1^i, x_2^i, \dotsc, x_T^i)`$ where
$`x_t^i \in \mathbb{R}^\ell`$ corresponds to the observation at time $`t`$ for
the $`i`$th instance. For the purposes of this code, we adopt the convention
that collections of time-delineated sequences of vectors will be stored as
3-tensors, where the first dimension spans time $`1\leq t \leq T`$, the second
dimension spans instances $`1\leq i \leq n_d`$ (these will almost always
correspond to an individual or participant), and the third dimension spans the
components of each state or observation vector (and so will have dimension
either $`d`$ or $`\ell`$). We accommodate trajectories of differing lengths by
standardising to the longest available trajectory in a dataset and appending
`np.nan`'s to shorter trajectories.

#### Model specification

We adopt a mixture of state space models for the data:

$$
p(z^i_{1:T}, x^i_{1:T})
		= \sum_{c=1}^{n_c} \pi_{c} \delta_{\{c=c^i\}} \big( p(z_1^i| c)
		\prod_{t=2}^T p(z_t^i | z_{t-1}^i, c) \prod_{t=1}^T p(x_t^i | z_t^i, c) \big)
$$

Each individual $`i`$ is independently assigned to some cluster $`c^i`$ with
probability $`\pi_{c}`$, and then conditional on this cluster assignment, their
initial state $`z_1^i`$ is drawn according to $`p(z_1^i| c)`$, with each
subsequent state $`z_t^i, 2\leq t \leq T`$ being drawn in turn using the
cluster-specific _state model_ $`p(z_t^i | z_{t-1}^i, c)`$, depending on the
previous state. At each point in time, we obtain an observation $`x_t^i`$ from
the cluster-specific _measurement model_ $`p(x_t^i | z_t^i, c)`$, depending on
the current state. In what follows, we assume both the state and measurement
models are stationary for each cluster, i.e. they are independent of $`t`$. In
particular, for a given individual, the relationship between the state and
measurement should not change over time.

In our main framework, we additionally assume that the cluster-specific state
initialisation is Gaussian, i.e. $`p(z_1^i| c) = \eta_d(z_1^i; m_c, S_c)`$, and
the cluster-specific state and measurement models are linear Gaussian, i.e.
$`p(z_t^i | z_{t-1}^i, c) = \eta_d(z_t^i; z_{t-1}^iA_c, \Gamma_c)`$ and
$`p(x_t^i | z_t^i, c) = \eta_\ell(x_t^i; z_t^iH_c, \Lambda_c)`$, where
$`\eta_d(\cdot, \mu, \Sigma)`$ denotes the multivariate $`d`$-dimensional
Gaussian density with mean $`\mu`$ and covariance $`\Sigma`$, yielding:

$$
p(z^i_{1:T}, x^i_{1:T})
		= \sum_{c=1}^{n_c} \pi_{c} \delta_{\{c=c^i\}} \big( \eta_d(z_1^i; m_c, S_c)
		\prod_{t=2}^T \eta_d(z_t^i; z_{t-1}^iA_c, \Gamma_c) \prod_{t=1}^T
		\eta_\ell(x_t^i; z_t^iH_c, \Lambda_c) \big).
$$

In particular, we assume that the variables we are modeling are continuous and
changing over time. When we train a model like the above, we take a dataset
$`\mathcal{D}`$ and an arbitrary set of cluster assignments $`c^i`$ (as these
are also latent/ hidden from us) and iteratively perform M and E steps (from
which EM gets its name):

- (E) Expectation step: given the current model, we assign each data instance
  $`(z^i_{1:T}, x^i_{1:T})`$ to the cluster to which it is mostly likely to
  belong under the current model
- (M) Maximization step: given the current cluster assignments, we compute the
  sample-level cluster assignment probabilities (the $`\pi_c`$) and optimal
  cluster-specific parameters

Optimization completes after a fixed (large) number of steps or when no data
instances change their cluster assignment at a given iteration.

The approach (in [framework](./framework)) uses mixtures the above modeling
formulation for training and inference. A Bayesian version of this approach was
pioneered by Chiappa and Barber[^2]. The EM[^3] approach we take with hard
cluster assignment allows us to extend the above model to to nonlinear
specifications (see [framework_extended](./framework_extended)), with the
tradeoff that certain types of model marginalisation are currently unsupported.

#### To run the code

Model training and predictions are done in Python. Statistical tests on the
results are mostly performed in R (with survival models being a notable
exception). An environment suitable for running this code can be created using
either:

1. a [conda](https://docs.conda.io/en/latest/miniconda.html) environment as
   described in `environment.yml` and an
   [renv](https://cran.r-project.org/web/packages/renv) environment as in
   `renv.lock` —

   ```sh
   conda env create -f environment.yml
   conda activate hydrangea
   R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
   R -e "remotes::install_github('rstudio/renv@v1.0.0')"
   R -e "renv::restore()"
   ```

   or

2. [docker](https://docker.com) —

   ```sh
   docker build -t thistly-cross .
   ```

After choosing one of the above, on a machine with
[Make](https://www.gnu.org/software/make/) installed, and with all required
data in the `data` folder, one can then

```sh
make all
```

to reproduce the figures and analyses related to ADNI. (Due to patient privacy,
we provide the code used to produce the results for MACC here, but not the
data.) If using docker, uncomment line #9 of the [Makefile](Makefile) to define
`docker` prior to running.

#### Adapting the code for your own use

We provide a publicly available package
[`unsupervised-multimodal-trajectory-modeling`](https://pypi.org/project/unsupervised-multimodal-trajectory-modeling/)
on pypi. A template repository containing starter code is available here:
[https://github.com/burkh4rt/Unsupervised-Trajectory-Clustering-Starter](https://github.com/burkh4rt/Unsupervised-Trajectory-Clustering-Starter)

##### Caveats & Troubleshooting

Some efforts have been made to automatically handle edge cases. For a given
training run, if any cluster becomes too small (fewer than 3 members), training
terminates. In order to learn a model, we make assumptions about our training
data as described above. While our approach seems to be robust to some types of
model misspecification, we have encountered training issues with the following
problems:

1. Extreme outliers. An extreme outlier tends to want to form its own cluster
   (and that's problematic). In many cases this may be due to a typo or failed
   data-cleaning (i.e. an upstream problem). Generating histograms of each
   feature is one way to recognize this problem.
2. Discrete / static features. Including discrete data violates our Gaussian
   assumptions. If we learn a cluster where each trajectory has the same value
   for one of the states or observations at a given time step, then we are
   prone to estimating a singular covariance structure for this cluster which
   yields numerical instabilities. Adding a small bit of noise to discrete
   features may remediate numerical instability to some extent.

[^1]:
    M. Burkhart, L. Lee, D. Vaghari, A. Toh, E. Chong, C. Chen, P. Tiňo, and Z.
    Kourtzi, _AI-guided early dementia prediction using unsupervised multimodal
    modeling of brain health trajectories,_ Sci. Rep. 14 (2024)

[^2]:
    S. Chiappa and D. Barber, _Dirichlet Mixtures of Bayesian Linear Gaussian
    State-Space Models: a Variational Approach,_ Tech. rep. 161, Max Planck
    Institute for Biological Cybernetics, 2007.

[^3]:
    A. Dempster, N. Laird, and D. Rubin. _Maximum Likelihood from  
    Incomplete Data via the EM Algorithm._ J. Roy. Stat. Soc. Ser. B (Stat.
    Methodol.) 39 (1977).

<!---

format code with:
```
isort --profile black .
black .
prettier --write --print-width 79 --prose-wrap always **/*.md
R -e 'styler::style_dir("posthoc", transformers = styler::tidyverse_style(strict = TRUE))'
```

-->
