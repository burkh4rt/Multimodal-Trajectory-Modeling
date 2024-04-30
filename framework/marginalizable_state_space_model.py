#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
We explicitly derive the joint distribution of a linear Gaussian
latent state space model.

The latent process
Z[1], Z[2], ..., Z[T]
in ℝˡ is governed by the state model
Z[i] | Z[i-1] ~ N(Z[i-1]*A, Γ) for i = 2, ..., T
with initialisation Z[1] ~ N(m, S)
and the observed latent states
X[1], X[2], ..., X[T]
in ℝᵈ are generated using the measurement model
X[i] | Z[i] ~ N(Z[i]*H, Λ) for i = 1, ..., T

As the resulting joint distribution (Z[1], ..., Z[T]; X[1], ..., X[T])
is multivariate Gaussian, this enables us to calculate marginal
distributions when we encounter hidden variables or missing data.
"""

import warnings

import numba as nb
import numpy as np
import scipy.stats as sp_stats

warnings.simplefilter("ignore")


@nb.jit(
    nb.float64[:, :](
        nb.int64,
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
    nopython=True,
    parallel=True,
)
def _CZZii(i: int, S: np.array, A: np.array, Γ: np.array) -> np.array:
    """Covariance matrix for the ith hidden random variable (1-indexed)

    Parameters
    ----------
    i
        index 1<=i<=T
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    Var(Z[i])
        d×d symmetric positive semi-definite matrix

    """
    if i == 1:
        return S
    return Γ + A.T @ _CZZii(i - 1, S, A, Γ) @ A


@nb.jit(
    nb.float64[:, :](
        nb.int64,
        nb.int64,
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
    nopython=True,
    parallel=True,
)
def _CZZij(i: int, j: int, S: np.array, A: np.array, Γ: np.array) -> np.array:
    """Covariance between the ith and jth hidden random variables (1-indexed)

    Parameters
    ----------
    i
        index 1<=i<=T
    j
        index 1<=j<=T
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    Cov(Z[i], Z[j])
        d×d matrix

    """
    if i == j:
        return _CZZii(i, S, A, Γ)
    elif j > i:
        return _CZZii(i, S, A, Γ) @ np.linalg.matrix_power(A, j - i)
    else:  # j < i
        return _CZZij(j, i, S, A, Γ).T


def CZZ(T: int, S: np.array, A: np.array, Γ: np.array) -> np.array:
    """Covariance for the full hidden autoregressive process

    Parameters
    ----------
    T
        integer number of time steps
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    Var(Z)
        dT×dT matrix

    """
    return np.block(
        [
            [_CZZij(i, j, S, A, Γ) for j in range(1, T + 1)]
            for i in range(1, T + 1)
        ]
    )


def _CZX(
    T: int, S: np.array, A: np.array, Γ: np.array, H: np.array
) -> np.array:
    """Covariance between the full hidden and observed processes
    Z & X, respectively

    Parameters
    ----------
    T
        integer number of time steps
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients

    Returns
    -------
    Cov(X, Z)
        dT×ℓT matrix

    """
    return np.block(
        [
            [_CZZij(i, j, S, A, Γ) @ H for j in range(1, T + 1)]
            for i in range(1, T + 1)
        ]
    )


@nb.jit(
    nb.float64[:, :](
        nb.int64,
        nb.int64,
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
    nopython=True,
    parallel=True,
)
def _CXXij(
    i: int,
    j: int,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """Covariance between the ith and jth observed random variables
    (1-indexed)

    Parameters
    ----------
    i
        index 1<=i<=T
    j
        index 1<=j<=T
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    Cov(X[i], X[j])
        ℓ×ℓ matrix

    """
    if i == j:
        return Λ + H.T @ _CZZii(i, S, A, Γ) @ H
    elif j > i:  # i!=j
        return H.T @ _CZZij(i, j, S, A, Γ) @ H
    else:
        return _CXXij(j, i, S, A, Γ, H, Λ).T


def CXX(
    T: int, S: np.array, A: np.array, Γ: np.array, H: np.array, Λ: np.array
) -> np.array:
    """Covariance over all observed random variables

    Parameters
    ----------
    T
        integer number of time steps
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    Var(X)
        ℓT×ℓT matrix

    """
    return np.block(
        [
            [_CXXij(i, j, S, A, Γ, H, Λ) for j in range(1, T + 1)]
            for i in range(1, T + 1)
        ]
    )


def CC(
    T: int, S: np.array, A: np.array, Γ: np.array, H: np.array, Λ: np.array
) -> np.array:
    """Full covariance matrix for the joint distribution (Z,X)

    Parameters
    ----------
    T
        integer number of time steps
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    Var([Z,X])
        (d+ℓ)T×(d+ℓ)T matrix

    """
    S, A, Γ, H, Λ = map(np.atleast_2d, (S, A, Γ, H, Λ))
    return np.block(
        [
            [CZZ(T, S, A, Γ), _CZX(T, S, A, Γ, H)],
            [_CZX(T, S, A, Γ, H).T, CXX(T, S, A, Γ, H, Λ)],
        ]
    )


def mmZ(T: int, m: np.array, A: np.array) -> np.array:
    """Full mean vector for the latent process Z

    Parameters
    ----------
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    A
        state space model coefficients

    Returns
    -------
    𝔼(Z)
        dT-length vector

    """
    return np.hstack(
        [m @ np.linalg.matrix_power(A, i) for i in range(T)]
    ).ravel()


def mmX(T: int, m: np.array, A: np.array, H: np.array) -> np.array:
    """Full mean vector for the observed process X

    Parameters
    ----------
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    A
        state space model coefficients
    H
        measurement model coefficients

    Returns
    -------
    𝔼(X)
        ℓT-length vector

    """
    return np.hstack(
        [[m @ np.linalg.matrix_power(A, i) @ H] for i in range(T)]
    ).ravel()


def mm(T: int, m: np.array, A: np.array, H: np.array) -> np.array:
    """Full mean vector for the joint distribution (Z, X)

    Parameters
    ----------
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    A
        state space model coefficients
    H
        measurement model coefficients

    Returns
    -------
    𝔼([Z,X])
        (d+ℓ)T-length vector

    """
    A, H = map(np.atleast_2d, (A, H))
    m = np.atleast_1d(m)
    return np.hstack([mmZ(T, m, A), mmX(T, m, A, H)]).ravel()


def full_log_prob(
    z: np.array,
    x: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """log of joint distribution function (p.d.f.) for (Z,X) calculated using
    our analytically calculated mean and variance functions

    Parameters
    ----------
    z
        T×n×d array of hidden states where
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    x
        T×n×ℓ array of observed variables  where
            T length of trajectories
            n number of trajectories
            ℓ dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    n-dimensional vector
        log of joint (Gaussian) distribution of the (Z,X) evaluated at (z,x)

    See Also
    --------
    mm
        the mean function we use here
    CC
        the covariance function we use here

    """
    z, x = map(np.atleast_3d, (z, x))
    return sp_stats.multivariate_normal(
        mean=mm(T, m, A, H),
        cov=CC(T, S, A, Γ, H, Λ),
        allow_singular=True,
    ).logpdf(np.hstack((*z[:], *x[:])))


def composite_log_prob(
    z: np.array,
    x: np.array,
    T: int,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """log of joint distribution function (p.d.f.) for (Z,X) calculated using
    the generation process

    Parameters
    ----------
    z
        T×n×d array of hidden states where
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    x
        T×n×ℓ array of observed variables where
            T length of trajectories
            n number of trajectories
            ℓ dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    n-dimensional vector
        log of joint (Gaussian) distribution of (Z,X) evaluated at (z,x)

    See Also
    --------
    full_log_prob
        a different way to calculate this quantity

    """
    z, x = map(np.atleast_3d, (z, x))
    S, A, Γ, H, Λ = map(np.atleast_2d, (S, A, Γ, H, Λ))
    log_likelihoods = sp_stats.multivariate_normal(
        mean=m,
        cov=S,
        allow_singular=True,
    ).logpdf(z[0, :, :])
    for t in range(T - 1):
        log_likelihoods += sp_stats.multivariate_normal(
            cov=Γ, allow_singular=True
        ).logpdf(z[t + 1, :, :] - z[t, :, :] @ A)
    for t in range(T):
        log_likelihoods += sp_stats.multivariate_normal(
            cov=Λ, allow_singular=True
        ).logpdf(x[t, :, :] - z[t, :, :] @ H)
    return log_likelihoods


def hidden_log_prob(
    z: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
) -> np.array:
    """log of distribution of Z evaluated at z evaluated using the calculations
    we've made

    Parameters
    ----------
    z
        T×n×d array of hidden states
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    n-dimensional vector
        log of (Gaussian) distribution of Z evaluated at z

    """
    z = np.atleast_3d(z)
    S, A, Γ = map(np.atleast_2d, (S, A, Γ))
    return sp_stats.multivariate_normal(
        mean=mmZ(T, m, A),
        cov=CZZ(T, S, A, Γ),
        allow_singular=True,
    ).logpdf(np.hstack(z[:]))


def composite_hidden_log_prob(
    z: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
) -> np.array:
    """log of distribution of Z evaluated at z calculated using the generation
    process

    Parameters
    ----------
    z
        T×n×d array of hidden states where
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    n-dimensional vector
        of the log of (Gaussian) distribution of Z evaluated at z

    See Also
    --------
    hidden_log_prob
        different way to calculate this quantity

    """
    z = np.atleast_3d(z)
    S, A, Γ = map(np.atleast_2d, (S, A, Γ))
    log_likelihoods = sp_stats.multivariate_normal(
        mean=m,
        cov=S,
        allow_singular=True,
    ).logpdf(z[0, :, :])
    for t in range(T - 1):
        log_likelihoods += sp_stats.multivariate_normal(
            cov=Γ, allow_singular=True
        ).logpdf(z[t + 1, :, :] - z[t, :, :] @ A)
    return log_likelihoods


def observed_log_prob(
    x: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """log of joint distribution function (p.d.f.) for X calculated using
    our analytically calculated mean and variance functions

    Parameters
    ----------
    x
        T×n×ℓ array of observed variables where
            T length of trajectories
            n number of trajectories
            ℓ dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
        n-dimensional vector of the log of joint (Gaussian) distribution of
        X evaluated at x

    See Also
    --------
    mmX
        the mean function we use here
    CXX
        the covariance function we use here

    """
    x = np.atleast_3d(x)
    S, A, Γ, H, Λ = map(np.atleast_2d, (S, A, Γ, H, Λ))
    return sp_stats.multivariate_normal(
        mean=mmX(T, m, A, H),
        cov=CXX(T, S, A, Γ, H, Λ),
        allow_singular=True,
    ).logpdf(np.hstack(x[:]))


def full_marginalizable_log_prob(
    z: np.array,
    x: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """log of joint distribution function (p.d.f.) for (Z,X),
    marginalising over missing dimensions of the data

    Parameters
    ----------
    z
        T×n×d array of hidden states, potentially including np.nan's, where
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    x
        T×n×ℓ array of observed variables, potentially including np.nan's,
        where
            T length of trajectories
            n number of trajectories
            ℓ dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    log of joint (Gaussian) distribution of (Z,X) evaluated at (z,x)
        non-finite dimensions have been marginalized out

    See Also
    --------
    mm
        the mean function we use here
    CC
        the covariance function we use here

    """
    z, x = map(np.atleast_3d, (z, x))
    zx = np.ma.masked_invalid(np.hstack((*z[:], *x[:])))
    p = np.zeros(zx.shape[0])

    # compute mean and covariance once
    mean = mm(T, m, A, H)
    cov = CC(T, S, A, Γ, H, Λ)

    # loop over data and calculate for each instance
    for i in range(zx.shape[0]):
        zx_i = zx[i, :]
        p[i] = sp_stats.multivariate_normal(
            mean=mean[~zx_i.mask],
            cov=cov[~zx_i.mask, :][:, ~zx_i.mask],
            allow_singular=True,
        ).logpdf(zx_i.compressed())
    return p


@nb.guvectorize(
    [
        (
            nb.float64[:, :],
            nb.float64[:],
            nb.float64[:, :],
            nb.float64[:],
        )
    ],
    "(n,d),(d),(d,d)->(n)",
    nopython=True,
    # fastmath=True,
)
def multivariate_normal_log_likelihood(
    x: np.array, μ: np.array, Σ: np.array, p: np.array
):
    """computes the log likelihood of a multivariate N(μ,Σ) distribution
    evaluated at the rows of x and assigns it to p;
    if x[i], 1<=i<=n, contains np.nan's or np.inf's for some elements
    (anything that is not finite), it marginalizes these out of the computation

    Parameters
    ----------
    x
        n×d-dimensional matrix containing rows of observations
    μ
        d-dimensional vector
    Σ
        d×d-dimensional covariance matrix
    p
        n-dimensional vector containing the log-likelihoods of interest

    Returns
    -------
        assigns p[i] to be the log likelihood of X~N(μ,Σ) at X=x[i,:]

    """
    x = np.atleast_2d(x)
    Σ = np.atleast_2d(Σ)

    for i in range(p.size):
        m = x[i, :].ravel() - μ.ravel()
        idx = np.argwhere(np.isfinite(m)).ravel()
        p[i] = -0.5 * np.log(
            (2 * np.pi) ** idx.size * np.linalg.det(Σ[idx, :][:, idx])
        ) - 0.5 * m[idx] @ np.linalg.solve(Σ[idx, :][:, idx], m[idx])


def sample_trajectory(
    n: int,
    T: int,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[np.array, np.array]:
    """Given model parameters, this function creates n samples of (Z,X)

    Parameters
    ----------
    n
        integer number of samples to create
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance
    rng
        random number generator

    Returns
    -------
    a tuple (z,x)
        n samples from from the joint distribution of (Z,X)
        with provided parameters
        z has shape T×n×d
        x has shape T×n×ℓ
    """
    S, A, Γ, H, Λ = map(np.atleast_2d, (S, A, Γ, H, Λ))

    z = np.zeros(shape=(T, n, m.shape[0]))
    x = np.zeros(shape=(T, n, H.shape[1]))

    z[0, :, :] = sp_stats.multivariate_normal(mean=m, cov=S).rvs(
        size=n, random_state=rng
    )
    x[0, :, :] = z[0, :, :] @ H + sp_stats.multivariate_normal(cov=Λ).rvs(
        size=n, random_state=rng
    )
    for t in range(T - 1):
        z[t + 1, :, :] = z[t, :, :] @ A + sp_stats.multivariate_normal(
            cov=Γ,
        ).rvs(size=n, random_state=rng)
        x[t + 1, :, :] = z[t + 1, :, :] @ H + sp_stats.multivariate_normal(
            cov=Λ
        ).rvs(size=n, random_state=rng)
    return z, x


def sample_nonlinear_nongaussian_trajectory(
    n: int,
    dz: int,
    dx: int,
    T: int,
    m: callable,
    f: callable,
    Γ: callable,
    h: callable,
    Λ: callable,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[np.array, np.array]:
    """Given model parameters, this function creates n samples of (Z,X)

    Parameters
    ----------
    n
        integer number of samples to create
    dz
        dimensionality of hidden / latent states
    dx
        dimensionality of measurements / observations
    T
        integer number of time steps
    m
        sampler for initial hidden random variable (taking size, rng as input)
        e.g. lambda size, rng: sp_stats.multivariate_normal(cov=Λ).rvs(
            size=size, random_state=rng
        )
    f
        function for state space model
        e.g. lambda z: z ** 2
    Γ
        noise part of state space model (taking size, rng as input)
    h
        function for measurement model
        e.g. lambda z: np.sin(z)
    Λ
        noise part of measurement model (taking size, rng as input)
    rng
        random number generator

    Returns
    -------
    a tuple (z,x)
        n samples from from the joint distribution of (Z,X)
        with provided parameters
        z has shape T×n×d
        x has shape T×n×ℓ
    """

    z = np.zeros(shape=(T, n, dz))
    x = np.zeros(shape=(T, n, dx))

    z[0, :, :] = m(n, rng)
    x[0, :, :] = np.apply_along_axis(func1d=h, axis=-1, arr=z[0, :, :]) + Λ(
        n, rng
    )

    for t in range(T - 1):
        z[t + 1, :, :] = np.apply_along_axis(
            func1d=f, axis=-1, arr=z[t, :, :]
        ) + Γ(n, rng)
        x[t + 1, :, :] = np.apply_along_axis(
            func1d=h, axis=-1, arr=z[t + 1, :, :]
        ) + Λ(n, rng)
    return z, x


def marginalizable_gaussian_log_prob(
    x: np.array, μ: np.array = None, Σ: np.array = None
):
    """gaussian log probability that marginalizes over np.nan values

    Parameters
    ----------
    x
        n×d-dimensional matrix where rows are observations
    μ
        d-dimensional vector; defaults to the zero vector
    Σ
        d×d-dimensional covariance matrix; defaults to the identity

    Returns
    -------
    log_probs
        n-dimensional vector of log(η(x[i];μ,Σ)) indexed over the rows i
    """
    x = np.atleast_2d(x)
    n, d = x.shape
    if μ is None:
        μ = np.zeros(shape=d)
    if Σ is None:
        Σ = np.eye(d)
    else:
        Σ = np.atleast_2d(Σ)
    xm = np.ma.masked_invalid(x)
    p = np.zeros(n)
    for i in range(n):
        p[i] = sp_stats.multivariate_normal(
            mean=μ[~xm[i].mask],
            cov=Σ[~xm[i].mask, :][:, ~xm[i].mask],
            allow_singular=True,
        ).logpdf(xm[i].compressed())
    return p


# run some tests if called as a script
if __name__ == "__main__":
    print("Running tests...")

    # make reproducible
    np.random.seed(42)
    rng = np.random.default_rng(42)

    n = 100000
    T = 10
    d_hidden = 5  # d
    d_observed = 3  # ℓ

    # randomly initialize model coefficients
    A = rng.normal(scale=0.5, size=(d_hidden, d_hidden))
    Γ = np.eye(d_hidden) / 2.0
    H = rng.normal(size=(d_hidden, d_observed))
    Λ = np.eye(d_observed) / 3.0
    m = rng.normal(size=(d_hidden))
    S = np.eye(d_hidden) / 5.0

    # generate a sample of n trajectories, each of length T
    z, x = sample_trajectory(n, T, m, S, A, Γ, H, Λ)

    # CC() is a positive definite, symmetric matrix
    assert np.alltrue(np.linalg.eig(CC(T, S, A, Γ, H, Λ))[0] > 0)
    assert np.allclose(CC(T, S, A, Γ, H, Λ), CC(T, S, A, Γ, H, Λ).T)
    print("CC() is a valid covariance matrix")

    # the average over the n samples should be close to the analytically
    # calculated mean
    assert np.allclose(
        np.hstack((*z.mean(axis=1)[:], *x.mean(axis=1)[:])),
        mm(T, m, A, H),
        rtol=0.03,
        atol=0.03,
    )
    print("The empirical and analytic estimates for 𝔼([Z,X]) are close")

    assert np.allclose(
        np.cov(np.hstack((*z[:], *x[:])), rowvar=False),
        CC(T, S, A, Γ, H, Λ),
        rtol=0.05,
        atol=0.05,
    )
    print("The empirical and analytic estimates for Var([Z,X]) are close")

    assert np.allclose(
        hidden_log_prob(z, T, m, S, A, Γ),
        composite_hidden_log_prob(z, T, m, S, A, Γ),
    )
    print("The analytic and composite calculations for p(Z=z) are close")

    assert np.allclose(
        full_log_prob(z, x, T, m, S, A, Γ, H, Λ),
        composite_log_prob(z, x, T, m, S, A, Γ, H, Λ),
    )
    print("The analytic and composite calculations for p(Z=z,X=x) are close")

    assert np.allclose(
        full_log_prob(z[:, :100, :], x[:, :100, :], T, m, S, A, Γ, H, Λ),
        full_marginalizable_log_prob(
            z[:, :100, :], x[:, :100, :], T, m, S, A, Γ, H, Λ
        ),
    )

    print(
        "The analytic and marginalizable distributions agree "
        "on fully available data"
    )

    z[1, 0, :] = z[3, 0, :] = z[5, 0, :] = np.nan
    x[2, 0, :] = x[4, 0, :] = x[6, 0, :] = np.nan
    assert np.isfinite(
        full_marginalizable_log_prob(
            z[:, 0:1, :], x[:, 0:1, :], T, m, S, A, Γ, H, Λ
        ).ravel()[0]
    )
    print("The marginalizable distribution works with nan's")

    p = multivariate_normal_log_likelihood(
        z[0, :, :], m, S, np.zeros_like(z[0, :, 0])
    )
    assert np.allclose(
        p,
        np.array(
            sp_stats.multivariate_normal(
                mean=m,
                cov=S,
            ).logpdf(z[0, :, :])
        ),
    )

    print("Our multivariate normal log-pdf agrees with the standard")

    Ξ = np.diag(range(1, 4))
    ζ = sp_stats.multivariate_normal(cov=Ξ).rvs(size=4, random_state=rng)
    ζ[0, 1] = ζ[1, 2] = ζ[3, 2] = np.nan
    π = multivariate_normal_log_likelihood(
        ζ, np.zeros(3), Ξ, np.zeros_like(ζ[:, 0])
    )
    assert np.isclose(
        π[0],
        sp_stats.multivariate_normal(
            mean=np.zeros(2),
            cov=np.diag([1, 3]),
        ).logpdf(np.ma.masked_invalid(ζ[0]).compressed()),
    )

    print("Marginalisation strategy appears consistent")

    # train initial state model, state transition model, and measurement model
    # using kernel density estimation

    import statsmodels.api as sm

    long = False
    state_init_mdl = sm.nonparametric.KDEMultivariate(
        data=z[0, -100:, :],
        var_type="c" * z.shape[-1],
        bw="cv_ml" if long else "normal_reference",
    )
    state_mdl = sm.nonparametric.KDEMultivariateConditional(
        endog=np.row_stack([*z[1:, -100:, :]]),
        exog=np.row_stack([*z[:-1, -100:, :]]),
        dep_type="c" * z.shape[-1],
        indep_type="c" * z.shape[-1],
        bw="cv_ml" if long else "normal_reference",
    )
    measurement_mdl = sm.nonparametric.KDEMultivariateConditional(
        endog=np.row_stack([*x[:, -100:, :]]),
        exog=np.row_stack([*z[:, -100:, :]]),
        dep_type="c" * x.shape[-1],
        indep_type="c" * z.shape[-1],
        bw="cv_ml" if long else "normal_reference",
    )

    log_prob_kde = np.log(state_init_mdl.pdf(z[0, :1000, :]))
    for t in range(T - 1):
        log_prob_kde += np.log(
            state_mdl.pdf(
                endog_predict=z[t + 1, :1000, :], exog_predict=z[t, :1000, :]
            )
        )
    for t in range(T):
        log_prob_kde += np.log(
            measurement_mdl.pdf(
                endog_predict=x[t, :1000, :], exog_predict=z[t, :1000, :]
            )
        )

    assert (
        sm.OLS(
            full_log_prob(
                z[:, 1:1000, :], x[:, 1:1000, :], T, m, S, A, Γ, H, Λ
            ),
            log_prob_kde[1:],
        )
        .fit()
        .rsquared
        > 0.99
    )

    print("KDE prototyping completed")

    w2 = sp_stats.multivariate_normal().rvs(size=(10, 2), random_state=rng)
    w3 = np.column_stack((w2, np.nan * np.ones(shape=(10,))))
    w4 = np.column_stack((w3, np.nan * np.ones(shape=(10,))))

    np.testing.assert_allclose(
        marginalizable_gaussian_log_prob(w2),
        marginalizable_gaussian_log_prob(w3),
    )

    np.testing.assert_allclose(
        marginalizable_gaussian_log_prob(w3),
        marginalizable_gaussian_log_prob(w4),
    )

    np.testing.assert_allclose(
        multivariate_normal_log_likelihood(
            w4,
            np.zeros(shape=(w4.shape[1])),
            np.eye(w4.shape[1]),
            np.zeros(shape=(w4.shape[0])),
        ),
        marginalizable_gaussian_log_prob(w4),
    )

    print("Marginalisation testing completed")

    # generate a sample of n nonlinear non-gaussian trajectories

    m0 = lambda size, rng: sp_stats.multivariate_normal(mean=m, cov=S).rvs(
        size=size, random_state=rng
    )

    f0 = lambda z: z @ A
    Γ0 = lambda size, rng: sp_stats.multivariate_normal(cov=Γ).rvs(
        size=size, random_state=rng
    )
    h0 = lambda z: z @ H
    Λ0 = lambda size, rng: sp_stats.multivariate_normal(cov=Λ).rvs(
        size=size, random_state=rng
    )
    z0, x0 = sample_nonlinear_nongaussian_trajectory(
        n, d_hidden, d_observed, T, m0, f0, Γ0, h0, Λ0
    )

    # we added some nan's the original so just compare the ends
    assert np.allclose(z[10:], z0[10:])
    assert np.allclose(x[10:], x0[10:])

    print("Nonlinear sampler agrees with linear sampler on linear model")

    assert np.allclose(
        full_log_prob(
            z[..., 0],
            x[..., 0],
            T,
            m[0],
            S[0, 0],
            A[0, 0],
            Γ[0, 0],
            H[0, 0],
            Λ[0, 0],
        )[10:],
        composite_log_prob(
            z[..., 0],
            x[..., 0],
            T,
            m[0],
            S[0, 0],
            A[0, 0],
            Γ[0, 0],
            H[0, 0],
            Λ[0, 0],
        )[10:],
    )
    print("The 1-d versions are in agreement")

    print("Tests succeeded")
