#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements a linear Gaussian state space model
"""

import pickle

import numpy as np

from framework import marginalizable_state_space_model as mssm
from framework_extended import state_space_model as ssm
from util import util_state_space as util

np_eps = np.finfo(float).eps


class StateSpaceLinearGaussian(ssm.StateSpaceModel):
    """Linear Gaussian state-space model;
    also known as a Linear Dynamical System / Kalman-type model
    """

    def __init__(self, alpha: float = 0.0):
        """instantiates model with hyperparameters as specified

        Parameters
        ----------
        alpha
            regularisation for linear model

        """
        super().__init__()
        self.alpha = alpha if alpha > 2 * np_eps else 0

    def __str__(self):
        return "State space model with linear Gaussian components"

    def fit(self, data: tuple[np.ndarray, np.ndarray]):
        self.data = tuple(map(np.atleast_3d, data))
        states, measurements = data

        self.state_init = {
            "mean": np.nanmean(states[0], axis=0),
            "cov": np.cov(
                util.take_finite_along_axis(states[0]), rowvar=False
            ),
        }

        self.state_model = dict(
            zip(
                ["coeff", "covar"],
                util.regress_alpha(
                    np.row_stack(states[:-1]),
                    np.row_stack(states[1:]),
                    self.alpha,
                )
                if self.alpha > 2 * np_eps
                else util.regress(
                    np.row_stack(states[:-1]), np.row_stack(states[1:])
                ),
            )
        )
        self.measurement_model = dict(
            zip(
                ["coeff", "covar"],
                util.regress_alpha(
                    np.row_stack(states[:]),
                    np.row_stack(measurements[:]),
                    self.alpha,
                )
                if self.alpha > 2 * np_eps
                else util.regress(
                    np.row_stack(states[:]), np.row_stack(measurements[:])
                ),
            )
        )

        return self

    def to_pickle(self) -> bytes:
        return pickle.dumps(
            {
                "state_init": self.state_init,
                "state_model": self.state_model,
                "measurement_model": self.measurement_model,
                "alpha": self.alpha,
            }
        )

    def from_pickle(self, p: bytes):
        pickle_dict = pickle.loads(p)
        self.state_init = pickle_dict["state_init"]
        self.state_model = pickle_dict["state_model"]
        self.measurement_model = pickle_dict["measurement_model"]
        self.alpha = pickle_dict["alpha"] if "alpha" in pickle_dict else 0
        return self

    def score(self, data: tuple[np.ndarray, np.ndarray] = None):
        if data is None:
            data = self.data
        states, measurements = data
        T = states.shape[0]

        full_mean_T0 = mssm.mm(
            T,
            self.state_init["mean"],
            self.state_model["coeff"],
            self.measurement_model["coeff"],
        )

        full_cov_T0 = mssm.CC(
            T,
            self.state_init["cov"],
            self.state_model["coeff"],
            self.state_model["covar"],
            self.measurement_model["coeff"],
            self.measurement_model["covar"],
        )

        return mssm.multivariate_normal_log_likelihood(
            np.hstack((*states, *measurements)),
            full_mean_T0,
            full_cov_T0,
            np.zeros(states.shape[1]),
        )

    def score_alt(self, data: tuple[np.ndarray, np.ndarray] = None):
        if data is None:
            data = self.data
        states, measurements = data
        T = states.shape[0]

        return mssm.full_marginalizable_log_prob(
            z=states,
            x=measurements,
            T=T,
            m=self.state_init["mean"],
            S=self.state_init["cov"],
            A=self.state_model["coeff"],
            Γ=self.state_model["covar"],
            H=self.measurement_model["coeff"],
            Λ=self.measurement_model["covar"],
        )


if __name__ == "__main__":
    print("Running tests...")

    # make reproducible
    rng = np.random.default_rng(42)

    n = 1000
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

    ztrain, xtrain = mssm.sample_trajectory(
        n, T, m, S, A, Γ, H, Λ, rng=np.random.default_rng(0)
    )
    ztest, xtest = mssm.sample_trajectory(
        n, T, m, S, A, Γ, H, Λ, rng=np.random.default_rng(1)
    )
    log_prob_lg = (
        mdl := StateSpaceLinearGaussian().fit((ztrain, xtrain))
    ).score((ztest, xtest))
    log_prob_alt = mdl.score_alt((ztest, xtest))

    assert np.allclose(log_prob_lg, log_prob_alt)

    assert np.allclose(
        log_prob_lg,
        mssm.full_log_prob(ztest, xtest, T, m, S, A, Γ, H, Λ),
        rtol=0.02,
        atol=0.02,
    )

    print("Tested model training and inference.")

    ztrain[np.random.default_rng(0).random(size=ztrain.shape) < 0.05] = np.nan
    xtrain[np.random.default_rng(0).random(size=xtrain.shape) < 0.05] = np.nan
    log_probs = (m := StateSpaceLinearGaussian().fit((ztrain, xtrain))).score(
        (ztest, xtest)
    )
    print("Tested training in the presence of missing data.")

    p = m.to_pickle()
    log_probs_pickled = (
        StateSpaceLinearGaussian().from_pickle(p).score((ztest, xtest))
    )
    assert np.allclose(log_probs, log_probs_pickled)
    print("Test of pickle-ability successful.")
