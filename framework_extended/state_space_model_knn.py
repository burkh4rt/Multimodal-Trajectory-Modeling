#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements a state space model with cross-validated kNN-based components
"""

import hashlib
import pickle

import numpy as np
import scipy.stats as sp_stats
import sklearn.model_selection as skl_ms
import sklearn.neighbors as skl_neighbors

from framework_extended import state_space_model as ssm
from util import util_state_space as util


class StateSpaceKNN(ssm.StateSpaceModel):
    """State space model where components are estimated with a cross-validated
    k-NN mean model with homoskedastic covariance; allows for non-linearities
    """

    def __init__(self, n_neighbors: int | list = 10, n_folds: int = 3):
        """instantiates model with hyperparameters as specified

        Parameters
        ----------
        n_neighbors
            number of neighbors to use for the k-NN models
        n_folds
            number of cross-validation folds to use
        """
        super().__init__()
        self.n_neighbors = (
            n_neighbors if isinstance(n_neighbors, list) else [n_neighbors]
        )
        self.n_folds = n_folds

    def __str__(self):
        return "State space model with k-NN-based components"

    def fit(self, data: tuple[np.ndarray, np.ndarray]):
        self.data = tuple(map(np.atleast_3d, data))
        states, measurements = data
        self.data_hash = hashlib.md5(
            states.tobytes() + measurements.tobytes()
        ).hexdigest()

        self.state_init = {
            "mean": np.nanmean(states[0], axis=0),
            "cov": np.cov(
                util.take_finite_along_axis(states[0]), rowvar=False
            ),
        }

        outp = np.row_stack(states[1:])
        inp = np.row_stack(states[:-1])
        trans_idx = np.isfinite(np.column_stack([inp, outp])).all(axis=1)
        trans_mdl = skl_ms.GridSearchCV(
            skl_neighbors.KNeighborsRegressor(),
            param_grid={"n_neighbors": self.n_neighbors},
            cv=self.n_folds,
            scoring="neg_mean_squared_error",
        )
        trans_mdl.fit(inp[trans_idx], outp[trans_idx])
        trans_inf = trans_mdl.predict(inp[trans_idx])

        self.state_model = {
            "mean": skl_neighbors.KNeighborsRegressor(
                n_neighbors=trans_mdl.best_params_["n_neighbors"]
            ).fit(
                inp[trans_idx],
                trans_inf,
            ),
            "cov": np.cov(outp[trans_idx] - trans_inf, rowvar=False),
        }

        outp = np.row_stack(measurements[:])
        inp = np.row_stack(states[:])
        meas_idx = np.isfinite(np.column_stack([inp, outp])).all(axis=1)
        out_mdl = skl_ms.GridSearchCV(
            skl_neighbors.KNeighborsRegressor(),
            param_grid={"n_neighbors": self.n_neighbors},
            cv=self.n_folds,
            scoring="neg_mean_squared_error",
        )
        out_mdl.fit(inp[meas_idx], outp[meas_idx])
        out_inf = out_mdl.predict(inp[meas_idx])

        self.measurement_model = {
            "mean": skl_neighbors.KNeighborsRegressor(
                n_neighbors=out_mdl.best_params_["n_neighbors"]
            ).fit(inp[meas_idx], out_inf),
            "cov": np.cov(outp[meas_idx] - out_inf, rowvar=False),
        }

        return self

    def to_pickle(self) -> bytes:
        return pickle.dumps(
            {
                "n_folds": self.n_folds,
                "n_neighbors": self.n_neighbors,
                "data_hash": self.data_hash,
                "state_init": self.state_init,
                "state_model": self.state_model,
                "measurement_model": self.measurement_model,
            }
        )

    def from_pickle(self, p: bytes):
        pickle_dict = pickle.loads(p)
        self.n_folds = pickle_dict["n_folds"]
        self.n_neighbors = pickle_dict["n_neighbors"]
        self.data_hash = pickle_dict["data_hash"]
        self.state_init = pickle_dict["state_init"]
        self.state_model = pickle_dict["state_model"]
        self.measurement_model = pickle_dict["measurement_model"]
        return self

    def score(self, data: tuple[np.ndarray, np.ndarray]):
        if data is None:
            data = self.data
        states, measurements = data
        T = states.shape[0]
        log_likelihoods = sp_stats.multivariate_normal(
            mean=self.state_init["mean"],
            cov=self.state_init["cov"],
            allow_singular=True,
        ).logpdf(states[0])
        for t in range(T - 1):
            states0, states1 = states[t], states[t + 1]
            idx_fin = np.isfinite(np.column_stack([states0, states1])).all(
                axis=1
            )
            log_likelihoods[idx_fin] += sp_stats.multivariate_normal(
                cov=self.state_model["cov"], allow_singular=True
            ).logpdf(
                states1[idx_fin]
                - self.state_model["mean"].predict(states0[idx_fin])
            )
        for t in range(T):
            states0, meas0 = states[t], measurements[t]
            idx_fin = np.isfinite(np.column_stack([states0, meas0])).all(
                axis=1
            )
            log_likelihoods[idx_fin] += sp_stats.multivariate_normal(
                cov=self.measurement_model["cov"], allow_singular=True
            ).logpdf(
                meas0[idx_fin]
                - self.measurement_model["mean"].predict(states0[idx_fin])
            )
        return log_likelihoods


if __name__ == "__main__":
    print("Running tests...")

    from framework import marginalizable_state_space_model as mssm

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
    log_prob_knn = (
        StateSpaceKNN(n_neighbors=[3, 5, 10])
        .fit((ztrain, xtrain))
        .score((ztest, xtest))
    )

    ztrain[np.random.default_rng(0).random(size=ztrain.shape) < 0.05] = np.nan
    xtrain[np.random.default_rng(0).random(size=xtrain.shape) < 0.05] = np.nan
    log_prob_knn = (
        m := StateSpaceKNN(n_neighbors=[3, 5, 10]).fit((ztrain, xtrain))
    ).score((ztest, xtest))
    print("Tested training in the presence of missing data.")

    p = m.to_pickle()
    log_prob_knn_pickled = (
        StateSpaceKNN(n_neighbors=[3, 5, 10])
        .from_pickle(p)
        .score((ztest, xtest))
    )
    assert np.allclose(log_prob_knn, log_prob_knn_pickled)
    print("Test of pickle-ability successful.")
