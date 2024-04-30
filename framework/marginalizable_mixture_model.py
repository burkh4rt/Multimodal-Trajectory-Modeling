#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements a mixture of linear gaussian state space models that can be trained
with EM and can handle missing data and trajectories of differing lengths
"""

from __future__ import annotations

import datetime
import glob
import gzip
import hashlib
import os
import pickle
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.stats as sp_stats
import sklearn.cluster as skl_cluster
import sklearn.linear_model as skl_lm
import statsmodels.api as sm

from framework import marginalizable_state_space_model as statespace
from util import util_state_space as util

plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["font.family"] = "serif"

np_eps = np.finfo(float).eps
home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MMLinGaussSS_marginalizable:
    """
    Provides a probabilistic mixture of linear gaussian state space models;
    Implements expectation maximisation for this mixture on provided data
    """

    def __init__(
        self,
        n_clusters: int,
        states: np.array,
        observations: np.array,
        random_seed: int = 42,
        init: str = "random",
        alpha: float = 0.0,
    ):
        """Creates a model instance

        Parameters
        ----------
        n_clusters: int
            number of clusters in the mixture
        states: np.array
            n_timesteps × n_data × d_states array of latent states
        observations: np.array
            n_timesteps × n_data × d_observations array of measurements
        random_seed: int
            for random number generation
        init: str
            cluster initialisation method;
            either "random"
            or ("kmeans","k-means") for k-means on first avail. hidden states
            or ("kmeans-all","k-means-all") for k-means on all hidden states
        alpha: float
            regularisation parameter for linear models

        """
        states, observations = map(np.atleast_3d, (states, observations))
        self.n_clusters = int(n_clusters)
        self.states = np.array(states)
        self.observations = np.array(observations)

        self.n_timesteps, self.n_data, self.d_states = self.states.shape
        self.d_observations = self.observations.shape[-1]

        self.cluster_propensities = (
            np.ones(shape=[self.n_clusters]) / self.n_clusters
        )

        self.init_state_means = [
            np.random.normal(size=[self.d_states])
            for _ in range(self.n_clusters)
        ]

        self.init_state_covs = [
            np.random.normal(size=[self.d_states, self.d_states])
            for _ in range(self.n_clusters)
        ]
        self.init_state_covs = [
            x @ x.T + np.eye(self.d_states) for x in self.init_state_covs
        ]

        self.transition_matrices = [
            np.random.normal(size=[self.d_states, self.d_states])
            for _ in range(self.n_clusters)
        ]

        self.transition_covs = [
            np.random.normal(size=[self.d_states, self.d_states])
            for _ in range(self.n_clusters)
        ]
        self.transition_covs = [
            x @ x.T + np.eye(self.d_states) for x in self.transition_covs
        ]

        self.measurement_matrices = [
            np.random.normal(size=[self.d_states, self.d_observations])
            for _ in range(self.n_clusters)
        ]

        self.measurement_covs = [
            np.random.normal(size=[self.d_observations, self.d_observations])
            for _ in range(self.n_clusters)
        ]
        self.measurement_covs = [
            x @ x.T + np.eye(self.d_observations)
            for x in self.measurement_covs
        ]

        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=self.random_seed)
        self.init = init
        self.alpha = alpha if alpha > 2 * np_eps else 0
        match self.init:
            case "k-means" | "kmeans":
                idx_first_non_null = np.argmax(
                    np.isfinite(self.states).all(axis=2), axis=0
                ).ravel()
                first_non_null_state = np.vstack(
                    [
                        self.states[idx_first_non_null[i], i, :]
                        for i in range(self.n_data)
                    ]
                )
                first_non_null_state = np.where(
                    np.isfinite(first_non_null_state),
                    first_non_null_state,
                    np.nanmean(first_non_null_state, axis=0, keepdims=True),
                )
                self.cluster_assignment = skl_cluster.KMeans(
                    n_clusters=self.n_clusters,
                    init="k-means++",
                    random_state=self.random_seed,
                ).fit_predict(first_non_null_state)
            case "kmeans-all" | "k-means-all":
                self.cluster_assignment = skl_cluster.KMeans(
                    n_clusters=self.n_clusters,
                    init="k-means++",
                    random_state=self.random_seed,
                ).fit_predict(
                    np.row_stack(
                        [
                            self.states[:, i, :].flatten()
                            for i in range(self.n_data)
                        ]
                    )
                )
            case _:
                self.cluster_assignment = self.rng.integers(
                    low=0, high=self.n_clusters, size=self.n_data
                )

        self._correspondence = dict(
            zip(range(self.n_clusters), string.ascii_uppercase)
        )
        self.inverse_correspondence = {
            v: k for k, v in self._correspondence.items()
        }

        self.hex_hash = hashlib.md5(
            self.states.tobytes()
            + self.observations.tobytes()
            + str(self.n_clusters).encode("utf-8")
            + (
                np.format_float_positional(self.alpha, unique=True).encode(
                    "utf-8"
                )
                if self.alpha > 2 * np_eps
                else b""
            )
        ).hexdigest()
        self.time_stamp = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .astimezone()
            .isoformat()
        )
        self.last_trained = None

    @property
    def n_free_params(self) -> int:
        return sum(
            [
                x.size
                for x in [self.cluster_propensities]
                + self.init_state_means
                + self.transition_matrices
                + self.measurement_matrices
            ]
        ) + sum(
            map(
                lambda x: len(np.triu_indices_from(np.atleast_2d(x))[0]),
                self.init_state_covs
                + self.transition_covs
                + self.measurement_covs,
            )
        )

    @property
    def correspondence(self) -> dict[int, str]:
        return self._correspondence

    @correspondence.setter
    def correspondence(self, corr: dict[int, str]) -> None:
        self._correspondence = corr
        self.inverse_correspondence = {
            v: k for k, v in self._correspondence.items()
        }

    def to_pickle(
        self,
        save_location: str | os.PathLike = os.path.join(home_dir, "tmp"),
        there_can_only_be_one: bool = True,
        include_training_data: bool = False,
    ):
        os.makedirs(save_location, exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%MZ"
        )
        if there_can_only_be_one:
            list(
                map(
                    os.remove,
                    glob.glob(
                        os.path.join(save_location, f"mmm-{self.hex_hash}*")
                    ),
                )
            )
        with gzip.open(
            os.path.join(
                save_location,
                f"mmm-{self.hex_hash}-{ts}.p.gz",
            ),
            "wb",
        ) as f:
            pickle.dump(
                {
                    "n_clusters": self.n_clusters,
                    "cluster_propensities": self.cluster_propensities,
                    "init_state_means": self.init_state_means,
                    "init_state_covs": self.init_state_covs,
                    "transition_matrices": self.transition_matrices,
                    "transition_covs": self.transition_covs,
                    "measurement_matrices": self.measurement_matrices,
                    "measurement_covs": self.measurement_covs,
                    "random_seed": self.random_seed,
                    "rng": self.rng,
                    "init": self.init,
                    "alpha": self.alpha,
                    "cluster_assignment": self.cluster_assignment,
                    "correspondence": self.correspondence,
                    "inverse_correspondence": self.inverse_correspondence,
                    "hex_hash": self.hex_hash,
                    "time_stamp": self.time_stamp,
                    "last_trained": self.last_trained,
                }
                | (
                    {"states": self.states, "observations": self.observations}
                    if include_training_data
                    else {}
                ),
                f,
            )

    @staticmethod
    def from_pickle(file: str | os.PathLike, training_data: dict = None):
        with gzip.open(file, "rb") if os.path.splitext(file)[
            -1
        ] == ".gz" else open(file, "rb") as f:
            mdl_dict = pickle.load(f)
        if training_data is not None:
            mdl = MMLinGaussSS_marginalizable(
                n_clusters=mdl_dict["n_clusters"],
                states=training_data["states"],
                observations=training_data["observations"],
                random_seed=mdl_dict["random_seed"],
                init=mdl_dict["init"],
                alpha=mdl_dict["alpha"] if "alpha" in mdl_dict else 0,
            )
        else:
            mdl = MMLinGaussSS_marginalizable(
                n_clusters=mdl_dict["n_clusters"],
                states=mdl_dict["states"],
                observations=mdl_dict["observations"],
                random_seed=mdl_dict["random_seed"],
                init=mdl_dict["init"],
                alpha=mdl_dict["alpha"] if "alpha" in mdl_dict else 0,
            )
        mdl.cluster_propensities = mdl_dict["cluster_propensities"]
        mdl.init_state_means = mdl_dict["init_state_means"]
        mdl.init_state_covs = mdl_dict["init_state_covs"]
        mdl.transition_matrices = mdl_dict["transition_matrices"]
        mdl.transition_covs = mdl_dict["transition_covs"]
        mdl.measurement_matrices = mdl_dict["measurement_matrices"]
        mdl.measurement_covs = mdl_dict["measurement_covs"]
        mdl.rng = mdl_dict["rng"]
        mdl.cluster_assignment = mdl_dict["cluster_assignment"]
        mdl.correspondence = mdl_dict["correspondence"]
        mdl.inverse_correspondence = mdl_dict["inverse_correspondence"]
        mdl.time_stamp = mdl_dict["time_stamp"]
        mdl.last_trained = mdl_dict["last_trained"]
        return mdl

    def print_model(
        self,
        *,
        verbose: bool = False,
        line_len: int = 79,
    ) -> None:
        """Print model parameters.

        Parameters
        ----------
        verbose: bool
            additionally prints out covariance parameters
        line_len: int
            how wide should the print out be?

        """
        print(
            "MixtureModelLinearGaussianStateSpace |".ljust(line_len, "=")
            + "\n"
        )
        for s in string.ascii_uppercase[: self.n_clusters]:
            c = self.inverse_correspondence[s]
            print(f"Cluster {s} |".ljust(line_len, "-"))
            print(f"Cluster propensity:\n {self.cluster_propensities[c]:.3f}")
            print(
                f"Initial state mean:\n "
                f"{np.round(self.init_state_means[c], 3)}"
            )
            if verbose:
                print(
                    f"Initial state cov:\n "
                    f"{np.round(self.init_state_covs[c], 3)}"
                )
            print(
                f"State transition coeffs:\n "
                f"{np.round(self.transition_matrices[c], 3)}"
            )
            if verbose:
                print(
                    f"Transition cov:\n {np.round(self.transition_covs[c], 3)}"
                )
            print(
                f"Measurement coeffs:\n "
                f"{np.round(self.measurement_matrices[c], 3)}"
            )
            if verbose:
                print(
                    f"Measurement cov:\n "
                    f"{np.round(self.measurement_covs[c], 3)}"
                )
        print(f"{self.last_trained=}")
        print(f"{self.hex_hash=}")
        print("=" * line_len)

    def print_tests(
        self,
        *,
        test_1: bool = False,
        test_01: bool = False,
        test_obs: bool = False,
    ) -> None:
        """Print tests for model parameters

        Parameters
        ----------
        test_1: bool
            Should we test learned state evolution coefficients against 1?
        test_01: bool
            Should we test x0=0 & x1=1?
        test_obs: bool
            Should we test the observation / measurement models?

        """
        for s in string.ascii_uppercase[: self.n_clusters]:
            c = self.inverse_correspondence[s]
            Zcprev = np.row_stack(
                [*self.states[:-1, self.cluster_assignment == c, :]]
            )
            Zcnext = np.row_stack(
                [*self.states[1:, self.cluster_assignment == c, :]]
            )
            trans_idx = np.isfinite(np.column_stack([Zcprev, Zcnext])).all(
                axis=1
            )
            Zcprev = Zcprev[trans_idx, :]
            Zcnext = Zcnext[trans_idx, :]
            for i in range(self.d_states):
                print(f" Cluster {s} -- State {i} ".center(78, "-"))
                res = sm.OLS(endog=Zcnext[:, i], exog=Zcprev).fit()
                print(res.summary())
                if test_1:
                    t_res = res.t_test(f"x{i+1}=1", use_t=True)
                    print(f"testing x{i+1}=1")
                    print(t_res)
                    print(f"dof={t_res.df_denom}")
                if test_01:
                    t_res = res.t_test(
                        f"x{1 if i+1 == 2 else 2}=0, x{i+1}=1", use_t=True
                    )
                    print(f"testing x{1 if i+1 == 2 else 2}=0, x{i+1}=1")
                    print(t_res)
                    print(f"dof={t_res.df_denom}")

            if test_obs:
                Xcs = np.row_stack(
                    [*self.observations[:, self.cluster_assignment == c, :]]
                )
                Zcs = np.row_stack(
                    [*self.states[:, self.cluster_assignment == c, :]]
                )
                meas_idx = np.isfinite(np.column_stack([Xcs, Zcs])).all(axis=1)
                Xcs = Xcs[meas_idx, :]
                Zcs = Zcs[meas_idx, :]

                for j in range(self.d_observations):
                    print(f" Cluster {s} -- Observation {j} ")
                    print(sm.OLS(endog=Xcs[:, j], exog=Zcs).fit().summary())

    def conditional_log_likelihoods_first_T0_steps(
        self,
        c: int,
        T0: int,
        *,
        states: np.array = None,
        observations: np.array = None,
    ) -> np.array:
        """Computes class-conditional log likelihoods for each data instance
        restricted to steps 1<=t<=T0

        Parameters
        ----------
        c: int
            cluster index between 0 and n_clusters-1
        T0: int
            time cutoff 1 <= T0 <= self.n_timesteps
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        n_data length array of log likelihoods
            for c-th mixture restricted to time steps 1<=t<=T0

        """
        assert 1 <= T0 <= self.n_timesteps

        if states is None:
            states = self.states
            observations = self.observations

        _T0 = min(T0, states.shape[0])
        full_mean_T0 = statespace.mm(
            _T0,
            self.init_state_means[c],
            self.transition_matrices[c],
            self.measurement_matrices[c],
        )
        full_cov_T0 = statespace.CC(
            _T0,
            self.init_state_covs[c],
            self.transition_matrices[c],
            self.transition_covs[c],
            self.measurement_matrices[c],
            self.measurement_covs[c],
        )

        return statespace.multivariate_normal_log_likelihood(
            np.hstack((*states[:_T0], *observations[:_T0])),
            full_mean_T0,
            full_cov_T0,
            np.zeros(states.shape[1]),
        )

    def conditional_log_likelihoods(
        self,
        c: int,
        *,
        states: np.array = None,
        observations: np.array = None,
    ) -> np.array:
        """Computes class-conditional log likelihoods for each data instance

        Parameters
        ----------
        c: int
            cluster index between 0 and n_clusters-1
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        n_data length array of log likelihoods for c-th mixture

        See Also
        --------
        conditional_log_likelihoods_first_T0_steps
            to restrict the time horizon

        """
        if states is None:
            states = self.states
            observations = self.observations

        return self.conditional_log_likelihoods_first_T0_steps(
            c, self.n_timesteps, states=states, observations=observations
        )

    def cluster_propensities_over_time(
        self, *, states: np.array = None, observations: np.array = None
    ) -> np.array:
        """Computes probabilities of cluster membership for each training
        datapoint given only first t timesteps, for 1 <= t <= T

        Parameters
        ----------
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        pc_t
            an n_timesteps × n_data × n_clusters array where pc_t[t,i,:] is a
            probability vector predicting cluster membership for the ith data
            instance using only the first t+1 timesteps

        """

        pc_t = np.stack(
            [
                np.column_stack(
                    [
                        self.cluster_propensities[c]
                        * np.exp(
                            self.conditional_log_likelihoods_first_T0_steps(
                                c,
                                t + 1,
                                states=states,
                                observations=observations,
                            )
                        )
                        for c in range(self.n_clusters)
                    ]
                )
                for t in range(min(self.n_timesteps, states.shape[0]))
            ],
            axis=0,
        )

        pc_t /= np.sum(pc_t, axis=-1, keepdims=True)
        assert np.all(pc_t >= 0.0) and np.allclose(np.sum(pc_t, axis=-1), 1.0)
        return pc_t

    def e_complete_data_log_lik(
        self,
        *,
        states: np.array = None,
        observations: np.array = None,
    ) -> float:
        """Computes expected complete data log likelihood

        Note: EM should increase this value after every iteration

        Parameters
        ----------
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        expected complete data log likelihood (Q)

        """
        if states is None:
            states = self.states
            observations = self.observations

        cluster_assignment = self.mle_cluster_assignment(
            states=states, observations=observations
        )
        conditional_log_likelihoods = np.column_stack(
            [
                self.conditional_log_likelihoods(
                    c, states=states, observations=observations
                )
                for c in range(self.n_clusters)
            ]
        )

        return np.sum(
            np.log(self.cluster_propensities[cluster_assignment])
        ) + np.sum(
            [
                conditional_log_likelihoods[i, cluster_assignment[i]]
                for i in range(cluster_assignment.size)
            ]
        )

    def model_log_likelihood(
        self,
        *,
        states: np.array = None,
        observations: np.array = None,
    ) -> np.array:
        """Computes log likelihood over i.i.d. samples

        Parameters
        ----------
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        log likelihood

        """
        if states is None:
            states = self.states
            observations = self.observations

        Zcs = np.column_stack(
            [
                self.cluster_propensities[c]
                * np.exp(
                    self.conditional_log_likelihoods(
                        c,
                        states=states,
                        observations=observations,
                    )
                )
                for c in range(self.n_clusters)
            ]
        )
        assert Zcs.shape == (states.shape[1], self.n_clusters)
        lZ = np.log(np.sum(Zcs, axis=1))
        assert lZ.shape[0] == states.shape[1]
        return np.sum(lZ)

    def aic(
        self, states: np.array = None, observations: np.array = None
    ) -> float:
        """computes the AIC for the model on a given dataset (defaults to the
        training dataset)

        Parameters
        ----------
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        AIC
            for the model on the dataset

        """
        return (
            -2
            * self.model_log_likelihood(
                states=states, observations=observations
            )
            + 2 * self.n_free_params
        )

    def bic(
        self, states: np.array = None, observations: np.array = None
    ) -> float:
        """computes the BIC for the model on a given dataset (defaults to the
        training dataset)

        Parameters
        ----------
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        BIC
            for the model on the dataset

        """

        return (
            -2
            * self.model_log_likelihood(
                states=states, observations=observations
            )
            + np.log(self.n_data if states is None else states.shape[1])
            * self.n_free_params
        )

    def mle_cluster_assignment(
        self,
        *,
        return_probs: bool = False,
        return_prenormalized_log_probs: bool = False,
        states: np.array = None,
        observations: np.array = None,
    ) -> (
        tuple[np.array, np.array, np.array]
        | tuple[np.array, np.array]
        | np.array
    ):
        """Hard assignment of each data instance to a cluster according to
        maximum likelihood

        Parameters
        ----------
        return_probs: bool
            should we also return probs?
        return_prenormalized_log_probs: bool
            should we return prenormalized log-probs?
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        n_data length array of cluster indices in {0,...,n_clusters-1}
            or a tuple, with a probability vector for cluster assignment

        """
        if states is None:
            states = self.states
            observations = self.observations

        cluster_likelihoods = np.stack(
            [
                self.cluster_propensities[c]
                * np.exp(
                    self.conditional_log_likelihoods(
                        c, states=states, observations=observations
                    )
                )
                for c in range(self.n_clusters)
            ]
        )
        assignments = np.argmax(cluster_likelihoods, axis=0)
        if not (return_probs or return_prenormalized_log_probs):
            return assignments
        else:
            probs = np.divide(
                cluster_likelihoods,
                np.sum(cluster_likelihoods, axis=0, keepdims=True),
            )
            if not return_prenormalized_log_probs:
                return assignments, probs
            else:
                prenorm = np.stack(
                    [
                        np.log(self.cluster_propensities[c])
                        + self.conditional_log_likelihoods(
                            c, states=states, observations=observations
                        )
                        for c in range(self.n_clusters)
                    ]
                )
                return assignments, probs, prenorm

    def cluster_assignment_index(
        self,
        *,
        cluster: str = "A",
        states: np.array = None,
        observations: np.array = None,
    ) -> np.array:
        """Return pre-normalized log-odds of assignment to cluster `cluster`"""
        return self.mle_cluster_assignment(
            states=states,
            observations=observations,
            return_probs=True,
            return_prenormalized_log_probs=True,
        )[-1][self.inverse_correspondence[cluster]]

    def one_step_ahead_predictions(
        self, *, states, observations
    ) -> tuple[np.array, np.array]:
        """given trajectories of states & observations in the usual form,
        predicts cluster membership propensities and then forms the average
        one-step-ahead prediction for each provided data instance

        Parameters
        ----------
        states: np.array
            n_timesteps × n_data × d_states array of latent states
        observations: np.array
            n_timesteps × n_data × d_observations array of measurements

        Returns
        -------
        predicted_states: np.array
            1 × n_data × d_states array of predicted latent states
        predicted_observations: np.array
            1 × n_data × d_observations array of predicted measurements

        """
        assignment_probs = self.mle_cluster_assignment(
            states=states, observations=observations, return_probs=True
        )[1]
        next_states = np.zeros(shape=(1, *states.shape[1:]))
        next_observations = np.zeros(shape=(1, *observations.shape[1:]))
        these_states = states[-1]
        assert assignment_probs.shape == (
            self.n_clusters,
            these_states.shape[0],
        )
        for i in range(self.n_clusters):
            next_states_c = these_states @ self.transition_matrices[i]
            next_observations_c = next_states_c @ self.measurement_matrices[i]
            next_states += np.multiply(
                np.expand_dims(assignment_probs[i], axis=-1), next_states_c
            )
            next_observations += np.multiply(
                np.expand_dims(assignment_probs[i], axis=-1),
                next_observations_c,
            )
        return next_states, next_observations

    def one_step_ahead_predictions_no_history(
        self, *, states, observations
    ) -> tuple[np.array, np.array]:
        """given trajectories of states & observations in the usual form,
        predicts cluster membership propensities using only the most recent
        pair of states and measurements, and then forms the average
        one-step-ahead prediction for each provided data instance

        Parameters
        ----------
        states: np.array
            n_timesteps × n_data × d_states array of latent states
        observations: np.array
            n_timesteps × n_data × d_observations array of measurements

        Returns
        -------
        predicted_states: np.array
            1 × n_data × d_states array of predicted latent states
        predicted_observations: np.array
            1 × n_data × d_observations array of predicted measurements

        See Also
        --------
        one_step_ahead_predictions
            a version of this function that uses history for cluster assignment

        """
        states_no_history = np.nan * np.ones_like(states)
        states_no_history[-1] = states[-1]
        observations_no_history = np.nan * np.ones_like(observations)
        observations_no_history[-1] = observations[-1]

        assignment_probs = self.mle_cluster_assignment(
            states=states_no_history,
            observations=observations_no_history,
            return_probs=True,
        )[1]
        next_states = np.zeros(shape=(1, *states.shape[1:]))
        next_observations = np.zeros(shape=(1, *observations.shape[1:]))
        these_states = states[-1]
        assert assignment_probs.shape == (
            self.n_clusters,
            these_states.shape[0],
        )
        for i in range(self.n_clusters):
            next_states_c = these_states @ self.transition_matrices[i]
            next_observations_c = next_states_c @ self.measurement_matrices[i]
            next_states += np.multiply(
                np.expand_dims(assignment_probs[i], axis=-1), next_states_c
            )
            next_observations += np.multiply(
                np.expand_dims(assignment_probs[i], axis=-1),
                next_observations_c,
            )
        return next_states, next_observations

    def initial_full_data_cluster_assignment(
        self, *, states: np.array = None, observations: np.array = None
    ) -> np.array:
        """Hard assignment of each data instance to a cluster according to
        only data available initially (both hidden and observed)

        Parameters
        ----------
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        n_data length array of cluster indices in {0,...,n_clusters-1}
        or a tuple, with a probability vector for initial cluster assignment
        with both hidden and observed data

        """
        init_cluster_likelihoods = np.stack(
            [
                self.cluster_propensities[c]
                * np.exp(
                    self.conditional_log_likelihoods_first_T0_steps(
                        c, 1, states=states, observations=observations
                    )
                )
                for c in range(self.n_clusters)
            ]
        )
        assignments = np.argmax(init_cluster_likelihoods, axis=0)
        return assignments

    def predictions_from_initial_data(
        self, *, states: np.array = None, observations: np.array = None
    ) -> tuple[np.array, np.array]:
        """Predicted states and observations given only initial data, based
        on cluster assignment from inital data

        Parameters
        ----------
        states
            (optionally) override the default of self.states
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        tuple of predicted states and observations for each data instance using
        only information available at initial time point

        """
        assignments = self.initial_full_data_cluster_assignment(
            states=states, observations=observations
        )
        predicted_states = np.zeros_like(
            self.states if states is None else states
        )
        predicted_observations = np.zeros_like(
            self.observations if observations is None else observations
        )

        for i in range(self.n_data):
            predicted_states[:, i, :] = statespace.mmZ(
                predicted_states.shape[0],
                self.states[0, i, :],
                self.transition_matrices[assignments[i]],
            ).reshape(
                predicted_states.shape[0],
                self.d_states,
            )
            assert np.array_equal(
                predicted_states[0, i, :], self.states[0, i, :]
            )

            predicted_observations[:, i, :] = statespace.mmX(
                predicted_observations.shape[0],
                self.states[0, i, :],
                self.transition_matrices[assignments[i]],
                self.measurement_matrices[assignments[i]],
            ).reshape(
                predicted_observations.shape[0],
                self.d_observations,
            )

        return predicted_states, predicted_observations

    def observed_condl_log_lik_first_T0_steps(
        self, c: int, T0: int, *, observations: np.array = None
    ) -> np.array:
        """p(x|c), this marginalizes out the hidden states for a single
        state space model component

        Parameters
        ----------
        c: int
            0-based cluster index
        T0: int
            time cutoff 1 <= T0 <= self.n_timesteps
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        array of log likelihoods for the observations

        See Also
        --------
        conditional_log_likelihoods_first_T0_steps: uses both states and
            observations

        """
        if observations is None:
            observations = self.observations

        assert 1 <= T0 <= self.n_timesteps
        _T0 = min(T0, observations.shape[0])

        X_mean_T0 = statespace.mmX(
            _T0,
            self.init_state_means[c],
            self.transition_matrices[c],
            self.measurement_matrices[c],
        )
        X_cov_T0 = statespace.CXX(
            _T0,
            self.init_state_covs[c],
            self.transition_matrices[c],
            self.transition_covs[c],
            self.measurement_matrices[c],
            self.measurement_covs[c],
        )

        return statespace.multivariate_normal_log_likelihood(
            np.hstack(observations[:_T0]),
            X_mean_T0,
            X_cov_T0,
            np.zeros(observations.shape[1]),
        )

    def observed_conditional_log_likelihoods(
        self, c: int, observations: np.array = None
    ) -> np.array:
        """p(x|c), this marginalizes out the hidden states for a single
        state space model component

        Parameters
        ----------
        c: int
            0-based cluster index
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        n_data -length array of log likelihoods

        See Also
        --------
        observed_condl_log_lik_first_T0_steps
            to restrict the time horizon

        """
        return self.observed_condl_log_lik_first_T0_steps(
            c, self.n_timesteps, observations=observations
        )

    def observed_cluster_propensities_over_time(
        self, observations: np.array = None
    ) -> np.array:
        """Computes probabilities of cluster membership for each training
        datapoint given only first t timesteps, for 1 <= t <= T and
        only the observed data

        Parameters
        ----------
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        pc_t
            an n_timesteps × n_data × n_clusters array where pc_t[t,i,:] is a
            probability vector predicting cluster membership for the ith data
            instance using only the first t+1 timesteps and only the observed
            data

        """
        _T0 = (
            self.observations.shape[0]
            if observations is None
            else observations.shape[0]
        )
        pc_t = np.stack(
            [
                np.column_stack(
                    [
                        self.cluster_propensities[c]
                        * np.exp(
                            self.observed_condl_log_lik_first_T0_steps(
                                c, t + 1, observations=observations
                            )
                        )
                        for c in range(self.n_clusters)
                    ]
                )
                for t in range(_T0)
            ],
            axis=0,
        )

        pc_t /= np.sum(pc_t, axis=-1, keepdims=True)
        assert np.all(pc_t >= 0.0) and np.allclose(np.sum(pc_t, axis=-1), 1.0)
        return pc_t

    def observations_mle_cluster_assignment(
        self, *, return_probs: bool = False, observations: np.array = None
    ) -> np.array:
        """Hard assignment of each data observation to a cluster according to
        maximum likelihood

        Parameters
        ----------
        return_probs: np.array
            n_data length array of cluster indices in {0,...,n_clusters-1}
            or a tuple with additionally an n_data × n_clusters matrix of
            cluster membership probabilities for each data point
        observations
            (optionally) override the default of self.observations

        See also
        --------
        mle_cluster_assignment
            for using both the hidden and observed variables to assign
            cluster membership

        """
        cluster_likelihoods = np.stack(
            [
                self.cluster_propensities[c]
                * np.exp(
                    self.observed_conditional_log_likelihoods(
                        c, observations=observations
                    )
                )
                for c in range(self.n_clusters)
            ]
        )
        assignments = np.argmax(cluster_likelihoods, axis=0)
        if return_probs:
            probs = np.divide(
                cluster_likelihoods,
                np.sum(cluster_likelihoods, axis=0, keepdims=True),
            )
            return assignments, probs
        return assignments

    @staticmethod
    @numba.jit(
        numba.types.UniTuple(numba.float64[:, :], 2)(
            numba.float64[:, :], numba.float64[:, :]
        ),
        nopython=True,
    )
    def regress(
        input_exogenous: np.array, output_endogenous: np.array
    ) -> tuple[np.array, np.array]:
        """Finds the MLE estimates A_hat,S_hat for A,S
        where output|input ~ N(input*A, S)

        Parameters
        ----------
        input_exogenous: np.array
            n_data × in_dim array of inputs
        output_endogenous: np.array
            n_data × out_dim array of outputs

        Returns
        -------
        tuple of results
            in_dim × out_dim A_hat coefficient matrix
            out_dim × out_dim covariance matrix S_hat

        """
        A_hat = np.linalg.lstsq(input_exogenous, output_endogenous, rcond=-1)[
            0
        ]
        S_hat = np.cov(
            output_endogenous - input_exogenous @ A_hat, rowvar=False
        )
        return A_hat, S_hat

    @staticmethod
    def regress_alpha(
        input_exogenous: np.array, output_endogenous: np.array, alpha: float
    ) -> tuple[np.array, np.array]:
        """Finds the MLE estimates A_hat,S_hat for A,S
        where output|input ~ N(input*A, S)

        Parameters
        ----------
        input_exogenous: np.array
            n_data × in_dim array of inputs
        output_endogenous: np.array
            n_data × out_dim array of outputs
        alpha
            regularisation parameter for scikit-learn ridge regression

        Returns
        -------
        tuple of results
            in_dim × out_dim A_hat coefficient matrix
            out_dim × out_dim covariance matrix S_hat

        """
        A_hat = (
            skl_lm.Ridge(alpha=alpha, fit_intercept=False, copy_X=True)
            .fit(input_exogenous, output_endogenous)
            .coef_.T
        )
        S_hat = np.cov(
            output_endogenous - input_exogenous @ A_hat, rowvar=False
        )
        return A_hat, S_hat

    def E_step(self) -> int:
        """Performs the Expectation or `E` step of EM

        determines the expected cluster assignment for each data instance

        Returns
        -------
        number of data instances that have changed assignment
            if nothing changes, algorithm has converged

        """
        new_assignment = self.mle_cluster_assignment()

        n_switches = int(
            np.sum(np.not_equal(self.cluster_assignment, new_assignment))
        )
        self.cluster_assignment = new_assignment
        return n_switches

    def M_step(self) -> None:
        """Performs the Maximisation or `M` step of EM

        Updates each model parameter according to the cluster membership
        as determined in the `E` step
        """
        for c in range(self.n_clusters):
            self.cluster_propensities[c] = np.mean(
                self.cluster_assignment == c
            )
            Zc = self.states[:, self.cluster_assignment == c, :]
            Xc = self.observations[:, self.cluster_assignment == c, :]

            init_index = np.isfinite(Zc[0, :, :]).all(axis=1)
            Zc_init = Zc[0, init_index, :]
            self.init_state_means[c] = np.mean(Zc_init, axis=0)
            self.init_state_covs[c] = np.cov(Zc_init, rowvar=False)

            Zprev = np.row_stack([*Zc[:-1, :, :]])
            Znext = np.row_stack([*Zc[1:, :, :]])
            trans_idx = np.isfinite(np.column_stack([Zprev, Znext])).all(
                axis=1
            )
            Zprev = Zprev[trans_idx, :]
            Znext = Znext[trans_idx, :]
            if self.alpha > 2 * np_eps:
                (
                    self.transition_matrices[c],
                    self.transition_covs[c],
                ) = MMLinGaussSS_marginalizable.regress_alpha(
                    Zprev, Znext, self.alpha
                )
            else:
                (
                    self.transition_matrices[c],
                    self.transition_covs[c],
                ) = MMLinGaussSS_marginalizable.regress(Zprev, Znext)

            Xcs = np.row_stack([*Xc])
            Zcs = np.row_stack([*Zc])
            meas_idx = np.isfinite(np.column_stack([Xcs, Zcs])).all(axis=1)
            Xcs = Xcs[meas_idx, :]
            Zcs = Zcs[meas_idx, :]
            if self.alpha > 2 * np_eps:
                (
                    self.measurement_matrices[c],
                    self.measurement_covs[c],
                ) = MMLinGaussSS_marginalizable.regress_alpha(
                    Zcs, Xcs, self.alpha
                )
            else:
                (
                    self.measurement_matrices[c],
                    self.measurement_covs[c],
                ) = MMLinGaussSS_marginalizable.regress(Zcs, Xcs)

    def train(self, *, verbose: bool = False, n_steps: int = 1000):
        """Trains with EM for n_steps or until convergence

        converge occurs when cluster assignment does not change at all

        Parameters
        ----------
        verbose: bool
            print progress updates?
        n_steps: int
            number of training steps

        Returns
        -------
        self
            a trained version of the model

        """
        if (
            np.min(
                np.bincount(self.cluster_assignment, minlength=self.n_clusters)
            )
            <= 3
        ):
            if verbose:
                print(f"Encountered near-empty cluster.")
            return self
        self.M_step()
        if verbose:
            print(np.round(self.e_complete_data_log_lik(), 3))
        for i in range(n_steps):
            n_switches = self.E_step()
            if n_switches == 0:
                if verbose:
                    print(f"Optimisation completed in {i} steps.")
                break
            if (
                np.min(
                    np.bincount(
                        self.cluster_assignment, minlength=self.n_clusters
                    )
                )
                <= 3
            ):
                if verbose:
                    print(f"Encountered near-empty cluster.")
                break
            self.M_step()
            if verbose:
                print(np.round(self.e_complete_data_log_lik(), 3))
        self.last_trained = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .astimezone()
            .isoformat()
        )
        return self

    def train_with_multiple_random_starts(
        self,
        *,
        n_starts: int = 10,
        verbose: bool = False,
        n_steps: int = 100,
        return_objectives: bool = False,
        use_cache: bool = True,
    ):
        """train n_starts models from random initialisations

        Parameters
        ----------
        n_starts: int
            number of random initialisations to try
        verbose: bool
            pass verbosity to training function?
        n_steps: int
            number of training steps to perform for each model
        return_objectives: bool
            should we return the sequence of objective function values for
            each trained model?
        use_cache: bool
            should we use a cache-based scheme to save and reload results from
            previous training sessions?

        Returns
        -------
        model with best likelihood

        """

        if bool(use_cache):
            try:
                pfile = sorted(
                    glob.glob(
                        os.path.join(
                            home_dir,
                            "tmp",
                            f"mmm-{self.hex_hash}*",
                        )
                    ),
                    key=os.path.getmtime,
                ).pop()
                best_mdl = MMLinGaussSS_marginalizable.from_pickle(
                    pfile,
                    training_data={
                        "states": self.states,
                        "observations": self.observations,
                    },
                )
                assert self.hex_hash == best_mdl.hex_hash
                if verbose:
                    print(f"Loaded model {best_mdl.last_trained=} from cache.")
                return best_mdl
            except IndexError:
                if verbose:
                    print("No model found in cache.")
            except Exception as err:
                if verbose:
                    print(f"Issue loading cached model -- encountered {err}")

        best_mdl = MMLinGaussSS_marginalizable(
            n_clusters=self.n_clusters,
            states=self.states,
            observations=self.observations,
            random_seed=0,
            init="kmeans",
            alpha=self.alpha,
        )

        try:
            best_mdl = best_mdl.train(verbose=verbose, n_steps=n_steps)
        except Exception:
            pass

        if return_objectives:
            objective_list = [best_mdl.e_complete_data_log_lik()]

        for i in range(n_starts):
            try:
                mix_mdl = MMLinGaussSS_marginalizable(
                    n_clusters=self.n_clusters,
                    states=self.states,
                    observations=self.observations,
                    random_seed=100 + i,
                    alpha=self.alpha,
                ).train(verbose=verbose, n_steps=n_steps)
                if return_objectives:
                    objective_list.append(mix_mdl.e_complete_data_log_lik())
                if (
                    mix_mdl.e_complete_data_log_lik()
                    > best_mdl.e_complete_data_log_lik()
                ):
                    best_mdl = mix_mdl
            except:
                pass
        if not np.isfinite(best_mdl.e_complete_data_log_lik()):
            raise Exception("training failed")
        if use_cache:
            best_mdl.to_pickle()
        if return_objectives:
            return best_mdl, np.array(objective_list)
        return best_mdl

    def plot_cluster_propensity_evolution(
        self,
        savename: str,
        *,
        title: str = "Cluster Assignment Probability (using observed only) \n"
        "vs. Number of Time steps",
        observations: np.array = None,
    ) -> None:
        """determine and plot the mean likelihood +/- 1sem
        of belonging to the ultimately assigned
        cluster vs. time using observed data alone

        Parameters
        ----------
        savename
            file name for saving the plot
        title
            title for plot
        observations
            (optionally) override the default of self.observations

        Returns
        -------
        plot of likelihood of cluster membership for the ultimately assigned
        cluster vs time for each point in the training dataset

        """
        _T0 = (
            self.observations.shape[0]
            if observations is None
            else observations.shape[0]
        )
        propensities_over_time = self.observed_cluster_propensities_over_time(
            observations=observations
        )
        final_assignments = self.observations_mle_cluster_assignment(
            observations=observations
        )
        assert final_assignments.shape[0] == propensities_over_time.shape[1]
        likelihood_of_selected_cluster_over_time = np.stack(
            [
                propensities_over_time[:, i, final_assignments[i]]
                for i in range(propensities_over_time.shape[1])
            ]
        )
        assert (
            likelihood_of_selected_cluster_over_time.shape
            == propensities_over_time.T.shape[1:]
        )

        fig, ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        for i, c in enumerate(string.ascii_uppercase[: self.n_clusters]):
            c_mean = np.mean(
                likelihood_of_selected_cluster_over_time[
                    final_assignments == self.inverse_correspondence[c]
                ],
                axis=0,
            )
            c_sem = sp_stats.sem(
                likelihood_of_selected_cluster_over_time[
                    final_assignments == self.inverse_correspondence[c]
                ],
                axis=0,
            )

            plt.errorbar(
                x=np.arange(_T0) + 0.025 * (i - int(self.n_clusters / 2)),
                y=c_mean.T,
                yerr=c_sem.T,
                color=(
                    "#0072CE",
                    "#E87722",
                    "#64A70B",
                    "#93328E",
                    "#A81538",
                    "#4E5B31",
                )[i],
                linestyle="solid",
                label=f"cluster {c}",
                capsize=5,
            )
        handles, labels = ax.get_legend_handles_labels()
        unique_labels_dict = dict(zip(labels, handles))
        ax.legend(
            unique_labels_dict.values(),
            unique_labels_dict.keys(),
            fontsize="large",
        )
        plt.xticks(
            ticks=range(self.n_timesteps),
            labels=range(1, self.n_timesteps + 1),
        )
        plt.title(title)
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Probability")
        plt.savefig(savename, transparent=True)

    def plot_overall_cluster_propensity_evolution(
        self,
        savename: str,
        *,
        title: str = "Cluster Assignment Probability\n"
        "vs. Number of Time steps",
        observations: np.array = None,
        states: np.array = None,
    ) -> None:
        """determine and plot the mean likelihood +/- 1sem
        of belonging to the ultimately assigned
        cluster vs. time using observed data alone

        Parameters
        ----------
        savename
            file name for saving the plot
        title
            title for plot
        observations
            (optionally) override the default of self.observations
        states
            (optionally) override the default of self.states

        Returns
        -------
        plot of likelihood of cluster membership for the ultimately assigned
        cluster vs time for each point in the training dataset

        """
        if observations is None:
            observations = self.observations
            states = self.states
        _T0 = observations.shape[0]
        propensities_over_time = self.cluster_propensities_over_time(
            states=states, observations=observations
        )
        final_assignments = self.mle_cluster_assignment(
            states=states, observations=observations
        )
        assert final_assignments.shape[0] == propensities_over_time.shape[1]
        likelihood_of_selected_cluster_over_time = np.stack(
            [
                propensities_over_time[:, i, final_assignments[i]]
                for i in range(propensities_over_time.shape[1])
            ]
        )
        assert (
            likelihood_of_selected_cluster_over_time.shape
            == propensities_over_time.T.shape[1:]
        )

        fig, ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        for i, c in enumerate(string.ascii_uppercase[: self.n_clusters]):
            c_mean = np.nanmean(
                likelihood_of_selected_cluster_over_time[
                    final_assignments == self.inverse_correspondence[c]
                ],
                axis=0,
            )
            c_sem = sp_stats.sem(
                likelihood_of_selected_cluster_over_time[
                    final_assignments == self.inverse_correspondence[c]
                ],
                axis=0,
            )

            plt.errorbar(
                x=np.arange(_T0) + 0.025 * (i - int(self.n_clusters / 2)),
                y=c_mean.T,
                yerr=c_sem.T,
                color=(
                    "#0072CE",
                    "#E87722",
                    "#64A70B",
                    "#93328E",
                    "#A81538",
                    "#4E5B31",
                )[i],
                linestyle="solid",
                label=f"cluster {c}",
                capsize=5,
            )
        handles, labels = ax.get_legend_handles_labels()
        unique_labels_dict = dict(zip(labels, handles))
        ax.legend(
            unique_labels_dict.values(),
            unique_labels_dict.keys(),
            fontsize="large",
        )
        plt.xticks(
            ticks=range(self.n_timesteps),
            labels=range(1, self.n_timesteps + 1),
        )
        plt.title(title)
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Probability")
        plt.savefig(savename, transparent=True)

    def superimpose_model_on_plot(
        self, ax: mpl.axes.Axes, std_param: dict[str, np.array]
    ) -> None:
        for i, c in enumerate(string.ascii_uppercase[: self.n_clusters]):
            me, co = util.unstandardize_mean_and_cov(
                self.init_state_means[self.inverse_correspondence[c]],
                self.init_state_covs[self.inverse_correspondence[c]],
                params=std_param,
            )

            xv, yv = np.meshgrid(
                np.linspace(*ax.get_xlim(), num=1000),
                np.linspace(*ax.get_ylim(), num=1000),
            )
            pos = np.dstack((xv, yv))
            zv = sp_stats.multivariate_normal(mean=me, cov=co).pdf(pos)
            ax.contour(
                xv,
                yv,
                zv,
                colors=(
                    "#0072CE",
                    "#E87722",
                    "#64A70B",
                    "#93328E",
                    "#A81538",
                    "#4E5B31",
                )[i],
                linewidths=np.flip(1.5 ** -np.arange(10)),
            )

    def get_initial_means_and_stds(
        self, std_param: dict[str, np.array] = None
    ) -> dict[str, dict[str, np.array]]:
        """returns a dictionary with clusters as keys and dictionaries with
        keys "μ" & "σ", corresponding to the initial means and standard
        deviations of the features for that cluster according to the model;
        unstandardizes standardized values if `std_param` is passed

        """
        μσ_dict = {}
        for j in range(self.n_clusters):
            mz = self.init_state_means[j]
            cz = self.init_state_covs[j]
            mx = mz @ self.measurement_matrices[j]
            cx = (
                self.measurement_covs[j]
                + self.measurement_matrices[j].T
                @ cz
                @ self.measurement_matrices[j]
            )
            if std_param is not None:
                mz, cz = util.unstandardize_mean_and_cov(
                    mz, cz, params=std_param
                )
            mzx = np.concatenate([mz, mx])
            czx = np.concatenate(
                [np.diag(np.atleast_2d(cz)), np.diag(np.atleast_2d(cx))]
            )
            μσ_dict[self.correspondence[j]] = {"μ": mzx, "σ": np.sqrt(czx)}
        return μσ_dict

    def get_initial_diffs_means_and_stds(
        self, std_param: dict[str, np.array] = None
    ) -> dict[str, dict[str, np.array]]:
        """returns a dictionary with clusters as keys and dictionaries with
        keys "μ" & "σ", corresponding to the initial means and standard
        deviations of the initial differences of the features for that cluster
        according to the model--- i.e. second time step minus first time step;
        unstandardizes standardized values if `std_param` is passed

        """
        μσ_Δ_dict = {}
        coeff = np.block(
            [
                [
                    -np.eye(self.d_states),
                    np.eye(self.d_states),
                    np.zeros((self.d_states, 2 * self.d_observations)),
                ],
                [
                    np.zeros((self.d_observations, 2 * self.d_states)),
                    -np.eye(self.d_observations),
                    np.eye(self.d_observations),
                ],
            ]
        )
        for j in range(self.n_clusters):
            mmz0z1x0x1 = statespace.mm(
                T=2,
                m=self.init_state_means[j],
                A=self.transition_matrices[j],
                H=self.measurement_matrices[j],
            )
            CCz0z1x0x1 = statespace.CC(
                T=2,
                S=self.init_state_covs[j],
                A=self.transition_matrices[j],
                Γ=self.transition_covs[j],
                H=self.measurement_matrices[j],
                Λ=self.measurement_covs[j],
            )

            mmΔzΔx = coeff @ mmz0z1x0x1
            CCΔzΔx = coeff @ CCz0z1x0x1 @ coeff.T

            if std_param is not None:
                (
                    mmΔzΔx[: self.d_states],
                    CCΔzΔx[: self.d_states, : self.d_states],
                ) = util.unstandardize_mean_and_cov_diffs(
                    mmΔzΔx[: self.d_states],
                    CCΔzΔx[: self.d_states, : self.d_states],
                    params=std_param,
                )
            μσ_Δ_dict[self.correspondence[j]] = {
                "μ": mmΔzΔx,
                "σ": np.sqrt(np.diag(np.atleast_2d(CCΔzΔx))),
            }
        return μσ_Δ_dict

    @staticmethod
    def plot_matrix(
        mat: np.array,
        *,
        show_colorbar: bool = False,
        show_labels: bool = True,
        xticks: list = None,
        xlabel: str = None,
        yticks: list = None,
        ylabel: str = None,
        title: str = None,
        fmt_str: str = "{:.2f}",
        figsize: tuple = (6.4, 4.8),
        savename: os.PathLike | str = None,
        show: bool = False,
    ):
        mat = np.atleast_2d(mat)
        fig, ax = plt.subplots(layout="constrained", figsize=figsize)
        im = ax.matshow(mat, cmap="cividis")
        if show_colorbar:
            ax.figure.colorbar(im, ax=ax)
        if xticks:
            ax.set_xticks(np.arange(len(xticks)), labels=xticks)
            plt.setp(
                ax.get_xticklabels(),
                rotation=-30,
                ha="right",
                rotation_mode="anchor",
            )
        if yticks is not None:
            ax.set_yticks(np.arange(len(yticks)), labels=yticks)
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if show_labels:
            # thanks, @geology.beer
            for (_i, _j), _z in np.ndenumerate(mat):
                ax.text(
                    _j,
                    _i,
                    fmt_str.format(_z),
                    ha="center",
                    va="center",
                    c="black" if _z > np.mean(im.get_clim()) else "white",
                )
        plt.tick_params(bottom=False)
        _m, _n = mat.shape
        if _m == 1:
            plt.tick_params(left=False, labelleft=False, bottom=False)

        if savename is not None:
            plt.savefig(savename, transparent=True)
        if show:
            plt.show()

    @staticmethod
    def ponder():
        import webbrowser

        webbrowser.open_new_tab("https://doi.org/10.1017/prm.2023.96")

    def generate_model_plots(self, folder: str | os.PathLike, **kwargs):
        os.makedirs(
            os.path.join(
                folder,
                "{hh}-{nc}cl".format(hh=self.hex_hash, nc=self.n_clusters),
            ),
            exist_ok=True,
        )
        self.plot_matrix(
            self.cluster_propensities[
                np.argsort(
                    np.array(
                        [
                            self.correspondence[i]
                            for i in range(self.n_clusters)
                        ]
                    )
                )
            ],
            savename=os.path.join(
                folder,
                "{hh}-{nc}cl".format(hh=self.hex_hash, nc=self.n_clusters),
                "{}-propensities.pdf".format(self.hex_hash),
            ),
            title="Overall cluster propensities",
            **kwargs,
        )
        for c in range(self.n_clusters):
            for param in [
                "init_state_means",
                "init_state_covs",
                "transition_matrices",
                "transition_covs",
                "measurement_matrices",
                "measurement_covs",
            ]:
                self.plot_matrix(
                    getattr(self, param)[c],
                    savename=os.path.join(
                        folder,
                        "{hh}-{nc}cl".format(
                            hh=self.hex_hash, nc=self.n_clusters
                        ),
                        "{hh}-{par}-{cl}.pdf".format(
                            hh=self.hex_hash,
                            par=param,
                            cl=self.correspondence[c],
                        ),
                    ),
                    title="Cluster {c} {param}".format(
                        c=self.correspondence[c],
                        param=param[:-1]
                        .replace("_", " ")
                        .replace("matrice", "matrix"),
                    ),
                    **kwargs,
                )


# run some tests if called as a script
if __name__ == "__main__":
    print("Running tests...")

    # make reproducible
    np.random.seed(42)
    rng = np.random.default_rng(42)

    n_clusters = 2
    n_timesteps = 25
    n_data = 1000
    d_hidden = 2
    d_observed = 3
    cluster_propensities = np.array([0.4, 0.6])

    # form model parameters
    A = np.empty(shape=(n_clusters, d_hidden, d_hidden))
    G = np.empty(shape=(n_clusters, d_hidden, d_hidden))
    H = np.empty(shape=(n_clusters, d_hidden, d_observed))
    L = np.empty(shape=(n_clusters, d_observed, d_observed))

    for c in range(n_clusters):
        A[c] = rng.normal(scale=0.5, size=(d_hidden, d_hidden))
        G[c] = np.eye(d_hidden) / (c + 2.0)
        H[c] = rng.normal(size=(d_hidden, d_observed))
        L[c] = (c + 1.0) * np.eye(d_observed)

    # generate data according to model
    z = np.empty(shape=(n_timesteps, n_data, d_hidden))
    x = np.empty(shape=(n_timesteps, n_data, d_observed))
    c = np.empty(shape=(n_data,))

    for i in range(n_data):
        ci = rng.choice(np.arange(n_clusters), p=cluster_propensities)
        z[0, i, :] = sp_stats.multivariate_normal(cov=G[ci]).rvs(
            random_state=rng
        )
        x[0, i, :] = sp_stats.multivariate_normal(
            mean=z[0, i, :] @ H[ci], cov=L[ci]
        ).rvs(random_state=rng)
        for t in range(n_timesteps - 1):
            z[t + 1, i, :] = sp_stats.multivariate_normal(
                mean=z[t, i, :] @ A[ci],
                cov=G[ci],
            ).rvs(random_state=rng)
            x[t + 1, i, :] = sp_stats.multivariate_normal(
                mean=z[t + 1, i, :] @ H[ci],
                cov=L[ci],
            ).rvs(random_state=rng)
        c[i] = ci

    best_mdl = MMLinGaussSS_marginalizable(
        n_clusters=n_clusters,
        states=z,
        observations=x,
        init="kmeans",
    ).train_with_multiple_random_starts()

    assert np.allclose(
        np.sort(cluster_propensities),
        np.sort(best_mdl.cluster_propensities),
        rtol=1e-1,
    )

    print("Cluster propensities recovered successfully")

    correspondence = dict(
        zip(
            np.argsort(cluster_propensities),
            np.argsort(best_mdl.cluster_propensities),
        )
    )

    for c_true, c_inferred in correspondence.items():
        assert np.allclose(
            A[c_true],
            best_mdl.transition_matrices[c_inferred],
            rtol=1e-1,
            atol=1e-1,
        )
        print(f"State transition coefficients recovered for class {c_true}")

        assert np.allclose(
            G[c_true],
            best_mdl.transition_covs[c_inferred],
            rtol=1e-1,
            atol=2e-1,
        )
        print(f"State transition covariance recovered for class {c_true}")

        assert np.allclose(
            H[c_true],
            best_mdl.measurement_matrices[c_inferred],
            rtol=1e-1,
            atol=1e-1,
        )

        print(f"Measurement coefficients recovered for class {c_true}")

        assert np.allclose(
            L[c_true],
            best_mdl.measurement_covs[c_inferred],
            rtol=1e-1,
            atol=2e-1,
        )

        print(f"Measurement covariance recovered for class {c_true}")

    MMLinGaussSS_marginalizable(
        n_clusters=n_clusters,
        states=z,
        observations=x,
        init="random",
    ).train_with_multiple_random_starts(verbose=True, use_cache=True)

    z_pred, x_pred = best_mdl.one_step_ahead_predictions(
        states=z[:-1], observations=x[:-1]
    )

    assert np.allclose((z_pred - z[-1]).squeeze().mean(axis=0), 0.0, atol=0.02)
    assert np.allclose((x_pred - x[-1]).squeeze().mean(axis=0), 0.0, atol=0.05)

    print("One step ahead predictions are reasonable")

    (
        z_pred_0hist,
        x_pred_0hist,
    ) = best_mdl.one_step_ahead_predictions_no_history(
        states=z[:-1], observations=x[:-1]
    )

    assert np.allclose(
        (z_pred_0hist - z[-1]).squeeze().mean(axis=0), 0.0, atol=0.02
    )
    assert np.allclose(
        (x_pred_0hist - x[-1]).squeeze().mean(axis=0), 0.0, atol=0.05
    )

    print("History-free one step ahead predictions are reasonable")

    H0_est, L0_est = MMLinGaussSS_marginalizable.regress(
        z[0, c == 0], x[0, c == 0]
    )
    assert np.allclose(H0_est, H[0], atol=0.2)
    assert np.allclose(L0_est, L[0], atol=0.2)

    print("Tests of regression functionality completed")

    best_mdl.to_pickle()
    assert (
        len(
            glob.glob(
                os.path.join(
                    home_dir,
                    "tmp",
                    f"mmm-{best_mdl.hex_hash}*",
                )
            )
        )
        == 1
    )
    print("Highlander test succeeded")

    best_mdl = MMLinGaussSS_marginalizable(
        n_clusters=n_clusters,
        states=z,
        observations=x,
        init="kmeans",
        alpha=0.1,
    ).train_with_multiple_random_starts(use_cache=True, verbose=True)

    best_mdl.aic()
    best_mdl.bic(states=z[:, :10], observations=x[:, :10])
    print("Model selection functions working")

    best_mdl = MMLinGaussSS_marginalizable(
        n_clusters=n_clusters,
        states=z[..., 0].reshape(*z.shape[:-1], 1),
        observations=x[..., 0].reshape(*x.shape[:-1], 1),
        init="kmeans",
        alpha=0.1,
    ).train_with_multiple_random_starts(use_cache=True)
    print("Training on 1-d states and observations is working")

    print("Tests succeeded")
