#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements a mixture of state space models that can be trained with EM
"""

from __future__ import annotations

import datetime
import glob
import gzip
import hashlib
import json
import os
import pickle
import string
import warnings

import numpy as np
from sklearn import base as skl_base
from sklearn import cluster as skl_cluster

from util import util_state_space as util

home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StateSpaceMixtureModel(skl_base.BaseEstimator, skl_base.DensityMixin):
    """Mixture of state space models"""

    def __init__(
        self,
        n_clusters: int,
        data: tuple[np.ndarray, np.ndarray],
        component_model,  # : type(ssm.StateSpaceModel),
        *,
        component_model_hyperparams: dict = dict(),
        rng: np.random.Generator = np.random.default_rng(seed=42),
    ):
        super().__init__()
        self.rng = rng

        self.states, self.observations = map(np.atleast_3d, data)
        self.n_timesteps, self.n_data, self.d_states = self.states.shape
        self.d_observations = self.observations.shape[-1]

        self.n_clusters = n_clusters
        self.cluster_propensities = np.ones(self.n_clusters) / self.n_clusters
        self.cluster_assignment = self.rng.integers(
            self.n_clusters, size=self.n_data
        )

        self.component_model = component_model
        self.component_model_hyperparams = component_model_hyperparams
        self.cluster_models = [
            self.component_model(**self.component_model_hyperparams)
            for _ in range(self.n_clusters)
        ]

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
            + str(self.component_model).encode("utf-8")
            + (
                json.dumps(
                    self.component_model_hyperparams, sort_keys=True
                ).encode("utf-8")
                if self.component_model_hyperparams != {}
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

    def __str__(self):
        return "Mixture of state space models with {} components".format(
            self.component_model
        )

    @property
    def data(self) -> tuple[np.array, np.array]:
        return self.states, self.observations

    @property
    def correspondence(self) -> dict[int, str]:
        return self._correspondence

    @correspondence.setter
    def correspondence(self, corr: dict[int, str]) -> None:
        self._correspondence = corr
        self.inverse_correspondence = {
            v: k for k, v in self._correspondence.items()
        }

    def _E_step(self):
        """performs Expectation step of EM by hard assigning each training
        point to its mostly likely cluster
        """
        new_cluster_assignment = np.argmax(
            np.column_stack(
                [
                    self.cluster_propensities[c]
                    * np.exp(self.cluster_models[c].score(self.data))
                    for c in range(self.n_clusters)
                ]
            ),
            axis=1,
        )
        assert new_cluster_assignment.size == self.n_data
        assert set(new_cluster_assignment) == set(range(self.n_clusters))
        n_switches = int(
            np.sum(
                np.not_equal(self.cluster_assignment, new_cluster_assignment)
            )
        )
        self.cluster_assignment = new_cluster_assignment
        return n_switches

    def _M_step(self):
        """performs Maximisation step of EM by finding the parameters of best
        fit for each component model according to the inferred cluster
        memberships
        """
        for c in range(self.n_clusters):
            self.cluster_propensities[c] = np.mean(
                self.cluster_assignment == c
            )
            self.cluster_models[c].fit(
                (
                    self.states[:, self.cluster_assignment == c],
                    self.observations[:, self.cluster_assignment == c],
                )
            )
        assert np.isclose(sum(self.cluster_propensities), 1.0)

    def fit(
        self,
        *,
        init: str = "random",
        n_iter: int = 1000,
        n_restarts: int = 0,
        use_cache: bool = True,
        verbose: bool = False,
    ):
        """fits the mixture model with EM

        Parameters
        ----------
        init
            cluster initialisation; either uniformly random (default),
            "k-means" according to the initial hidden state, or
            "k-means-all" via k-means on the whole sequence of hidden states
        n_iter
            maximum number of EM iterations to perform; algorithm terminates
            after completing this many iterations, or after algorithm
            converges
        n_restarts
            the model can be retrained from multiple different random
            initialisations; in this case, the model with the best expected
            complete data log likelihood will be returned
        use_cache: bool
            should we use a cache-based scheme to save and reload results from
            previous training sessions?
        verbose: bool
            print info as we go along?

        Returns
        -------
        fitted model instance (if n_restarts>0, the best trained model)

        """

        if bool(use_cache):
            try:
                pfile = sorted(
                    glob.glob(
                        os.path.join(home_dir, "tmp", f"mmm-{self.hex_hash}*")
                    ),
                    key=os.path.getmtime,
                ).pop()
                best_mdl = StateSpaceMixtureModel.from_pickle(
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
            except AssertionError:
                if verbose:
                    print(
                        "Model found in cache does not match our "
                        "requirements."
                    )
            except Exception as err:
                if verbose:
                    print(f"Issue loading cached model -- encountered {err}")

        match init:
            case "k-means" | "kmeans":
                self.cluster_assignment = skl_cluster.KMeans(
                    n_clusters=self.n_clusters,
                    init="k-means++",
                    random_state=0,
                ).fit_predict(self.states[0])
            case "kmeans-all" | "k-means-all":
                self.cluster_assignment = skl_cluster.KMeans(
                    n_clusters=self.n_clusters,
                    init="k-means++",
                    random_state=0,
                ).fit_predict(
                    np.row_stack(
                        [
                            self.states[:, i, :].flatten()
                            for i in range(self.n_data)
                        ]
                    )
                )
            case "kmeans-take-finite" | "k-means-take-finite":
                self.cluster_assignment = skl_cluster.KMeans(
                    n_clusters=self.n_clusters,
                    init="k-means++",
                    random_state=0,
                ).fit_predict(
                    np.column_stack(
                        util.take_finite_along_axis(self.states, 0)
                    )
                )
            case _:
                self.cluster_assignment = self.rng.integers(
                    low=0, high=self.n_clusters, size=self.n_data
                )
        assert len(self.cluster_assignment) == self.n_data
        if np.min(np.bincount(self.cluster_assignment)) <= 3:
            warnings.warn(
                "Cluster initialisation method yielded a nearly"
                "empty cluster"
            )
            self.cluster_assignment = self.rng.integers(
                low=0, high=self.n_clusters, size=self.n_data
            )

        try:
            self._M_step()
            for i in range(n_iter):
                # print(self.score())
                n_switches = self._E_step()
                # print(f"{n_switches=}")
                if n_switches == 0:
                    # print(f"Optimisation completed in {i} steps.")
                    break
                if np.min(np.bincount(self.cluster_assignment)) <= 3:
                    raise Exception("Encountered nearly empty cluster.")
                self._M_step()
        except Exception:  # Encountered nearly empty cluster
            pass

        try:
            score = self.score()
        except TypeError:
            score = -np.infty
        best_mdl, best_score = self, score
        for i in range(n_restarts):
            try:
                new_candidate = StateSpaceMixtureModel(
                    n_clusters=self.n_clusters,
                    data=self.data,
                    component_model=self.component_model,
                    component_model_hyperparams=self.component_model_hyperparams,
                    rng=np.random.default_rng(seed=i),
                ).fit(init="random", n_iter=n_iter)
                if (new_score := new_candidate.score()) > best_score:
                    best_mdl, best_score = new_candidate, new_score
            except Exception:  # Encountered nearly empty cluster
                pass
        if best_score == -np.infty:
            raise Exception("training failed")

        best_mdl.last_trained = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .astimezone()
            .isoformat()
        )
        if use_cache:
            best_mdl.to_pickle(include_training_data=False)
        return best_mdl

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
                    "component_model": self.component_model,
                    "component_model_hyperparams": self.component_model_hyperparams,
                    "cluster_models": [
                        cm.to_pickle() for cm in self.cluster_models
                    ],
                    "rng": self.rng,
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
                mdl = StateSpaceMixtureModel(
                    n_clusters=mdl_dict["n_clusters"],
                    data=(
                        training_data["states"],
                        training_data["observations"],
                    ),
                    component_model=mdl_dict["component_model"],
                    component_model_hyperparams=mdl_dict[
                        "component_model_hyperparams"
                    ]
                    if "component_model_hyperparams" in mdl_dict
                    else dict(),
                    rng=mdl_dict["rng"],
                )
            else:
                mdl = StateSpaceMixtureModel(
                    n_clusters=mdl_dict["n_clusters"],
                    data=(mdl_dict["states"], mdl_dict["observations"]),
                    component_model=mdl_dict["component_model"],
                    component_model_hyperparams=mdl_dict[
                        "component_model_hyperparams"
                    ]
                    if "component_model_hyperparams" in mdl_dict
                    else dict(),
                    rng=mdl_dict["rng"],
                )
            mdl.cluster_propensities = mdl_dict["cluster_propensities"]
            mdl.cluster_models = [
                mdl.component_model().from_pickle(p)
                for p in mdl_dict["cluster_models"]
            ]
            mdl.rng = mdl_dict["rng"]
            mdl.cluster_assignment = mdl_dict["cluster_assignment"]
            mdl.correspondence = mdl_dict["correspondence"]
            mdl.inverse_correspondence = mdl_dict["inverse_correspondence"]
            mdl.last_trained = mdl_dict["last_trained"]
            return mdl

    def predict_proba(
        self,
        data: tuple[np.ndarray, np.ndarray] = None,
        return_prenormalized_log_probs: bool = False,
    ) -> np.array | tuple[np.array, np.array]:
        if data is None:
            data = self.data

        preds = np.column_stack(
            [
                self.cluster_propensities[c]
                * np.exp(self.cluster_models[c].score(data))
                for c in range(self.n_clusters)
            ]
        )
        preds /= preds.sum(axis=1, keepdims=True)

        if return_prenormalized_log_probs:
            return preds, np.stack(
                [
                    np.log(self.cluster_propensities[c])
                    + self.cluster_models[c].score(data)
                    for c in range(self.n_clusters)
                ]
            )
        else:
            return preds

    def predict(
        self,
        *,
        data: tuple[np.ndarray, np.ndarray] = None,
        letters: bool = True,
    ) -> np.array:
        preds = np.argmax(self.predict_proba(data=data), axis=1)

        if letters:
            return np.array([self.correspondence[i] for i in preds])
        else:
            return preds

    def score(self, data: tuple[np.ndarray, np.ndarray] = None) -> float:
        if data is None:
            data = self.data

        cluster_assignment = self.predict(data=data, letters=False)

        try:
            assert set(cluster_assignment) == set(range(self.n_clusters))
            assert cluster_assignment.size == data[0].shape[1]
        except AssertionError:
            return -np.infty

        conditional_log_likelihoods = np.column_stack(
            [
                self.cluster_models[c].score(data)
                for c in range(self.n_clusters)
            ]
        )

        return float(
            np.sum(np.log(self.cluster_propensities[cluster_assignment]))
            + np.sum(
                [
                    conditional_log_likelihoods[i, cluster_assignment[i]]
                    for i in range(cluster_assignment.size)
                ]
            )
        )

    def model_log_likelihood(
        self, data: tuple[np.ndarray, np.ndarray] = None
    ) -> float:
        if data is None:
            data = self.data

        return float(
            np.sum(
                np.log(
                    np.sum(
                        np.column_stack(
                            [
                                self.cluster_propensities[c]
                                * np.exp(self.cluster_models[c].score(data))
                                for c in range(self.n_clusters)
                            ]
                        ),
                        axis=1,
                    )
                )
            )
        )

    def cluster_assignment_index(
        self, *, cluster: str = "A", data: tuple[np.ndarray, np.ndarray] = None
    ) -> np.array:
        """Return pre-normalized log-odds of assignment to cluster `cluster`"""

        return self.predict_proba(
            data=data, return_prenormalized_log_probs=True
        )[-1][self.inverse_correspondence[cluster]]


# run some tests if called as a script
if __name__ == "__main__":
    print("Running tests...")

    import sklearn.metrics as skl_metrics
    import state_space_model_knn as ssm_knn
    import state_space_model_linear_gaussian as ssm_lg

    from framework import marginalizable_state_space_model as mssm

    rng = np.random.default_rng(42)

    n_clusters = 2
    n_timesteps = 25
    n_data = 100
    d_hidden = 2
    d_observed = 3
    cluster_propensities = np.array([0.4, 0.6])

    # form model parameters
    A = np.empty(shape=(n_clusters, d_hidden, d_hidden))
    Γ = np.empty(shape=(n_clusters, d_hidden, d_hidden))
    H = np.empty(shape=(n_clusters, d_hidden, d_observed))
    Λ = np.empty(shape=(n_clusters, d_observed, d_observed))

    for c in range(n_clusters):
        A[c] = rng.normal(scale=0.5, size=(d_hidden, d_hidden))
        Γ[c] = np.eye(d_hidden) / (c + 2.0)
        H[c] = rng.normal(size=(d_hidden, d_observed))
        Λ[c] = (c + 1.0) * np.eye(d_observed)

    # generate data according to model
    z = np.empty(shape=(n_timesteps, n_data, d_hidden))
    x = np.empty(shape=(n_timesteps, n_data, d_observed))
    c = np.empty(shape=(n_data,), dtype=int)

    for i in range(n_data):
        c[i] = rng.choice(np.arange(n_clusters), p=cluster_propensities)
        z[:, i, :], x[:, i, :] = map(
            np.squeeze,
            mssm.sample_trajectory(
                1,
                n_timesteps,
                np.zeros(shape=(d_hidden)),
                Γ[c[i]],
                A[c[i]],
                Γ[c[i]],
                H[c[i]],
                Λ[c[i]],
            ),
        )

    ssmm_lg = StateSpaceMixtureModel(
        n_clusters=2,
        data=(z, x),
        component_model=ssm_lg.StateSpaceLinearGaussian,
        component_model_hyperparams={"alpha": 1.0},
    ).fit(n_restarts=10, use_cache=True, verbose=True)

    print(
        skl_metrics.confusion_matrix(
            c,
            ssmm_lg.cluster_assignment,
        )
    )

    ssmm_knn = StateSpaceMixtureModel(
        n_clusters=2,
        data=(z, x),
        component_model=ssm_knn.StateSpaceKNN,
        component_model_hyperparams={"n_neighbors": 10},
    ).fit(n_restarts=10, use_cache=True, verbose=True)

    # seems that we have solved the overfitting problem that kde encountered
    print(
        skl_metrics.confusion_matrix(
            c,
            ssmm_knn.cluster_assignment,
        )
    )

    ssmm_lg.to_pickle(there_can_only_be_one=False)
    ssmm_lg.to_pickle(there_can_only_be_one=True)
    assert (
        len(
            glob.glob(
                os.path.join(home_dir, "tmp", f"mmm-{ssmm_lg.hex_hash}*")
            )
        )
        == 1
    )
    print("Highlander test succeeded")
