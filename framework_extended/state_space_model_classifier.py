#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements a Bayesian classifier for state space models
"""

import numpy as np
from sklearn import base as skl_base

from framework_extended import state_space_model as ssm


class StateSpaceModelClassifier(
    skl_base.BaseEstimator, skl_base.DensityMixin, skl_base.ClassifierMixin
):
    """A generative classifier where p(data|class) is learned as a state
    space model
    """

    def __init__(
        self,
        component_model: type(ssm.StateSpaceModel),
    ):
        super().__init__()

        self.component_model = component_model
        self.classes, self.n_classes = None, None
        self.propensities = None
        self.class_models = None
        self.data = None

    def fit(self, data: tuple[np.ndarray, np.ndarray], labels: np.ndarray):
        self.data = tuple(map(np.atleast_3d, data))
        states, measurements = data
        self.classes, cts = np.unique(labels, return_counts=True)
        self.n_classes = len(self.classes)
        self.propensities = cts / np.sum(cts)
        self.class_models = [self.component_model() for _ in self.classes]
        for i, c in enumerate(self.classes):
            self.class_models[i].fit(
                data=(states[:, labels == c], measurements[:, labels == c])
            )
        return self

    def score(self, data: tuple[np.ndarray, np.ndarray] = None) -> float:
        if data is None:
            data = self.data
        else:
            data = tuple(map(np.atleast_3d, data))

        jt_p = np.sum(
            np.column_stack(
                [
                    self.propensities[i]
                    * np.exp(self.class_models[i].score(data=data))
                    for i in range(self.n_classes)
                ]
            ),
            axis=1,
        )
        assert jt_p.shape[0] == data[0].shape[1]
        return float(np.sum(np.log(jt_p)))

    def predict_proba(
        self, data: tuple[np.ndarray, np.ndarray] = None
    ) -> np.ndarray:
        if data is None:
            data = self.data
        else:
            data = tuple(map(np.atleast_3d, data))

        pc = np.column_stack(
            [
                self.propensities[i]
                * np.exp(self.class_models[i].score(data=data))
                for i in range(self.n_classes)
            ]
        )
        pc /= np.sum(pc, axis=1, keepdims=True)

        assert pc.shape[0] == data[0].shape[1]
        assert np.all(pc >= 0.0) and np.allclose(np.sum(pc, axis=-1), 1.0)
        return pc

    def predict(
        self, data: tuple[np.ndarray, np.ndarray] = None
    ) -> np.ndarray:
        if data is None:
            data = self.data
        else:
            data = tuple(map(np.atleast_3d, data))

        preds = self.classes[np.argmax(self.predict_proba(data), axis=1)]
        assert preds.size == data[0].shape[1]
        return preds


# run some tests if called as a script
if __name__ == "__main__":
    import sklearn.metrics as skl_metrics
    import state_space_model_linear_gaussian as ssm_lg

    from framework import marginalizable_state_space_model as mssm

    rng = np.random.default_rng(0)

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

    clr = StateSpaceModelClassifier(
        component_model=ssm_lg.StateSpaceLinearGaussian
    ).fit(data=(z, x), labels=c)

    print(f"{clr.score()=:.2f}")
    # clr.score()=-17039.91

    print(
        skl_metrics.confusion_matrix(
            c,
            clr.predict(),
        )
    )

    z[int(n_timesteps / 2) :, int(n_data / 2) :] = np.nan
    x[int(n_timesteps / 2) :, int(n_data / 2) :] = np.nan
    clr2 = StateSpaceModelClassifier(
        component_model=ssm_lg.StateSpaceLinearGaussian
    ).fit(data=(z, x), labels=c)

    print(
        skl_metrics.confusion_matrix(
            c,
            clr2.predict(),
        )
    )
