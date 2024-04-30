#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements an abstract state space model
"""

import abc

import numpy as np
import sklearn.base as skl_base


class StateSpaceModel(
    skl_base.BaseEstimator, skl_base.DensityMixin, metaclass=abc.ABCMeta
):
    """abstract base class for a state space model"""

    def __init__(self):
        super().__init__()
        self.state_init = None
        self.state_model = None
        self.measurement_model = None
        self.data = None
        self.data_hash = None

    def __str__(self):
        """string name for the class"""
        return "State space model"

    def fit(self, data: tuple[np.ndarray, np.ndarray]):
        """fits the model on the data"""
        pass

    def score(self, data: tuple[np.ndarray, np.ndarray]):
        """provides overall score for the model on data"""
        pass

    @property
    def n_params(self):
        raise NotImplementedError
