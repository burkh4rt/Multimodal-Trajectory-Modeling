#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run survival modelling on features created at baseline
"""

import pathlib

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation

pd.options.display.width = 79
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 100
pd.options.display.float_format = "{:,.3f}".format

whereami = pwd = pathlib.Path(__file__).absolute().parent


def main():
    # load data
    data = pd.read_csv(
        whereami.joinpath("results", "survival_modelling_data.csv"),
        index_col=0,
    )
    assert not data.isnull().any().any()  # no missing values

    # drop persons with baseline AD diagnosis
    data = data[data["days_to_ad_or_last_obs"] != 0]

    """
    fit univariate Cox-PH models
    """

    xval_conc_uni = {
        c: k_fold_cross_validation(
            CoxPHFitter(penalizer=0.1, l1_ratio=0.0),
            data[[c, "days_to_ad_or_last_obs", "ad_outcome"]],
            duration_col="days_to_ad_or_last_obs",
            event_col="ad_outcome",
            k=10,
            scoring_method="concordance_index",
            fitter_kwargs={
                # "robust": True,
                "fit_options": {"step_size": 0.001, "max_steps": 1000},
            },
            seed=0,
        )
        for c in data.columns[:-2]
    }

    print(
        pd.DataFrame.from_dict(
            data={k: np.mean(v) for k, v in xval_conc_uni.items()},
            columns=["avg. conc."],
            orient="index",
        )
        .rename_axis("variable", axis="columns")
        .sort_values("avg. conc.", ascending=False)
    )
    pd.DataFrame.from_records(
        data=[
            (k, i, c)
            for k, v in xval_conc_uni.items()
            for i, c in enumerate(v)
        ],
        columns=["variable", "batch", "concordance"],
        index="variable",
    ).to_csv(
        whereami.joinpath(
            "results", "paired_concordances_from_baseline_meas.csv"
        ),
        index_label="variable",
    )

    """
    fit multivariate Cox models
    """

    cog_init = ["adni_mem_init", "adni_ef_init", "moca_init", "adas13_init"]
    bio_init = ["amyloid_init", "gm_init"]

    opts = dict(penalizer=0.01)

    xval_conc_multi = {
        tuple(x[:6] for x in c_list): k_fold_cross_validation(
            CoxPHFitter(**opts),
            data[list(c_list) + ["days_to_ad_or_last_obs", "ad_outcome"]],
            duration_col="days_to_ad_or_last_obs",
            event_col="ad_outcome",
            k=10,
            scoring_method="concordance_index",
            fitter_kwargs={
                "robust": True,
                "fit_options": {
                    "step_size": 0.01,
                    "max_steps": 10000,
                    # "show_progress": True,
                },
            },
            seed=0,
        )
        for c_list in [
            ["our_index_snapshot_init"],
            *[[b] for b in bio_init],
            *[[c] for c in cog_init],
            cog_init,
            bio_init,
            cog_init + bio_init,
        ]
    }

    print("-" * 79)
    print("Models by concordance:")
    print(
        pd.DataFrame.from_dict(
            data={k: np.mean(v) for k, v in xval_conc_multi.items()},
            columns=["avg. conc."],
            orient="index",
        )
        .rename_axis("variables", axis="columns")
        .sort_values("avg. conc.", ascending=False)
    )

    pd.DataFrame.from_records(
        data=[
            (k, i, c)
            for k, v in xval_conc_multi.items()
            for i, c in enumerate(v)
        ],
        columns=["variables", "batch", "concordance"],
        index="variables",
    ).to_csv(
        whereami.joinpath(
            "results", "paired_concordances_from_baseline_meas_multiv.csv"
        ),
        index_label="variables",
    )


if __name__ == "__main__":
    main()


"""
variable                 avg. conc.
our_index_snapshot_init       0.836
adas13_init                   0.830
adni_mem_init                 0.829
amyloid_init                  0.807
moca_init                     0.801
adni_ef_init                  0.749
gm_init                       0.703
age_init                      0.541
-------------------------------------------------------------------------------
Models by concordance:
variables                                         avg. conc.
(adni_m, adni_e, moca_i, adas13, amyloi, gm_ini)       0.887
(adni_m, adni_e, moca_i, adas13)                       0.851
(our_in,)                                              0.836
(amyloi, gm_ini)                                       0.833
(adas13,)                                              0.830
(adni_m,)                                              0.829
(amyloi,)                                              0.807
(moca_i,)                                              0.801
(adni_e,)                                              0.749
(gm_ini,)                                              0.703
"""
