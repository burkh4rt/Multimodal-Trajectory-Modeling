#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run prognostic models for Δ(age-adjusted MMSE) ~ features @ baseline
"""

import pathlib

import numpy as np
import pandas as pd
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_mdl_sel

pd.options.display.width = 88
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 100
pd.options.display.float_format = "{:,.3f}".format

whereami = pwd = pathlib.Path(__file__).absolute().parent


def main():
    # load data
    data = pd.read_csv(
        whereami.joinpath("results", "prognostics_all.csv"),
        index_col=0,
    )
    data = data.loc[~data.isna().any(axis=1)]

    cog_init = ["adni_mem_init", "adni_ef_init", "moca_init", "adas13_init"]
    bio_init = ["amyloid_init", "gm_init"]

    cv_mse = {
        tuple(x[:6] for x in c): -skl_mdl_sel.cross_val_score(
            skl_lm.Ridge(alpha=0.01),
            data[c].values,
            data.ann_mmse_change_age_adjusted.values.reshape(-1, 1),
            cv=10,
            scoring="neg_mean_squared_error",
        )
        for c in [
            ["our_index_snapshot_init"],
            *[[b] for b in bio_init],
            *[[c] for c in cog_init],
            ["mmse_init"],
            cog_init,
            bio_init,
            # cog_init + ["our_index_snapshot_init"],
            # bio_init + ["our_index_snapshot_init"],
            cog_init + bio_init,
            # cog_init + bio_init + ["mmse_init"],
            # cog_init + bio_init + ["mmse_init"] + ["our_index_snapshot_init"],
        ]
    }

    print("-" * 79)
    print("Models by MSE:")
    print(
        pd.DataFrame.from_dict(
            data={
                k: [np.mean(v), np.std(v) / np.sqrt(len(v))]
                for k, v in cv_mse.items()
            },
            columns=["avg. MSE", "std. err."],
            orient="index",
        )
        .rename_axis("variables", axis="columns")
        .sort_values("avg. MSE", ascending=False)
    )

    pd.DataFrame.from_records(
        data=[(k, i, c) for k, v in cv_mse.items() for i, c in enumerate(v)],
        columns=["variables", "batch", "MSE"],
        index="variables",
    ).to_csv(
        whereami.joinpath(
            "results", "paired_prognostic_mse_from_baseline.csv"
        ),
        index_label="variables",
    )


if __name__ == "__main__":
    main()

"""
-------------------------------------------------------------------------------
Models by MSE:
variables                                                         avg. MSE  std. err.
(mmse_i,)                                                            1.136      0.130
(gm_ini,)                                                            1.096      0.149
(amyloi,)                                                            1.019      0.115
(amyloi, gm_ini)                                                     1.003      0.122
(adni_e,)                                                            1.001      0.103
(moca_i,)                                                            0.998      0.095
(adni_m,)                                                            0.982      0.109
(amyloi, gm_ini, our_in)                                             0.926      0.107
(our_in,)                                                            0.916      0.095
(adas13,)                                                            0.900      0.087
(adni_m, adni_e, moca_i, adas13)                                     0.893      0.082
(adni_m, adni_e, moca_i, adas13, amyloi, gm_ini)                     0.880      0.081
(adni_m, adni_e, moca_i, adas13, our_in)                             0.851      0.082
(adni_m, adni_e, moca_i, adas13, amyloi, gm_ini, mmse_i)             0.791      0.081
(adni_m, adni_e, moca_i, adas13, amyloi, gm_ini, mmse_i, our_in)     0.750      0.082
"""
