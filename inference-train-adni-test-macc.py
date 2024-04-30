#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests a model on NUS data that had been trained on ADNI data
"""

import glob
import os
import string

import numpy as np
import pandas as pd
import scipy.stats as sp_stats

from framework import marginalizable_mixture_model as mixmodel
from util import util_adni as data_adni
from util import util_macc as data_macc
from util import util_post_hoc as util_ph
from util import util_state_space as util

pd.options.display.width = 79
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 79
pd.options.display.float_format = "{:,.3f}".format

home_dir = os.path.dirname(os.path.abspath(__file__))
n_clusters = 3
hex_hash = "49cb9120d3a1713763a0bdbdb97779bb"


def main():
    ztrain_orig, xtrain, *_ = data_adni.get_trajectories()
    ztrain, std_param = util.standardize(ztrain_orig, return_params=True)
    (
        ztest_orig,
        xtest,
        dtest,
        mmse_cdrsum_cdrglobal_test,
        lengthtest,
        idstest,
        agestest,
    ) = data_macc.get_data()
    ztest = util.standardize(ztest_orig, params=std_param)
    final_dx = data_macc.get_final_diagnoses(dtest)
    demog = data_macc.get_demographics(idstest)

    pfile = sorted(
        glob.glob(f"./tmp/mmm-{hex_hash}*"),
        key=os.path.getmtime,
    ).pop()
    best_mdl = mixmodel.MMLinGaussSS_marginalizable.from_pickle(
        pfile, training_data={"states": ztrain, "observations": xtrain}
    )

    ctest_all = np.array(
        [
            best_mdl.correspondence[c]
            for c in best_mdl.mle_cluster_assignment(
                states=ztest, observations=xtest
            )
        ]
    )

    indextest_all = best_mdl.cluster_assignment_index(
        states=ztest, observations=xtest
    )

    df_mmse = (
        pd.DataFrame(
            data={"cluster": ctest_all.ravel(), "health_index": indextest_all},
            index=idstest.ravel(),
        )
        .join(
            pd.DataFrame(
                index=np.tile(idstest, (max(lengthtest), 1)).ravel(),
                data=dict(
                    zip(
                        "mmse_cdrsum_cdrglobal".split("_"),
                        map(
                            np.ravel,
                            np.split(mmse_cdrsum_cdrglobal_test, 3, axis=-1),
                        ),
                    )
                )
                | {
                    "age": agestest.ravel(),
                    "time_in_years": 2
                    * np.repeat(np.arange(ztest.shape[0]), ztest.shape[1]),
                    "age_diff": np.concatenate(
                        [
                            np.expand_dims(np.zeros_like(agestest[0]), axis=0),
                            np.diff(agestest, axis=0),
                        ],
                        axis=0,
                    ).ravel(),
                },
            )
        )
        .loc[lambda df: ~df.mmse.isna()]
        .reset_index()
    )

    ctest_snap = np.array(
        [
            best_mdl.correspondence[c]
            for c in best_mdl.mle_cluster_assignment(
                states=util.mask_all_but_time_i_vect(ztest, lengthtest - 1),
                observations=util.mask_all_but_time_i_vect(
                    xtest, lengthtest - 1
                ),
            )
        ]
    )

    ctest_moca = np.array(
        [
            best_mdl.correspondence[c]
            for c in best_mdl.mle_cluster_assignment(
                states=np.nan * ztest, observations=xtest
            )
        ]
    )

    pd.concat(
        [
            pd.crosstab(
                pd.Series(
                    cs,
                    name="cluster",
                ),
                pd.Series(final_dx.ravel(), name="diagnosis"),
                normalize="index",
            )[data_macc.diagnosis_list].stack()
            for cs in [ctest_all, ctest_moca, ctest_snap]
        ],
        axis=1,
    ).rename(
        columns={
            0: "trajectories",
            1: "cognitive_only",
            2: "single_assessment",
        }
    ).to_csv(
        os.path.join(
            home_dir,
            "figure-metadata-macc",
            "counts_by_cluster_and_outcome_macc.csv",
        )
    )

    print(
        pd.concat(
            [
                pd.Series(cs).value_counts(normalize=True)[
                    list(string.ascii_uppercase[:n_clusters])
                ]
                for cs in [ctest_all, ctest_moca, ctest_snap]
            ],
            axis=1,
        ).rename(
            columns={
                0: "trajectories",
                1: "cognitive_only",
                2: "single_assessment",
            }
        )
    )

    for ns, cs in {
        "all": ctest_all,
        "moca_only": ctest_moca,
        "snapshot": ctest_snap,
    }.items():
        χ2 = sp_stats.chi2_contingency(
            pd.crosstab(
                pd.Series(
                    cs,
                    name="cluster",
                ),
                pd.Series(final_dx.ravel(), name="diagnosis"),
            )[data_macc.diagnosis_list].values
        )
        print(
            "Pearson's χ^2 contingency test for {}: ".format(ns)
            + "stat={stat:.2f}, pval={pval:.2E}, dof={dof}".format(
                stat=χ2[0], pval=χ2[1], dof=χ2[2]
            )
        )

        util.pie(
            assignments=cs,
            savename=os.path.join(
                "figures", f"MACC_{n_clusters}_overall_pie_{ns}.pdf"
            ),
            colors=data_macc.cluster_colors,
        )

        util.pie(
            assignments=final_dx.ravel(),
            savename=os.path.join(
                "figures", f"MACC_{n_clusters}_overall_pie_{ns}_dx.pdf"
            ),
            colors=data_macc.diagnosis_colors,
            cluster_ordering=data_macc.diagnosis_list,
        )

        util.pies_by_cluster(
            savename=os.path.join(
                "figures",
                f"MACC_{n_clusters}_pie_charts_{ns}.pdf",
            ),
            categories=final_dx.ravel(),
            category_ordering=data_macc.diagnosis_list,
            category_legend_names={
                "NCI": "CN",
                "CIND": "mild MCI",
                "VCIND": "moderate MCI",
                "AD": "AD",
            },
            clusters=cs,
            legend_bbox_to_anchor=(2.25 - 0.4 * int(n_clusters == 3), 1),
            fig_length=6.0,
            fig_width=4.0 + 0.5 * int(n_clusters == 3),
            halo_colors=data_macc.cluster_colors,
            slice_colors=data_macc.diagnosis_colors,
        )

        util.pies_by_cluster(
            savename=os.path.join(
                "figures",
                f"MACC_{n_clusters}_pie_charts_{ns}_dx.pdf",
            ),
            categories=cs,
            clusters=final_dx,
            cluster_ordering=data_macc.diagnosis_list,
            halo_colors=data_macc.diagnosis_colors,
            slice_colors=data_macc.cluster_colors,
            legend_bbox_to_anchor=(2.25 - 0.4 * int(n_clusters == 3), 1),
            fig_length=6.0,
            fig_width=4.0 + 0.5 * int(n_clusters == 3),
        )

        for name, met in dict(
            zip(
                ["MMSE", "CDR (sum)", "CDR (global)"],
                np.split(mmse_cdrsum_cdrglobal_test, 3, axis=-1),
            )
        ).items():
            util.plot_metric_vs_clusters_over_time(
                metric=met.squeeze(),
                assignments=cs,
                metric_name=name,
                savename=f"figures/"
                f"MACC_trajectories_{util.make_str_nice(name)}"
                f"_all{n_clusters}c_{ns}.pdf",
                title="",
                xticks=np.array([0, 2, 4]),
                xlabel="Years from baseline",
                legend_loc="upper right",
            )

    util.histograms_by_cluster(
        metrics=indextest_all.reshape(-1, 1),
        clusters=ctest_all,
        savename=os.path.join(
            "figures",
            f"MACC_{n_clusters}c_index_by_cluster.pdf",
        ),
        nbins=30,
        nrows=1,
        ncols=1,
        metric_names=[""],
        mean_overlay=False,
        density=True,
        title="",
        tighten=False,
    )

    df_mmse.assign(
        mmse_age_adjusted=lambda df: util_ph.regressed_out_effect_cv(
            df.mmse.values.reshape(-1, 1), df.age.values.reshape(-1, 1)
        )
    ).to_csv(
        os.path.join(
            "posthoc", "results", "mmse_by_cluster_over_time_macc.csv"
        ),
        index=None,
    )

    contingency = pd.crosstab(
        pd.Series(ctest_all, name="cluster"),
        pd.Series(final_dx.ravel(), name="diagnosis"),
        normalize="index",
    )[data_macc.diagnosis_list]
    print(contingency.div(contingency.sum(axis=0), axis=1))

    print(
        demog.assign(cluster=ctest_all, age=agestest[0])
        .groupby("cluster")
        .agg("mean")[["age", "is_female", "edu_yrs", "apoe4_pos"]]
    )


if __name__ == "__main__":
    main()


"""
   trajectories  cognitive_only  single_assessment
A         0.203           0.392              0.361
B         0.120           0.297              0.215
C         0.677           0.310              0.424
Pearson's χ^2 contingency test for all: stat=39.48, pval=5.77E-07, dof=6
Pearson's χ^2 contingency test for moca_only: stat=97.18, pval=9.72E-19, dof=6
Pearson's χ^2 contingency test for snapshot: stat=54.30, pval=6.41E-10, dof=6
diagnosis   NCI  CIND  VCIND    AD
cluster                           
A         0.600 0.207  0.581 0.098
B         0.144 0.568  0.245 0.166
C         0.256 0.225  0.174 0.736
           age  is_female  edu_yrs  apoe4_pos
cluster                                      
A       73.641      0.344    8.812      0.219
B       74.973      0.211    8.684      0.158
C       72.390      0.673    7.421      0.355
"""
