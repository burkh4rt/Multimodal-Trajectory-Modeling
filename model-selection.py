#!/usr/bin/env python3

"""
Compares resulting model likelihood vs. number of clusters
"""

import os

import matplotlib.pyplot as plt

from framework import marginalizable_mixture_model as mixmodel

# from util import util_macc as data_macc
from util import util_adni as data_adni
from util import util_state_space as util

plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["font.family"] = "serif"

home_dir = os.path.dirname(os.path.abspath(__file__))
alpha = 1.0
n_cluster_list = range(1, 8)


def main():
    ztrain_orig, xtrain, *_ = data_adni.get_trajectories()
    ztrain, std_param = util.standardize(ztrain_orig, return_params=True)

    # (
    #     ztest_orig,
    #     xtest,
    #     dtest,
    #     mmsetest,
    #     lengthtest,
    #     idstest,
    #     agestest,
    # ) = data_macc.get_data()
    # ztest = util.standardize(ztest_orig, params=std_param)

    """
    train models with different numbers of clusters and compare results
    """

    mdls = [
        mixmodel.MMLinGaussSS_marginalizable(
            n_clusters=n_clusters,
            states=ztrain,
            observations=xtrain,
            init="k-means",
            alpha=alpha,
        ).train_with_multiple_random_starts(n_starts=1000, use_cache=True)
        for n_clusters in n_cluster_list
    ]

    for dset in ["ADNI"]:  # "MACC"
        for s, attr in {
            "Expected complete data log likelihood": "e_complete_data_log_lik",
            "AIC": "aic",
            "BIC": "bic",
        }.items():
            fig, ax = plt.subplots()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            values = [
                getattr(m, attr)(
                    states=ztrain,  # if dset == "ADNI" else ztest,
                    observations=xtrain,  # if dset == "ADNI" else xtest,
                )
                for m in mdls
            ]

            plt.plot(
                n_cluster_list,
                values,
                "o-",
                color="#0072CE",
                linestyle="solid",
            )
            plt.xticks(
                ticks=n_cluster_list,
                labels=n_cluster_list,
            )
            # plt.title(f"{s} vs. number of clusters")
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel(s)
            plt.tight_layout()
            os.makedirs("figures", exist_ok=True)
            plt.savefig(
                os.path.join(
                    "figures",
                    f"{dset}_elbow_plot_{attr.upper()}.pdf",
                ),
                bbox_inches="tight",
                transparent=True,
            )

    for m in mdls:
        m.to_pickle()


if __name__ == "__main__":
    main()
