#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loads and processes results from `inference-adni-xval.py`
"""

import glob
import gzip
import itertools
import os
import pickle
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from util import util_adni as data
from util import util_post_hoc as util_ph
from util import util_state_space as st_sp

plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["font.family"] = "serif"

pd.options.display.width = pd.options.display.max_colwidth = 79
pd.options.display.max_columns = 1000
pd.options.display.float_format = "{:,.3f}".format

rng = np.random.default_rng(seed=0)

n_splits, n_clusters = 10, 4
lme_add_ci = True


def main():
    (
        z,
        x,
        d,
        ids,
        time,
        mmse,
        approx_age,
    ) = data.get_trajectories(return_mmse=True, return_approx_age=True)

    d_fin = data.get_final_diagnoses(d)
    df_fin = pd.DataFrame(
        data=d_fin, index=ids[0].ravel(), columns=["diagnosis"]
    )

    df_results = pd.read_csv(
        sorted(
            glob.glob(
                os.path.join(
                    "results",
                    f"ADNI_r7_xval{n_splits}_{n_clusters}clusters_results.csv",
                )
            ),
            key=os.path.getmtime,
        )[-1]
    ).set_index("id")

    with gzip.open(
        sorted(
            glob.glob(
                os.path.join(
                    "results",
                    f"ADNI_r7_xval{n_splits}_{n_clusters}clusters_results.p.gz",
                )
            ),
            key=os.path.getmtime,
        )[-1],
        "rb",
    ) as f:
        d_tr = pickle.load(f)

    print(
        data.return_profiling_dataframe(ids[0])
        .assign(
            cluster=df_results.our_cluster.ravel()[: z[0].shape[0]],
            age=approx_age[0],
        )
        .groupby("cluster")
        .agg("mean")[["age", "is_female", "edu_yrs", "apoe4_pos"]]
    )

    splits_idx = pd.RangeIndex(stop=n_splits, name="split")

    clusters_idx = pd.CategoricalIndex(
        data=list(string.ascii_uppercase[:n_clusters]),
        name="cluster",
        ordered=True,
    )

    diagnoses_idx = pd.CategoricalIndex(
        data=data.diagnosis_list, name="diagnosis", ordered=True
    )

    outcomes_by_cluster_train = pd.DataFrame(
        index=pd.MultiIndex.from_product([clusters_idx, diagnoses_idx]),
        columns=splits_idx,
    )

    outcomes_by_cluster_test = pd.DataFrame(
        index=pd.MultiIndex.from_product([clusters_idx, diagnoses_idx]),
        columns=splits_idx,
    )

    outcomes_by_cluster_snapshot_test = pd.DataFrame(
        index=pd.MultiIndex.from_product([clusters_idx, diagnoses_idx]),
        columns=splits_idx,
    )

    outcomes_by_cluster_snapshot_initial_test = pd.DataFrame(
        index=pd.MultiIndex.from_product([clusters_idx, diagnoses_idx]),
        columns=splits_idx,
    )

    outcomes_by_cluster_snapshot_initial_nh_test = pd.DataFrame(
        index=pd.MultiIndex.from_product([clusters_idx, diagnoses_idx]),
        columns=splits_idx,
    )

    outcomes_by_cluster_snapshot_nh_test = pd.DataFrame(
        index=pd.MultiIndex.from_product([clusters_idx, diagnoses_idx]),
        columns=splits_idx,
    )

    outcomes_by_cluster_no_hidden_test = pd.DataFrame(
        index=pd.MultiIndex.from_product([clusters_idx, diagnoses_idx]),
        columns=splits_idx,
    )

    clusters_by_outcome_train = pd.DataFrame(
        index=pd.MultiIndex.from_product([diagnoses_idx, clusters_idx]),
        columns=splits_idx,
    )

    clusters_by_outcome_test = pd.DataFrame(
        index=pd.MultiIndex.from_product([diagnoses_idx, clusters_idx]),
        columns=splits_idx,
    )

    for i_split in range(n_splits):
        df_training_results = d_tr[i_split]
        df_train_final = (
            df_training_results.loc[lambda df: ~df.diagnosis.isnull()]
            .sort_values("time")
            .groupby(level=0)
            .last()
        )

        df_testing_results = df_results.loc[
            lambda df: (df.split == i_split) & (~df.diagnosis.isnull())
        ]
        df_test_final = (
            df_testing_results.sort_values("time").groupby(level=0).last()
        )
        assert (
            df_test_final.join(df_fin, rsuffix="_")
            .assign(df_test=lambda df: df.diagnosis == df.diagnosis_)[
                "df_test"
            ]
            .all()
        )
        df_test_initial = (
            df_testing_results.sort_values("time")
            .groupby(level=0)
            .first()
            .drop(columns="diagnosis")
            .join(df_fin)
        )

        def outcomes_by_col(col: str, df: pd.DataFrame):
            """report diagnostic outcomes by col where col is a column in df"""
            assert col in df.columns.to_list()
            return (
                df.groupby([col, "diagnosis"])
                .agg(dx_cts=("diagnosis", "count"))
                .reset_index()
                .pivot(index=col, columns="diagnosis", values="dx_cts")
                .fillna(0)
                # .apply(lambda row: row / np.sum(row), axis=1)
            )

        # Outcomes by cluster (train)
        outcomes_by_cluster_train.loc[:, i_split] = outcomes_by_col(
            "our_cluster", df_train_final
        ).stack()

        # Outcomes by cluster (test)
        outcomes_by_cluster_test.loc[:, i_split] = outcomes_by_col(
            "our_cluster", df_test_final
        ).stack()

        # Outcomes by cluster snapshot (test)
        outcomes_by_cluster_snapshot_test.loc[:, i_split] = outcomes_by_col(
            "our_cluster_snapshots", df_test_final
        ).stack()

        outcomes_by_cluster_snapshot_initial_test.loc[
            :, i_split
        ] = outcomes_by_col("our_cluster_snapshots", df_test_initial).stack()

        outcomes_by_cluster_snapshot_nh_test.loc[:, i_split] = outcomes_by_col(
            "our_cluster_snapshots_no_hidden", df_test_final
        ).stack()

        outcomes_by_cluster_snapshot_initial_nh_test.loc[
            :, i_split
        ] = outcomes_by_col(
            "our_cluster_snapshots_no_hidden", df_test_initial
        ).stack()

        outcomes_by_cluster_no_hidden_test.loc[:, i_split] = outcomes_by_col(
            "our_cluster_no_hidden", df_test_final
        ).stack()

        def col_by_outcome(col: str, df: pd.DataFrame):
            """report col by diagnostic outcomes where col is a column in df

            See Also
            --------
            outcomes_by_col

            """
            assert col in df.columns.to_list()
            return (
                df.groupby(["diagnosis", col])
                .agg(dx_cts=(col, "count"))
                .reset_index()
                .pivot(index="diagnosis", columns=col, values="dx_cts")
                .fillna(0)
                .apply(lambda row: row / np.sum(row), axis=1)
            )

        # Cluster by outcome (train)
        clusters_by_outcome_train.loc[:, i_split] = col_by_outcome(
            "our_cluster", df_train_final
        ).stack()

        # Cluster by outcome (test)
        clusters_by_outcome_test.loc[:, i_split] = col_by_outcome(
            "our_cluster", df_test_final
        ).stack()

    outcomes_sum = (
        pd.concat(
            [
                outcomes_by_cluster_test.fillna(0.0).assign(
                    ours_test=lambda df: df.sum(axis=1, numeric_only=True)
                )[["ours_test"]],
                outcomes_by_cluster_no_hidden_test.fillna(0.0).assign(
                    ours_testnh=lambda df: df.sum(axis=1, numeric_only=True)
                )[["ours_testnh"]],
                outcomes_by_cluster_snapshot_initial_test.fillna(0.0).assign(
                    ours_testsnapshotsinit=lambda df: df.sum(
                        axis=1, numeric_only=True
                    )
                )[["ours_testsnapshotsinit"]],
                outcomes_by_cluster_snapshot_initial_nh_test.fillna(
                    0.0
                ).assign(
                    ours_testsnapshotsinitnh=lambda df: df.sum(
                        axis=1, numeric_only=True
                    )
                )[
                    ["ours_testsnapshotsinitnh"]
                ],
                outcomes_by_cluster_snapshot_test.fillna(0.0).assign(
                    ours_testsnapshots=lambda df: df.sum(
                        axis=1, numeric_only=True
                    )
                )[["ours_testsnapshots"]],
                outcomes_by_cluster_snapshot_nh_test.fillna(0.0).assign(
                    ours_testsnapshotsnh=lambda df: df.sum(
                        axis=1, numeric_only=True
                    )
                )[["ours_testsnapshotsnh"]],
            ],
            axis=1,
        )
        .pipe(
            lambda df: df.set_axis(
                pd.MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in df.columns]
                ),
                axis=1,
            )
        )
        .loc[lambda df: df.index.get_level_values(1) != "MCI_tbd"]
    )
    assert (outcomes_sum.sum(axis=0) == 571).all()

    print("outcomes by cluster")
    print(outcomes_sum.groupby(level=0).apply(lambda x: x / x.sum()))

    print("clusters by approach")
    print(
        outcomes_sum.groupby(level=0).sum().apply(lambda x: x / x.sum(axis=0))
    )

    print("population-level cluster prevalences")
    print(
        pd.concat(
            [
                df_results.loc[lambda df: df.time == 0, [c]]
                .join(df_fin)
                .groupby([c])
                .agg(ct=("diagnosis", "count"))
                .apply(lambda x: x / x.sum())
                .rename(columns={"ct": c})
                for c in (
                    "our_cluster",
                    "our_cluster_snapshots",
                    "gmm_init_predictions",
                )
            ],
            axis=1,
        )
    )

    print("diagnostic outcomes by cluster")
    print(
        pd.concat(
            [
                df_results.loc[lambda df: df.time == 0, [c]]
                .join(df_fin)
                .groupby([c, "diagnosis"])
                .agg(ct=("diagnosis", "count"))
                .apply(lambda x: x / x.sum())
                .rename(columns={"ct": c})
                for c in (
                    "our_cluster",
                    "our_cluster_snapshots",
                    "gmm_init_predictions",
                )
            ],
            axis=1,
        ).pipe(
            lambda df: df.loc[
                sorted(
                    df.index,
                    key=lambda x: (x[0], data.diagnosis_list.index(x[1])),
                )
            ]
        )
    )

    data.plot_2d_trajectories(
        model=None,
        savename=os.path.join(
            "figures",
            f"ADNI_r7_xval{n_splits}_our_model_plot{n_clusters}_results.pdf",
        ),
        title="",
        states=z,
        inferred_clusters=df_results.our_cluster.ravel()[: z[0].shape[0]],
        xlabel="β-amyloid burden (centiloid)",
        # intensities=np.exp(df_results.prob_c_all.ravel())[: z[0].shape[0]],
    )

    data.plot_2d_trajectories(
        model=None,
        savename=os.path.join(
            "figures",
            f"ADNI_r7_xval{n_splits}_our_model_plot"
            f"{n_clusters}_results_gm_vs_adnimem.pdf",
        ),
        title=f"",
        states=np.stack((x[..., 0], z[..., -1]), axis=-1),
        inferred_clusters=df_results.our_cluster.ravel()[: z[0].shape[0]],
        xlabel="ADNI-Mem",
        xlim=(np.nanmin(x[..., 0]) - 0.2, np.nanmax(x[..., 0]) + 0.2),
        ylabel="Gray matter density",
        arrow_width=0.01,
        # intensities=np.exp(df_results.prob_c_all.ravel())[: z[0].shape[0]],
    )

    final_id_ti = (
        df_results.loc[lambda df: ~df.diagnosis.isnull()]
        .sort_values("time")
        .groupby(level=0)
        .last()
        .set_index("time", append=True)
        .index
    )

    print("Our cluster vs. Our snapshot final")
    print(
        snapshot_xt_fin := pd.crosstab(
            df_results.set_index("time", append=True)
            .loc[final_id_ti]
            .our_cluster,
            df_results.set_index("time", append=True)
            .loc[final_id_ti]
            .our_cluster_snapshots,
        )
    )
    print(
        "{0:.3f} maintain labels".format(
            snapshot_xt_fin.values.diagonal().sum()
            / snapshot_xt_fin.values.sum()
        )
    )

    print("Our cluster vs. Our snapshot init")
    print(
        snapshot_xt_ini := pd.crosstab(
            df_results.loc[lambda df: df.time == 0].our_cluster,
            df_results.loc[lambda df: df.time == 0].our_cluster_snapshots,
        )
    )
    print(
        "{0:.3f} maintain labels".format(
            snapshot_xt_ini.values.diagonal().sum()
            / snapshot_xt_ini.values.sum()
        )
    )

    print("Ours using all data vs. missing hidden data")
    print(
        no_h_xt := pd.crosstab(
            df_results.groupby(level=0).first().our_cluster,
            df_results.groupby(level=0).first().our_cluster_no_hidden,
        )
    )
    print(
        "{0:.3f} maintain labels".format(
            no_h_xt.values.diagonal().sum() / no_h_xt.values.sum()
        )
    )
    print(
        "{0:.3f} maintain labels or move 1 label".format(
            (
                no_h_xt.values.diagonal().sum()
                + no_h_xt.values.diagonal(offset=1).sum()
                + no_h_xt.values.diagonal(offset=-1).sum()
            )
            / no_h_xt.values.sum()
        )
    )

    n_movers = no_h_xt.values.sum() - no_h_xt.values.diagonal().sum()
    n_move_1 = (
        no_h_xt.values.diagonal(offset=1).sum()
        + no_h_xt.values.diagonal(offset=-1).sum()
    )

    print(
        "{0:.3f} of those that do move, move only one label away".format(
            n_move_1 / n_movers
        )
    )

    print("Our cluster vs. GMM init preds")
    print(
        ours_x_gmm := pd.crosstab(
            df_results.loc[lambda df: df.time == 0].our_cluster,
            df_results.loc[lambda df: df.time == 0].gmm_init_predictions,
        )
    )

    print(
        "{0:.3f} maintain labels".format(
            ours_x_gmm.values.diagonal().sum() / ours_x_gmm.values.sum()
        )
    )
    print(
        "of those that move, {0:.3f} are lower triangle".format(
            np.tril(ours_x_gmm.values, -1).sum()
            / (
                np.tril(ours_x_gmm.values, -1).sum()
                + np.triu(ours_x_gmm.values, 1).sum()
            )
        )
    )

    print("Our cluster init vs. GMM init preds")
    print(
        ours_init_x_gmm := pd.crosstab(
            df_results.loc[lambda df: df.time == 0].our_cluster_snapshots,
            df_results.loc[lambda df: df.time == 0].gmm_init_predictions,
        )
    )

    print(
        "{0:.3f} maintain labels".format(
            ours_init_x_gmm.values.diagonal().sum()
            / ours_init_x_gmm.values.sum()
        )
    )
    print(
        "of those that move, {0:.3f} are lower triangle".format(
            np.tril(ours_init_x_gmm.values, -1).sum()
            / (
                np.tril(ours_init_x_gmm.values, -1).sum()
                + np.triu(ours_init_x_gmm.values, 1).sum()
            )
        )
    )

    pd.crosstab(
        df_results.loc[lambda df: df.time == 0].gmm_init_predictions.values,
        d_fin,
        rownames=["cluster"],
    ).reset_index().to_csv(
        os.path.join(
            "posthoc",
            "results",
            "contingency_table_gmm_init.csv",
        ),
        index=False,
    )

    st_sp.plot_metric_vs_clusters_over_time(
        metric=mmse[:, :, 0],
        assignments=df_results.our_cluster.ravel()[: z[0].shape[0]],
        metric_name="MMSE",
        savename=os.path.join(
            "figures",
            f"ADNI_r7_xval10_{n_clusters}clusterwise_MMSE_o_time.pdf",
        ),
        title="",  # f"{k} by cluster over time",
        xticks=np.arange(0, 2 * z.shape[0], 2),
        xlabel="Time (years)",
        legend_loc="upper right",
        # ylim=(10, 30),
        colors=data.cluster_colors,
    )

    cs = df_results.our_cluster.ravel()[: z[0].shape[0]]
    st_sp.pie(
        assignments=cs,
        savename=os.path.join(
            "figures", f"ADNI_r7_xval10_{n_clusters}_overall_pie_all.pdf"
        ),
        colors=data.cluster_colors,
        show=False,
    )

    st_sp.pie(
        assignments=df_fin.values.ravel(),
        savename=os.path.join(
            "figures",
            f"ADNI_r7_xval10_{n_clusters}_overall_pie_all_dx.pdf",
        ),
        colors=data.diagnosis_colors,
        cluster_ordering=data.diagnosis_list,
        show=False,
    )

    st_sp.pies_by_cluster(
        savename=os.path.join(
            "figures",
            f"ADNI_r7_xval10_{n_clusters}_pie_charts_all.pdf",
        ),
        categories=df_fin.values.ravel(),
        halo_colors=data.cluster_colors,
        category_ordering=data.diagnosis_list,
        clusters=cs,
        slice_colors=data.diagnosis_colors,
        legend_bbox_to_anchor=(1.75 + 0.25 * int(n_clusters == 3), 1),
        fig_length=6.0,
        fig_width=3.0,
        show=False,
    )

    st_sp.pies_by_cluster(
        savename=os.path.join(
            "figures",
            f"ADNI_r7_xval10_{n_clusters}_pie_charts_all_dx.pdf",
        ),
        categories=cs,
        halo_colors=data.diagnosis_colors,
        clusters=df_fin.values.ravel(),
        cluster_ordering=data.diagnosis_list,
        slice_colors=data.cluster_colors,
        legend_bbox_to_anchor=(1.75 + 0.25 * int(n_clusters == 3), 1),
        fig_length=6.0,
        fig_width=3.0,
        show=False,
    )

    df_biomarkers = pd.DataFrame(
        data={
            "ids": ids[:-1].ravel(),
            "amyl_prev": z[:-1, :, 0].ravel(),
            "gm_diff": np.diff(z[..., -1], axis=0).ravel(),
            "moca_diff": np.diff(x[..., -1], axis=0).ravel(),
            "adni_mem_diff": np.diff(x[..., 0], axis=0).ravel(),
            "cluster": df_results.our_cluster[: ids[:-1].size],
            "age": approx_age[:-1].ravel(),
        }
    ).loc[lambda df: ~df.isna().any(axis=1)]

    res = smf.mixedlm(
        "gm_diff ~ cluster * amyl_prev",
        df_biomarkers,
        groups=df_biomarkers["ids"],
    ).fit()
    fig, axs = plt.subplots(layout="constrained")
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    df_biomarkers.assign(
        color=lambda df: df.cluster.apply(
            dict(
                zip(
                    string.ascii_uppercase,
                    data.cluster_colors,
                )
            ).__getitem__
        )
    ).plot.scatter(x="amyl_prev", y="gm_diff", c="color", ax=axs)
    mn, mx = axs.get_xlim()
    for i, c in enumerate(string.ascii_uppercase[:n_clusters]):
        intercept = (
            res.params["Intercept"]
            if c == "A"
            else res.params["cluster[T.{}]".format(c)]
        )
        slope = (
            res.params["amyl_prev"]
            if c == "A"
            else res.params["cluster[T.{}]:amyl_prev".format(c)]
        )
        color = data.cluster_colors[i]
        axs.add_artist(
            mpl.lines.Line2D(
                [mn, mx],
                [intercept + mn * slope, intercept + mx * slope],
                color=color,
                label=c,
                zorder=5,
                linestyle=(
                    "solid",
                    "dashdot",
                    "dashed",
                    "dotted",
                    "densely dashdotted",
                    "loosely dashdotted",
                )[i],
            )
        )
        if lme_add_ci:
            n_mc, n_pts = 10000, 100
            intercept_bse = (
                res.bse["Intercept"]
                if c == "A"
                else res.bse["cluster[T.{}]".format(c)]
            )
            slope_bse = (
                res.bse["amyl_prev"]
                if c == "A"
                else res.bse["cluster[T.{}]:amyl_prev".format(c).format(c)]
            )
            intercepts = rng.normal(
                loc=intercept, scale=intercept_bse, size=(n_mc, 1)
            )
            slopes = rng.normal(loc=slope, scale=slope_bse, size=(n_mc, 1))
            pts = np.linspace(mn, mx, n_pts).reshape(1, -1)
            samples = slopes * pts + intercepts  # .shape = (n_mc,n_pts)
            q_lo_up = np.quantile(samples, [0.275, 0.975], axis=0)
            axs.fill_between(
                x=pts.ravel(),
                y1=q_lo_up[0],
                y2=q_lo_up[1],
                alpha=0.15,
                color=data.cluster_colors[i],
                linestyle=(
                    "solid",
                    "dashdot",
                    "dashed",
                    "dotted",
                    "densely dashdotted",
                    "loosely dashdotted",
                )[i],
            )
        plt.legend(fontsize="large")
        axs.set_xlabel("β-amyloid burden (centiloid)", fontsize="large")
        axs.set_ylabel("Grey matter density change", fontsize="large")
    fig.savefig(
        os.path.join(
            "figures", "gm_diff_vs_amyloid_{}c_affine.pdf".format(n_clusters)
        ),
        bbox_inches="tight",
        transparent=True,
    )

    res = smf.mixedlm(
        "adni_mem_diff ~ cluster * gm_diff",
        df_biomarkers,
        groups=df_biomarkers["ids"],
    ).fit()
    fig, axs = plt.subplots(layout="constrained")
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    df_biomarkers.assign(
        color=lambda df: df.cluster.apply(
            dict(
                zip(
                    string.ascii_uppercase,
                    data.cluster_colors,
                )
            ).__getitem__
        )
    ).plot.scatter(x="gm_diff", y="adni_mem_diff", c="color", ax=axs)
    mn, mx = axs.get_xlim()
    for i, c in enumerate(string.ascii_uppercase[:n_clusters]):
        intercept = (
            res.params["Intercept"]
            if c == "A"
            else res.params["cluster[T.{}]".format(c)]
        )
        slope = (
            res.params["gm_diff"]
            if c == "A"
            else res.params["cluster[T.{}]:gm_diff".format(c)]
        )
        color = data.cluster_colors[i]
        axs.add_artist(
            mpl.lines.Line2D(
                [mn, mx],
                [intercept + mn * slope, intercept + mx * slope],
                color=color,
                label=c,
                zorder=5,
                linestyle=(
                    "solid",
                    "dashdot",
                    "dashed",
                    "dotted",
                    "densely dashdotted",
                    "loosely dashdotted",
                )[i],
            )
        )
        if lme_add_ci:
            n_mc, n_pts = 10000, 100
            intercept_bse = (
                res.bse["Intercept"]
                if c == "A"
                else res.bse["cluster[T.{}]".format(c)]
            )
            slope_bse = (
                res.bse["gm_diff"]
                if c == "A"
                else res.bse["cluster[T.{}]:gm_diff".format(c).format(c)]
            )
            intercepts = rng.normal(
                loc=intercept, scale=intercept_bse, size=(n_mc, 1)
            )
            slopes = rng.normal(loc=slope, scale=slope_bse, size=(n_mc, 1))
            pts = np.linspace(mn, mx, n_pts).reshape(1, -1)
            samples = slopes * pts + intercepts  # .shape = (n_mc, n_pts)
            q_lo_up = np.quantile(samples, [0.275, 0.975], axis=0)
            axs.fill_between(
                x=pts.ravel(),
                y1=q_lo_up[0],
                y2=q_lo_up[1],
                alpha=0.15,
                color=data.cluster_colors[i],
                linestyle=(
                    "solid",
                    "dashdot",
                    "dashed",
                    "dotted",
                    "densely dashdotted",
                    "loosely dashdotted",
                )[i],
            )
        plt.legend(fontsize="large")
        axs.set_xlabel("Grey matter density change", fontsize="large")
        axs.set_ylabel("Change in ADNI-Mem", fontsize="large")
    fig.savefig(
        os.path.join(
            "figures",
            "adni_mem_diff_vs_gm_diff_{}c_affine.pdf".format(n_clusters),
        ),
        transparent=True,
        bbox_inches="tight",
    )

    df_results0 = df_results.assign(
        mmse=mmse.ravel(),
        age=approx_age.ravel(),
        amyloid=z[..., 0].ravel(),
        gm=z[..., 1].ravel(),
        adni_mem=x[..., 0].ravel(),
        adni_ef=x[..., 1].ravel(),
        adas13=x[..., 2].ravel(),
        moca=x[..., 3].ravel(),
    ).pipe(
        lambda x: x.join(
            x.loc[lambda df: df.time == 0].assign(
                mmse_init=lambda df: df.mmse,
                age_init=lambda df: df.age,
                amyloid_init=lambda df: df.amyloid,
                gm_init=lambda df: df.gm,
                adni_mem_init=lambda df: df.adni_mem,
                adni_ef_init=lambda df: df.adni_ef,
                adas13_init=lambda df: df.adas13,
                moca_init=lambda df: df.moca,
                our_index_snapshot_init=lambda df: df.our_index_snapshots,
                our_cluster_snapshot_init=lambda df: df.our_cluster_snapshots,
            )[
                [
                    "mmse_init",
                    "age_init",
                    "amyloid_init",
                    "gm_init",
                    "adni_mem_init",
                    "adni_ef_init",
                    "adas13_init",
                    "moca_init",
                    "our_index_snapshot_init",
                    "our_cluster_snapshot_init",
                ]
            ]
        )
    )
    df_final = (
        df_results0.assign(years=lambda df: 2 * df.time)
        .set_index("time", append=True)
        .loc[final_id_ti]
        .assign(
            ann_mmse_change=lambda df: (df.mmse - df.mmse_init) / df.years,
            ann_mmse_change_age_adjusted=lambda df: util_ph.regressed_out_effect_cv(
                df.ann_mmse_change.values.reshape(-1, 1),
                df.age_init.values.reshape(-1, 1),
            ),
        )
    )

    drop_outliers = True
    ycol = "ann_mmse_change_age_adjusted"
    yname = "Annualized MMSE change (age adjusted)"
    for xcol, xname in {
        "mmse_init": "MMSE",
        "moca_init": "MoCA",
        "our_index_snapshot_init": "MTM-derived index",
    }.items():
        in_col, out_col = (
            df_final[xcol].values,
            df_final[ycol].values,
        )
        is_not_outlier = np.abs(
            in_col - np.nanmean(in_col, axis=0, keepdims=True)
        ) < 3 * np.nanstd(in_col, axis=0, keepdims=True)
        is_finite = np.all(
            np.isfinite(np.column_stack([in_col, out_col])), axis=1
        )
        ids_to_keep = np.logical_and(
            is_finite, is_not_outlier if drop_outliers else True
        )
        print(
            "{:.2f} kept (μ={:.2f}, σ={:.2f})".format(
                ids_to_keep.astype(int).mean(),
                in_col[ids_to_keep].mean(),
                in_col[ids_to_keep].std(),
            )
        )

        fig, ax = plt.subplots(layout="constrained")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        sns.regplot(
            data=df_final[ids_to_keep],
            x=xcol,
            y=ycol,
            ax=ax,
            color="black",
            scatter=False,
        )
        for i, c in enumerate(string.ascii_uppercase[:n_clusters]):
            df_c = df_final[ids_to_keep].loc[
                lambda df: df.our_cluster_snapshot_init == c
            ]
            plt.scatter(
                df_c[xcol].values,
                df_c[ycol].values,
                color=data.cluster_colors[i],
                marker=("o", "v", "^", "s", "+", "x")[i],
                label=f"cluster {c}",
                s=2,
                alpha=0.9,
            )
        ax.set_xlabel("{} (at baseline)".format(xname), fontsize="large")
        ax.set_ylabel(yname, fontsize="large")
        ax.legend(fontsize="large", loc="lower right")
        fig.savefig(
            os.path.join(
                "figures",
                "{}_vs_{}_bl_{}c{}.pdf".format(
                    ycol,
                    xcol,
                    n_clusters,
                    "_no_outliers" if drop_outliers else "",
                ),
            ),
            transparent=True,
        )

    # export data for analysis in R
    df_biomarkers.to_csv(
        os.path.join(
            "posthoc", "results", "biomarkers_by_cluster_over_time.csv"
        )
    )
    df_final.to_csv(os.path.join("posthoc", "results", "prognostics_all.csv"))

    df_results0.rename(columns={"our_cluster": "cluster"}).assign(
        time_in_years=lambda df: 2 * df.time
    )[["time_in_years", "mmse", "age", "cluster"]].loc[
        lambda df: ~df.isna().any(axis=1)
    ].assign(
        mmse_age_adjusted=lambda df: util_ph.regressed_out_effect_cv(
            df.mmse.values.reshape(-1, 1), df.age.values.reshape(-1, 1)
        )
    ).to_csv(
        os.path.join("posthoc", "results", "mmse_by_cluster_over_time.csv")
    )
    pd.pivot(
        outcomes_sum[("ours", "test")].reset_index(),
        index="cluster",
        columns="diagnosis",
        values=("ours", "test"),
    ).to_csv(
        os.path.join("posthoc", "results", "contingency_table_ours_test.csv")
    )

    df_results0.loc[lambda df: df.time == 0].join(
        pd.read_csv(os.path.join("data", "adni-profiling.csv")).set_index(
            "RID"
        )
    )[
        [
            "our_index_snapshot_init",
            "amyloid_init",
            "gm_init",
            "adni_ef_init",
            "adas13_init",
            "moca_init",
            "adni_mem_init",
            "age_init",
            "ad_outcome",
            "days_to_ad_or_last_obs",
        ]
    ].to_csv(
        os.path.join("posthoc", "results", "survival_modelling_data.csv")
    )

    print("-" * 79, "Clusters by diagnosis", sep="\n")
    c = ("ours", "test")
    contingency = pd.pivot(
        outcomes_sum[c].reset_index(),
        index="cluster",
        columns="diagnosis",
        values=c,
    )
    print(
        contingency.div(contingency.sum(axis=0), axis=1)[data.diagnosis_list]
    )

    cog_init = [
        "adni_mem_init",
        "adni_ef_init",
        "moca_init",
        "adas13_init",
        "mmse_init",
    ]
    bio_init = ["amyloid_init", "gm_init"]

    df_final_prog = (
        df_final[
            ["our_index_snapshot_init"] + cog_init + bio_init + ["diagnosis"]
        ]
        .loc[lambda df: ~df.isna().any(axis=1)]
        .reset_index("time")
        .join(
            data.return_profiling_dataframe(ids[0])[
                ["ad_outcome", "days_to_ad_or_last_obs"]
            ]
        )
    )

    """
    run cross-validated logistic regression on baseline features to predict
    conversion to AD
    """

    batch_aucs_by_feature = {
        tuple(x[:6] for x in c_list): util_ph.stratified_logit_cv_metrics(
            df_final_prog[c_list].values,
            df_final_prog[["diagnosis"]]
            .apply(lambda x: x == "AD")
            .astype(int)
            .values,
        )
        for c_list in [
            ["our_index_snapshot_init"],
            bio_init,
            *map(list, itertools.product(bio_init, cog_init)),
        ]
    }

    pd.DataFrame.from_dict(
        batch_aucs_by_feature, orient="index"
    ).stack().to_frame().reset_index().set_axis(
        ["feature", "batch", "auc"], axis=1
    ).to_csv(
        os.path.join(
            "posthoc",
            "results",
            "paired_prognostic_aucs_from_baseline_meas.csv",
        ),
        index=False,
    )

    """ distribution of diagnostic outcome given cluster
    """

    print("Cluster assignment rates (training)")
    print(
        tr_cl := outcomes_by_cluster_train.groupby(level=0)
        .sum()
        .apply(lambda x: x / x.sum())
    )
    tr_cl.to_csv(
        os.path.join(
            "results",
            "cluster_rates_training.csv",
        )
    )

    print("Outcomes by cluster (training)")
    print(
        tr_out_x_cl := outcomes_by_cluster_train.fillna(0.0)
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
    )
    tr_out_x_cl.to_csv(
        os.path.join(
            "results",
            "outcomes_by_cluster_training.csv",
        )
    )

    """ distribution of of cluster given diagnostic outcome
    """

    # print("Outcomes rates (training)")
    # print(
    #     out_cl := outcomes_by_cluster_train.groupby(level=1)
    #     .sum()
    #     .apply(lambda x: x / x.sum())
    #     .loc[data.diagnosis_list]
    # )
    # out_cl.to_csv(
    #     os.path.join(
    #         "results",
    #         "outcome_rates_training.csv",
    #     )
    # )
    #
    # print("Cluster by outcome (training)")
    # print(tr_cl_x_out := clusters_by_outcome_train.fillna(0.0))
    # tr_cl_x_out.to_csv(
    #     os.path.join(
    #         "results",
    #         "cluster_by_outcomes_training.csv",
    #     )
    # )

    """
    print out conversion rate comparisons for confusion matrix with baseline 
    GMM
    """
    print("Conversion rate pivots for ours vs. GMM @ baseline")

    xt = (
        df_results.loc[
            lambda df: df.time == 0,
            ["our_cluster_snapshots", "gmm_init_predictions", "our_cluster"],
        ]
        .join(pd.DataFrame(index=ids[0].ravel(), data={"final_dx": d_fin}))
        .assign(fin_AD=lambda df: (df.final_dx == "AD").astype(int))
    )

    for c in ["our_cluster", "our_cluster_snapshots"]:
        print(
            xt.groupby([c, "gmm_init_predictions"])
            .agg(AD_rate=("fin_AD", "mean"))
            .reset_index()
            .pivot(
                columns="gmm_init_predictions",
                index=c,
                values="AD_rate",
            )
            .fillna(0.0)
        )

    print("Clusters by trajectory length")
    print(
        df_final.reset_index()
        .assign(length=lambda df: df.time + 1)
        .groupby(["length", "our_cluster"])
        .agg(ct=("length", "count"))
        .reset_index()
        .pivot(index="our_cluster", columns="length", values="ct")
    )

    """
    Plot trajectories coloured to compare MTM and GMM-init predictions
    """
    # cf_mtm_gmm = df_results.loc[lambda df: df.time == 0].apply(
    #     lambda x: "MTM predicts healthier"
    #     if x.our_cluster < x.gmm_init_predictions
    #     else (
    #         "same predictions"
    #         if x.our_cluster == x.gmm_init_predictions
    #         else "GMM predicts healthier"
    #     ),
    #     axis=1,
    # )
    #
    # data.plot_2d_trajectories(
    #     model=None,
    #     savename=os.path.join(
    #         "figures",
    #         f"ADNI_r7_xval{n_splits}_cf_unsupervised_models.pdf",
    #     ),
    #     title="",
    #     states=z,
    #     xlabel="β-amyloid burden (centiloid)",
    #     inferred_clusters=cf_mtm_gmm,
    #     cluster_ordering=[
    #         "MTM predicts healthier",
    #         "same predictions",
    #         "GMM predicts healthier",
    #     ],
    # )
    #
    # data.plot_2d_trajectories(
    #     model=None,
    #     savename=os.path.join(
    #         "figures",
    #         f"ADNI_r7_xval{n_splits}_cf_unsupervised_models"
    #         f"_gm_vs_adnimem.pdf",
    #     ),
    #     title=f"",
    #     states=np.stack((x[..., 0], z[..., -1]), axis=-1),
    #     xlabel="ADNI-Mem",
    #     xlim=(np.nanmin(x[..., 0]) - 0.2, np.nanmax(x[..., 0]) + 0.2),
    #     ylabel="Gray matter density",
    #     arrow_width=0.01,
    #     inferred_clusters=cf_mtm_gmm,
    #     cluster_ordering=[
    #         "MTM predicts healthier",
    #         "same predictions",
    #         "GMM predicts healthier",
    #     ],
    # )

    # break into plots based on our cluster assignment
    # for c in string.ascii_uppercase[:n_clusters]:
    #     c_ids = df_results.loc[lambda df: df.time == 0].our_cluster == c
    #     data.plot_2d_trajectories(
    #         model=None,
    #         savename=os.path.join(
    #             "figures",
    #             f"ADNI_r7_xval{n_splits}_cf_unsupervised_models_{c}.pdf",
    #         ),
    #         title="",
    #         states=z[:, c_ids],
    #         xlabel="β-amyloid burden (centiloid)",
    #         inferred_clusters=cf_mtm_gmm[c_ids],
    #         cluster_ordering=[
    #             "MTM predicts healthier",
    #             "same predictions",
    #             "GMM predicts healthier",
    #         ],
    #     )

    """compare assignment on baseline vs. on trajectories
    """

    # for c in string.ascii_uppercase[:n_clusters]:
    #     c_ids = df_results.loc[lambda df: df.time == 0].our_cluster == c
    #     data.plot_2d_trajectories(
    #         model=None,
    #         savename=os.path.join(
    #             "figures",
    #             f"ADNI_r7_xval{n_splits}_our_cl_init_on_our_cluster_{c}.pdf",
    #         ),
    #         title="",
    #         states=z[:, c_ids],
    #         xlabel="β-amyloid burden (centiloid)",
    #         inferred_clusters=df_results.loc[
    #             lambda df: df.time == 0
    #         ].our_cluster_snapshots[c_ids],
    #         cluster_ordering=list(string.ascii_uppercase[:n_clusters]),
    #     )

    """
    plot histograms for training run cluster assignment
    """
    tr_cl = pd.concat(
        [
            df_i.loc[lambda x: x.time == 0, "our_cluster"]
            for df_i in d_tr.values()
        ],
        axis=1,
    ).fillna("X")

    hist_cl = np.column_stack(
        [
            np.sum(tr_cl.values == s, axis=1)
            for s in string.ascii_uppercase[:n_clusters]
        ]
    )
    assert np.all(
        np.sum(hist_cl, axis=1) == 9
    )  # everyone appears in exactly 9 of the 10 training sets

    # for i in range(n_clusters):
    #     c = string.ascii_uppercase[i]
    #     fig, ax = plt.subplots()
    #     ax.hist(
    #         hist_cl[:, i],
    #         bins=np.arange(-0.5, 10.5),
    #         density=True,
    #         color="#0072CE",
    #     )
    #     ax.set_xticks(np.arange(10))
    #     ax.set_ylim((0, 1))
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["top"].set_visible(False)
    #     ax.set_xlabel(
    #         "Number of training runs where "
    #         f"sample was assigned to cluster {c}",
    #         fontsize="large",
    #     )
    #     ax.set_ylabel(f"Empirical frequency", fontsize="large")
    #     plt.savefig(
    #         os.path.join(
    #             "figures",
    #             f"ADNI_r7_xval{n_splits}_tr_cl_{c}_hist.pdf",
    #         )
    #     )

    tr_te = tr_cl.join(
        df_results.loc[lambda x: x.time == 0, ["our_cluster"]].rename(
            columns={"our_cluster": "test"}
        )
    )

    for c in string.ascii_uppercase[:n_clusters]:
        tr_c = tr_te.loc[lambda x: x.test == c].drop(columns="test")
        ct_c = np.sum(tr_c.values == c, axis=1)
        fig, ax = plt.subplots()
        ax.hist(
            ct_c,
            bins=np.arange(-0.5, 10.5),
            density=True,
            color="black",
        )
        ax.set_xticks(np.arange(10))
        ax.set_ylim((0, 1))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel(
            f"Cluster {c} assignment (number of training runs)",
            fontsize="large",
        )
        ax.set_ylabel(f"Frequency", fontsize="large")
        plt.savefig(
            os.path.join(
                "figures",
                f"ADNI_r7_xval{n_splits}_tr_te_cl_{c}_hist.pdf",
            )
        )


if __name__ == "__main__":
    main()


"""
           age  is_female  edu_yrs  apoe4_pos
cluster                                      
A       70.582      0.478   16.748      0.201
B       71.634      0.509   16.189      0.438
C       75.705      0.441   16.528      0.410
D       75.031      0.463   15.756      0.683
outcomes by cluster
                   ours                                               \
                   test testnh testsnapshotsinit testsnapshotsinitnh   
cluster diagnosis                                                      
A       CN        0.654  0.649             0.638               0.613   
        sMCI      0.333  0.346             0.345               0.340   
        pMCI      0.006  0.005             0.006               0.010   
        AD        0.006  0.000             0.011               0.037   
B       CN        0.462  0.464             0.406               0.416   
        sMCI      0.467  0.470             0.461               0.426   
        pMCI      0.018  0.030             0.022               0.051   
        AD        0.053  0.036             0.111               0.107   
C       CN        0.311  0.227             0.284               0.241   
        sMCI      0.484  0.511             0.420               0.483   
        pMCI      0.075  0.057             0.071               0.034   
        AD        0.130  0.206             0.225               0.241   
D       CN        0.024  0.000             0.042               0.000   
        sMCI      0.171  0.099             0.208               0.132   
        pMCI      0.037  0.070             0.042               0.053   
        AD        0.768  0.831             0.708               0.816   
                                                 
                  testsnapshots testsnapshotsnh  
cluster diagnosis                                
A       CN                0.628           0.661  
        sMCI              0.360           0.333  
        pMCI              0.006           0.005  
        AD                0.006           0.000  
B       CN                0.475           0.443  
        sMCI              0.456           0.489  
        pMCI              0.019           0.029  
        AD                0.051           0.040  
C       CN                0.340           0.231  
        sMCI              0.477           0.523  
        pMCI              0.072           0.069  
        AD                0.111           0.177  
D       CN                0.042           0.026  
        sMCI              0.208           0.103  
        pMCI              0.042           0.051  
        AD                0.708           0.821  
clusters by approach
         ours                                                             \
         test testnh testsnapshotsinit testsnapshotsinitnh testsnapshots   
cluster                                                                    
A       0.278  0.335             0.305               0.335         0.287   
B       0.296  0.294             0.315               0.345         0.277   
C       0.282  0.247             0.296               0.254         0.268   
D       0.144  0.124             0.084               0.067         0.168   
                         
        testsnapshotsnh  
cluster                  
A                 0.331  
B                 0.305  
C                 0.228  
D                 0.137  
population-level cluster prevalences
   our_cluster  our_cluster_snapshots  gmm_init_predictions
A        0.278                  0.305                 0.450
B        0.296                  0.315                 0.331
C        0.282                  0.296                 0.144
D        0.144                  0.084                 0.075
diagnostic outcomes by cluster
             our_cluster  our_cluster_snapshots  gmm_init_predictions
  diagnosis                                                          
A CN               0.654                  0.638                 0.564
  sMCI             0.333                  0.345                 0.424
  pMCI             0.006                  0.006                 0.004
  AD               0.006                  0.011                 0.008
B CN               0.462                  0.406                 0.402
  sMCI             0.467                  0.461                 0.434
  pMCI             0.018                  0.022                 0.026
  AD               0.053                  0.111                 0.138
C CN               0.311                  0.284                 0.146
  sMCI             0.484                  0.420                 0.305
  pMCI             0.075                  0.071                 0.098
  AD               0.130                  0.225                 0.451
D CN               0.024                  0.042                 0.023
  sMCI             0.171                  0.208                 0.186
  pMCI             0.037                  0.042                 0.116
  AD               0.768                  0.708                 0.674
Our cluster vs. Our snapshot final
our_cluster_snapshots    A    B    C   D
our_cluster                             
A                      141   11    7   0
B                       15  138   14   2
C                        7    7  129  18
D                        1    2    3  76
0.848 maintain labels
Our cluster vs. Our snapshot init
our_cluster_snapshots    A    B    C   D
our_cluster                             
A                      142    9    8   0
B                       21  142    4   2
C                       11   15  133   2
D                        0   14   24  44
0.807 maintain labels
Ours using all data vs. missing hidden data
our_cluster_no_hidden    A   B   C   D
our_cluster                           
A                      103  38  18   0
B                       50  84  33   2
C                       38  40  78   5
D                        0   6  12  64
0.576 maintain labels
0.888 maintain labels or move 1 label
0.736 of those that do move, move only one label away
Our cluster vs. GMM init preds
gmm_init_predictions    A   B   C   D
our_cluster                          
A                     153   6   0   0
B                      62  87  18   2
C                      39  83  30   9
D                       3  13  34  32
0.529 maintain labels
of those that move, 0.870 are lower triangle
Our cluster init vs. GMM init preds
gmm_init_predictions     A   B   C   D
our_cluster_snapshots                 
A                      169   5   0   0
B                       53  96  27   4
C                       35  84  35  15
D                        0   4  20  24
0.567 maintain labels
of those that move, 0.794 are lower triangle
0.98 kept (μ=28.40, σ=1.82)
0.98 kept (μ=24.38, σ=3.01)
0.98 kept (μ=-25.87, σ=31.57)
-------------------------------------------------------------------------------
Clusters by diagnosis
diagnosis    CN  sMCI  pMCI    AD
cluster                          
A         0.444 0.237 0.053 0.011
B         0.333 0.353 0.158 0.096
C         0.214 0.348 0.632 0.223
D         0.009 0.062 0.158 0.670
Cluster assignment rates (training)
split       0     1     2     3     4     5     6     7     8     9
cluster                                                            
A       0.320 0.212 0.222 0.296 0.307 0.298 0.220 0.329 0.257 0.243
B       0.287 0.346 0.453 0.317 0.298 0.257 0.216 0.284 0.298 0.288
C       0.318 0.265 0.210 0.280 0.280 0.323 0.403 0.216 0.294 0.296
D       0.076 0.177 0.115 0.107 0.115 0.123 0.161 0.171 0.152 0.173
Outcomes by cluster (training)
split                 0     1     2     3     4     5     6     7     8     9
cluster diagnosis                                                            
A       CN        0.604 0.734 0.763 0.592 0.608 0.556 0.717 0.627 0.598 0.640
        sMCI      0.396 0.266 0.237 0.401 0.380 0.444 0.283 0.373 0.402 0.360
        pMCI      0.000 0.000 0.000 0.000 0.006 0.000 0.000 0.000 0.000 0.000
        AD        0.000 0.000 0.000 0.007 0.006 0.000 0.000 0.000 0.000 0.000
B       CN        0.490 0.393 0.416 0.497 0.451 0.591 0.468 0.418 0.503 0.466
        sMCI      0.449 0.528 0.506 0.429 0.477 0.386 0.414 0.459 0.464 0.480
        pMCI      0.020 0.034 0.030 0.031 0.033 0.008 0.063 0.041 0.007 0.020
        AD        0.041 0.045 0.047 0.043 0.039 0.015 0.054 0.082 0.026 0.034
C       CN        0.233 0.419 0.250 0.271 0.285 0.247 0.391 0.396 0.344 0.349
        sMCI      0.454 0.434 0.454 0.451 0.458 0.482 0.488 0.414 0.430 0.467
        pMCI      0.080 0.059 0.093 0.076 0.069 0.084 0.034 0.036 0.079 0.079
        AD        0.233 0.088 0.204 0.201 0.188 0.187 0.087 0.153 0.146 0.105
D       CN        0.026 0.066 0.000 0.018 0.000 0.016 0.036 0.045 0.038 0.056
        sMCI      0.077 0.209 0.119 0.109 0.102 0.079 0.145 0.261 0.167 0.180
        pMCI      0.000 0.044 0.017 0.000 0.017 0.032 0.048 0.057 0.051 0.045
        AD        0.897 0.681 0.864 0.873 0.881 0.873 0.771 0.636 0.744 0.719
Outcomes rates (training)
split         0     1     2     3     4     5     6     7     8     9
diagnosis                                                            
CN        0.409 0.414 0.411 0.411 0.401 0.399 0.422 0.418 0.411 0.403
sMCI      0.405 0.391 0.391 0.393 0.399 0.397 0.372 0.387 0.393 0.395
pMCI      0.031 0.035 0.035 0.031 0.033 0.033 0.035 0.029 0.033 0.037
AD        0.154 0.160 0.163 0.165 0.167 0.171 0.171 0.165 0.163 0.165
Cluster by outcome (training)
split                 0     1     2     3     4     5     6     7     8     9
diagnosis cluster                                                            
CN        A       0.471 0.376 0.412 0.427 0.466 0.415 0.373 0.493 0.374 0.386
          B       0.343 0.329 0.460 0.384 0.335 0.380 0.240 0.284 0.365 0.333
          C       0.181 0.268 0.128 0.185 0.199 0.200 0.373 0.205 0.246 0.256
          D       0.005 0.028 0.000 0.005 0.000 0.005 0.014 0.019 0.014 0.024
sMCI      A       0.312 0.144 0.134 0.302 0.293 0.333 0.168 0.317 0.262 0.222
          B       0.317 0.468 0.587 0.347 0.356 0.250 0.241 0.337 0.351 0.350
          C       0.356 0.294 0.244 0.322 0.322 0.392 0.529 0.231 0.322 0.350
          D       0.014 0.095 0.035 0.030 0.029 0.025 0.063 0.116 0.064 0.079
pMCI      A       0.000 0.000 0.000 0.000 0.059 0.000 0.000 0.000 0.000 0.000
          B       0.188 0.333 0.389 0.312 0.294 0.059 0.389 0.400 0.059 0.158
          C       0.812 0.444 0.556 0.688 0.588 0.824 0.389 0.267 0.706 0.632
          D       0.000 0.222 0.056 0.000 0.059 0.118 0.222 0.333 0.235 0.211
AD        A       0.000 0.000 0.000 0.012 0.012 0.000 0.000 0.000 0.000 0.000
          B       0.076 0.098 0.131 0.082 0.070 0.023 0.068 0.141 0.048 0.059
          C       0.481 0.146 0.262 0.341 0.314 0.352 0.205 0.200 0.262 0.188
          D       0.443 0.756 0.607 0.565 0.605 0.625 0.727 0.659 0.690 0.753
Conversion rate pivots for ours vs. GMM @ baseline
gmm_init_predictions     A     B     C     D
our_cluster                                 
A                    0.007 0.000 0.000 0.000
B                    0.000 0.080 0.111 0.000
C                    0.000 0.133 0.267 0.222
D                    0.333 0.615 0.794 0.844
gmm_init_predictions      A     B     C     D
our_cluster_snapshots                        
A                     0.006 0.200 0.000 0.000
B                     0.000 0.104 0.296 0.500
C                     0.029 0.167 0.429 0.533
D                     0.000 0.250 0.700 0.792
Clusters by trajectory length
length        2   3   4
our_cluster            
A            98  53   8
B            92  63  14
C            88  62  11
D            59  16   7
"""
