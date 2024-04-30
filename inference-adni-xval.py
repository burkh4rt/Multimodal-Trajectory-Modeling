#!/usr/bin/env python3

"""
performs cross-validated inference and harmonizes labels across folds
"""

import gzip
import os
import pickle
import string

import numpy as np
import pandas as pd
import sklearn.mixture as skl_mix
import sklearn.model_selection as skl_mdl_sel
import tqdm as tq

from framework import marginalizable_mixture_model as mixmodel
from framework_extended import state_space_model_classifier as ssm_c
from framework_extended import state_space_model_linear_gaussian as ssm_lg
from util import util_adni as data
from util import util_state_space as state_space_util

n_splits, n_clusters = 10, 4
alpha = 1.0


def main():
    z, x, d, ids, time = data.get_trajectories()
    z_dim = z.shape[-1]
    markers = np.concatenate([z, x], axis=-1)

    length = np.argmin(
        np.all(np.isfinite(np.concatenate([x, z], axis=-1)), axis=-1), axis=0
    )
    length[length == 0] = x.shape[0]

    df_results = pd.DataFrame(
        index=pd.MultiIndex.from_arrays(
            [np.row_stack(ids).flatten(), np.row_stack(time).flatten()],
            names=["id", "time"],
        ),
        columns=[
            "split",
            "our_cluster",
            "our_cluster_snapshots",
            "our_cluster_snapshots_no_hidden",
            "our_cluster_no_hidden",
        ],
    ).assign(diagnosis=np.row_stack(d).flatten())

    df_posterior = pd.DataFrame(
        index=ids[0].ravel(),
        columns=["prob_" + s for s in string.ascii_uppercase[:n_clusters]],
    )

    df_init_post = df_posterior.copy()

    df_gmm = pd.DataFrame(
        index=ids[0].ravel(),
        columns=["prob_" + s for s in string.ascii_uppercase[:n_clusters]],
    )

    dict_training_results = dict()

    """
    run training and prediction for each fold / split
    """

    for i_split, (train_mask, test_mask) in tq.tqdm(
        enumerate(
            skl_mdl_sel.KFold(
                n_splits=n_splits, shuffle=True, random_state=42
            ).split(ids[0].ravel())
        ),
        total=n_splits,
        desc="folds",
        position=1,
    ):
        # prepare data
        mtrain, dtrain, idtrain, timetrain, ztrain, xtrain = (
            markers[:, train_mask],
            d[:, train_mask],
            ids[:, train_mask],
            time[:, train_mask],
            z[:, train_mask],
            x[:, train_mask],
        )
        mtest, dtest, idtest, timetest, ztest, xtest = (
            markers[:, test_mask],
            d[:, test_mask],
            ids[:, test_mask],
            time[:, test_mask],
            z[:, test_mask],
            x[:, test_mask],
        )

        # record split assignment
        df_results.loc[
            lambda df: df.index.get_level_values("id").isin(
                idtest.ravel().tolist()
            ),
            "split",
        ] = i_split

        """
        our predictions
        """

        mtrain_ours = mtrain.copy()
        (
            mtrain_ours[:, :, : z.shape[-1]],
            std_params,
        ) = state_space_util.standardize(
            mtrain_ours[:, :, : z.shape[-1]], return_params=True
        )

        mtest_ours = mtest.copy()
        mtest_ours[:, :, : z.shape[-1]] = state_space_util.standardize(
            mtest_ours[:, :, : z.shape[-1]], params=std_params
        )

        mtest_ours_no_hidden = mtest_ours.copy()
        mtest_ours_no_hidden[:, :, :z_dim] = np.nan

        # train our mixture model
        best_mdl = mixmodel.MMLinGaussSS_marginalizable(
            n_clusters=n_clusters,
            states=mtrain_ours[:, :, : z.shape[-1]],
            observations=mtrain_ours[:, :, z.shape[-1] :],
            init="k-means",
            alpha=alpha,
        ).train_with_multiple_random_starts(n_starts=1000, use_cache=True)
        data.set_model_correspondence(best_mdl, dtrain)

        best_mdl.to_pickle()

        ctrain = np.array(
            [
                best_mdl.correspondence[c]
                for c in best_mdl.mle_cluster_assignment()
            ]
        )

        # test our mixture model
        ctest_all = np.array(
            [
                best_mdl.correspondence[c]
                for c in best_mdl.mle_cluster_assignment(
                    states=mtest_ours[:, :, :z_dim],
                    observations=mtest_ours[:, :, z_dim:],
                )
            ]
        )

        assignments, probs, prenorm = best_mdl.mle_cluster_assignment(
            states=mtest_ours[:, :, :z_dim],
            observations=mtest_ours[:, :, z_dim:],
            return_probs=True,
            return_prenormalized_log_probs=True,
        )
        df_posterior.loc[idtest[0].ravel()] = probs[
            [
                best_mdl.inverse_correspondence[s]
                for s in string.ascii_uppercase[:n_clusters]
            ]
        ].T

        _, probs0 = best_mdl.mle_cluster_assignment(
            states=state_space_util.mask_all_but_time_i(
                mtest_ours[..., :z_dim], 0
            ),
            observations=state_space_util.mask_all_but_time_i(
                mtest_ours[..., z_dim:], 0
            ),
            return_probs=True,
        )
        df_init_post.loc[idtest[0].ravel()] = probs0[
            [
                best_mdl.inverse_correspondence[s]
                for s in string.ascii_uppercase[:n_clusters]
            ]
        ].T

        prob_c_all = np.array(
            [prenorm[a, i] for i, a in enumerate(assignments)]
        )

        indextest_all = best_mdl.cluster_assignment_index(
            states=mtest_ours[:, :, :z_dim],
            observations=mtest_ours[:, :, z_dim:],
        )

        z_test_parc, x_test_parc = state_space_util.parcellate_arrays(
            mtest_ours[:, :, :z_dim], mtest_ours[:, :, z_dim:]
        )

        ctest_snapshots = np.array(
            [
                best_mdl.correspondence[c]
                for c in best_mdl.mle_cluster_assignment(
                    states=z_test_parc,
                    observations=x_test_parc,
                )
            ]
        )

        indextest_snapshots = best_mdl.cluster_assignment_index(
            states=z_test_parc,
            observations=x_test_parc,
        )

        ctest_snapshot_x = np.array(
            [
                best_mdl.correspondence[c]
                for c in best_mdl.mle_cluster_assignment(
                    states=np.nan * np.ones_like(z_test_parc),
                    observations=x_test_parc,
                )
            ]
        )

        mtest_ours_no_hidden = mtest_ours.copy()
        mtest_ours_no_hidden[:, :, :z_dim] = np.nan
        ctest_no_hidden = np.array(
            [
                best_mdl.correspondence[c]
                for c in best_mdl.mle_cluster_assignment(
                    states=mtest_ours_no_hidden[:, :, :z_dim],
                    observations=mtest_ours_no_hidden[:, :, z_dim:],
                )
            ]
        )

        indextest_no_hidden = best_mdl.cluster_assignment_index(
            states=mtest_ours_no_hidden[:, :, :z_dim],
            observations=mtest_ours_no_hidden[:, :, z_dim:],
        )

        indextest_no_hidden_init = best_mdl.cluster_assignment_index(
            states=state_space_util.mask_all_but_time_i(
                mtest_ours_no_hidden[:, :, :z_dim], 0
            ),
            observations=state_space_util.mask_all_but_time_i(
                mtest_ours_no_hidden[:, :, z_dim:], 0
            ),
        )

        """
        supervised classifier
        """
        clssfr = ssm_c.StateSpaceModelClassifier(
            component_model=ssm_lg.StateSpaceLinearGaussian
        ).fit(
            data=(
                mtrain_ours[:, :, : z.shape[-1]],
                mtrain_ours[:, :, z.shape[-1] :],
            ),
            labels=data.get_final_diagnoses(dtrain),
        )
        c_sprvsd = clssfr.predict(
            data=(
                mtest_ours[:, :, : z.shape[-1]],
                mtest_ours[:, :, z.shape[-1] :],
            )
        )

        """
        Gaussian mixture model on initial states & measurements alone
        """

        gmm_init = skl_mix.GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            max_iter=1000,
            init_params="kmeans",
            random_state=42,
        )
        gmm_init_train_preds = gmm_init.fit_predict(
            np.concatenate([ztrain, xtrain], axis=-1)[0]
        )
        gmm_init_correspondence = dict(
            zip(
                np.argsort(
                    [
                        np.mean(
                            data.get_final_diagnoses(dtrain)[
                                gmm_init_train_preds == c
                            ]
                            == "AD"
                        )
                        for c in range(n_clusters)
                    ]
                ),
                string.ascii_uppercase,
            )
        )
        gmm_train_preds = np.array(
            [gmm_init_correspondence[p] for p in gmm_init_train_preds]
        )
        gmm_test_preds = np.array(
            [
                gmm_init_correspondence[p]
                for p in gmm_init.predict(
                    np.concatenate([ztest, xtest], axis=-1)[0]
                )
            ]
        )

        df_gmm.loc[idtest[0].ravel()] = gmm_init.predict_proba(
            np.concatenate([ztest, xtest], axis=-1)[0]
        )[
            :,
            sorted(
                gmm_init_correspondence.keys(),
                key=gmm_init_correspondence.__getitem__,
            ),
        ]

        """
        store results
        """
        df_train = pd.DataFrame(
            data={
                "idx": idtrain.ravel(),
                "time": timetrain.ravel(),
                "diagnosis": dtrain.ravel(),
            }
        ).set_index("idx")
        df_train["our_cluster"] = df_train.index.to_series().apply(
            lambda id: dict(zip(idtrain[0].ravel(), ctrain.ravel()))[id]
        )
        df_train["gmm_init"] = df_train.index.to_series().apply(
            lambda id: dict(zip(idtrain[0].ravel(), gmm_train_preds.ravel()))[
                id
            ]
        )

        df_results.loc[lambda df: df.split == i_split, "our_cluster"] = (
            df_results.loc[lambda df: df.split == i_split]
            .index.get_level_values("id")
            .to_series()
            .apply(
                lambda id: dict(zip(idtest[0].ravel(), ctest_all.ravel()))[id]
            )
            .values
        )

        df_results.loc[lambda df: df.split == i_split, "prob_c_all"] = (
            df_results.loc[lambda df: df.split == i_split]
            .index.get_level_values("id")
            .to_series()
            .apply(
                lambda id: dict(zip(idtest[0].ravel(), prob_c_all.ravel()))[id]
            )
            .values
        )

        df_results.loc[lambda df: df.split == i_split, "our_index"] = (
            df_results.loc[lambda df: df.split == i_split]
            .index.get_level_values("id")
            .to_series()
            .apply(
                lambda id: dict(zip(idtest[0].ravel(), indextest_all.ravel()))[
                    id
                ]
            )
            .values
        )

        df_results.loc[
            lambda df: df.split == i_split,
            "our_cluster_no_hidden",
        ] = (
            df_results.loc[lambda df: df.split == i_split]
            .index.get_level_values("id")
            .to_series()
            .apply(
                lambda id: dict(
                    zip(idtest[0].ravel(), ctest_no_hidden.ravel())
                )[id]
            )
            .values
        )

        df_results.loc[
            lambda df: df.split == i_split,
            "our_index_no_hidden",
        ] = (
            df_results.loc[lambda df: df.split == i_split]
            .index.get_level_values("id")
            .to_series()
            .apply(
                lambda id: dict(
                    zip(
                        idtest[0].ravel(),
                        indextest_no_hidden.ravel(),
                    )
                )[id]
            )
            .values
        )

        df_results.loc[
            lambda df: df.split == i_split,
            "our_index_no_hidden_init",
        ] = (
            df_results.loc[lambda df: df.split == i_split]
            .index.get_level_values("id")
            .to_series()
            .apply(
                lambda id: dict(
                    zip(
                        idtest[0].ravel(),
                        indextest_no_hidden_init.ravel(),
                    )
                )[id]
            )
            .values
        )

        df_results.loc[
            lambda df: df.split == i_split,
            "supervised_classifier_predictions",
        ] = (
            df_results.loc[lambda df: df.split == i_split]
            .index.get_level_values("id")
            .to_series()
            .apply(
                lambda id: dict(zip(idtest[0].ravel(), c_sprvsd.ravel()))[id]
            )
            .values
        )

        df_results.loc[
            lambda df: df.split == i_split,
            "gmm_init_predictions",
        ] = (
            df_results.loc[lambda df: df.split == i_split]
            .index.get_level_values("id")
            .to_series()
            .apply(
                lambda id: dict(
                    zip(idtest[0].ravel(), gmm_test_preds.ravel())
                )[id]
            )
            .values
        )

        df_results.loc[
            zip(
                np.tile(idtest.ravel(), np.max(timetest) + 1),
                timetest.ravel(),
            ),
            "our_cluster_snapshots",
        ] = ctest_snapshots

        df_results.loc[
            zip(
                np.tile(idtest.ravel(), np.max(timetest) + 1),
                timetest.ravel(),
            ),
            "our_index_snapshots",
        ] = indextest_snapshots

        df_results.loc[
            zip(
                np.tile(idtest.ravel(), np.max(timetest) + 1),
                timetest.ravel(),
            ),
            "our_cluster_snapshots_no_hidden",
        ] = ctest_snapshot_x

        dict_training_results |= {i_split: df_train}

        os.makedirs("results", exist_ok=True)
        os.makedirs(os.path.join("posthoc", "results"), exist_ok=True)

    df_results.astype({"split": int}).to_csv(
        os.path.join(
            "results",
            f"ADNI_r7_xval{n_splits}_{n_clusters}clusters_results.csv",
        )
    )

    df_posterior.to_csv(
        os.path.join(
            "posthoc",
            "results",
            f"ADNI_r7_xval{n_splits}_{n_clusters}clusters_mtm_post.csv",
        ),
        index_label="id",
    )

    df_init_post.to_csv(
        os.path.join(
            "posthoc",
            "results",
            f"ADNI_r7_xval{n_splits}_{n_clusters}clusters_mtm_init_post.csv",
        ),
        index_label="id",
    )

    df_gmm.to_csv(
        os.path.join(
            "posthoc",
            "results",
            f"ADNI_r7_xval{n_splits}_{n_clusters}clusters_gmm_init_post.csv",
        ),
        index_label="id",
    )

    with gzip.open(
        os.path.join(
            "results",
            f"ADNI_r7_xval{n_splits}_{n_clusters}clusters_results.p.gz",
        ),
        "wb",
    ) as f:
        pickle.dump(dict_training_results, f)


if __name__ == "__main__":
    main()
