#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for post-hoc
"""

import warnings

import numpy as np
import scipy.stats as sp_stats
import sklearn.linear_model as skl_lm
import sklearn.metrics as skl_mets
import sklearn.model_selection as skl_mdl_sel


def regressed_out_effect_cv(
    regressand: np.array, effect: np.array, model=skl_lm.RidgeCV()
):
    """regressed out the effect of `effect` from `regressand` in a
    cross-validated manner"""
    fin_idx = np.isfinite(np.column_stack([regressand, effect])).all(axis=1)
    if not fin_idx.all():
        warnings.warn(
            "Encountered {} nans".format((~fin_idx).astype(int).sum())
        )
    preds_cv = skl_mdl_sel.cross_val_predict(
        model,
        X=effect[fin_idx],
        y=regressand[fin_idx],
        n_jobs=-1,
        cv=5,
    )
    resids = np.nan * np.ones_like(regressand)
    resids[fin_idx] = regressand[fin_idx] - preds_cv
    return resids


def logit_cv_auc(X: np.array, y: np.array, cv: int = 5):
    """AUC from the cross-validated logistic regression y~X"""
    idx = np.isfinite(np.column_stack([X, y])).all(axis=1)
    if (snan := sum((~idx).astype(int))) > 0:
        warnings.warn("Dropping {} nans".format(snan))
        X, y = X[idx], y[idx]
    preds_cv = skl_mdl_sel.cross_val_predict(
        skl_lm.LogisticRegressionCV(scoring="roc_auc"),
        X=X,
        y=y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    return skl_mets.roc_auc_score(y, preds_cv)


def stratified_logit_cv_metrics(
    X: np.array, y: np.array, return_perfs: bool = False
):
    pred_col = 0.0 * y
    batch_aucs = []
    for train_idx, test_idx in skl_mdl_sel.StratifiedKFold(
        n_splits=10, shuffle=True, random_state=42
    ).split(X, y):
        pred_col[test_idx] = (
            skl_lm.LogisticRegressionCV()
            .fit(X=X[train_idx], y=y[train_idx])
            .predict_proba(X[test_idx])[:, 1][:, np.newaxis]
        )
        batch_aucs.append(
            skl_mets.roc_auc_score(
                y_true=y[test_idx], y_score=pred_col[test_idx]
            )
        )
    perf = {
        "AUC": skl_mets.roc_auc_score(y_true=y, y_score=pred_col).round(4),
        "mean batch AUC": np.mean(batch_aucs).round(4),
        "std dev batch AUC": np.std(batch_aucs).round(4),
        "std err of the mean": sp_stats.sem(batch_aucs).round(4),
    }
    return batch_aucs if not return_perfs else (batch_aucs, perf)


# run tests if called as a script
if __name__ == "__main__":
    import statsmodels.api as sm

    n = 1000
    rng = np.random.default_rng(0)
    X = rng.normal(size=n)
    t = np.square(rng.normal(size=n))  # non-gaussian noise
    Y = X + t
    Y_less_t = regressed_out_effect_cv(Y.reshape(-1, 1), t.reshape(-1, 1))

    r2_before_regressing_out = (
        sm.regression.linear_model.OLS(endog=Y, exog=X).fit().rsquared
    )
    r2_after_regressing_out = (
        sm.regression.linear_model.OLS(endog=Y_less_t, exog=X).fit().rsquared
    )

    print(f"{r2_before_regressing_out=:.2f}")
    print(f"{r2_after_regressing_out=:.2f}")

    print(f"{logit_cv_auc(X.reshape(-1, 1), (Y > 0.5).astype(int))=:.2f}")
