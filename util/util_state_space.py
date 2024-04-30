#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for state-space modeling
"""

import datetime
import itertools
import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sp_opt
import scipy.stats as sp_stats
import sklearn.linear_model as skl_lm

plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["font.family"] = "serif"

pd.options.display.width = 79
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 79
pd.options.display.float_format = "{:,.3f}".format

home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def regress(
    X: np.array, Y: np.array, eps: float = 1e-6
) -> tuple[np.array, np.array]:
    """Finds the MLE estimates A_hat, S_hat for A, S where Y|X ~ N(X*A, S);
    protected against missing data

    Parameters
    ----------
    X: np.array
        n_data × X_dim array of inputs
    Y: np.array
        n_data × Y_dim array of outputs
    eps: float
        regularisation parameter

    Returns
    -------
    tuple containing
        X_dim × Y_dim A_hat coefficient matrix
        Y_dim × Y_dim covariance matrix S_hat

    """
    idx = np.isfinite(np.column_stack((X, Y))).all(axis=-1)
    X, Y = X[idx], Y[idx]
    A_hat = np.linalg.lstsq(
        X.T @ X + eps * np.eye(X.shape[1]), X.T @ Y, rcond=-1
    )[0]
    S_hat = np.cov(Y - X @ A_hat, rowvar=False)
    return A_hat, S_hat


def regress_alpha(
    X: np.array, Y: np.array, alpha: float
) -> tuple[np.array, np.array]:
    """Finds the MLE estimates A_hat,S_hat for A,S
    where output|input ~ N(input*A, S)

    Parameters
    ----------
    X: np.array
        n_data × in_dim array of inputs
    Y: np.array
        n_data × out_dim array of outputs
    alpha
        regularisation parameter for scikit-learn ridge regression

    Returns
    -------
    tuple of results
        in_dim × out_dim A_hat coefficient matrix
        out_dim × out_dim covariance matrix S_hat

    """
    idx = np.isfinite(np.column_stack((X, Y))).all(axis=-1)
    X, Y = X[idx], Y[idx]
    A_hat = (
        skl_lm.Ridge(alpha=alpha, fit_intercept=False, copy_X=True)
        .fit(X, Y)
        .coef_.T
    )
    S_hat = np.cov(Y - X @ A_hat, rowvar=False)
    return A_hat, S_hat


def nancat(arr1: np.array, arr2: np.array) -> np.array:
    """add nan's to array with shorter time length as needed in order to
    concatenate along dimension 1

    Parameters
    ----------
    arr1
        n_time1 × n_data1 × dim_data array
    arr2
        n_time2 × n_data2 × dim_data array
    Returns
    -------
    nancat_arr1_arr2
        max(n_time1,n_time2) × (n_data1 + n_data2) × dim_data array that
        was nancat'ed

    """
    assert arr1.shape[2:] == arr2.shape[2:]
    arr1_cat = (
        np.concatenate(
            [
                arr1,
                np.tile(
                    np.nan, (arr2.shape[0] - arr1.shape[0], *arr1.shape[1:])
                ),
            ]
        )
        if arr2.shape[0] > arr1.shape[0]
        else arr1
    )
    arr2_cat = (
        np.concatenate(
            [
                arr2,
                np.tile(
                    np.nan, (arr1.shape[0] - arr2.shape[0], *arr2.shape[1:])
                ),
            ]
        )
        if arr1.shape[0] > arr2.shape[0]
        else arr2
    )
    return np.concatenate([arr1_cat, arr2_cat], axis=1)


def standardize(
    arr: np.array,
    *,
    params: dict[str, np.array] = None,
    return_params: bool = False,
):
    """standardize array elements to [0.1, 1] along the 3rd axis

    Parameters
    ----------
    arr
        n_time × n_data × dim_data array
    params
        supply values for arr_mn and arr_mx used by this function
    return_params
        should we return parameters that can be used to apply this function
        to future data?

    Returns
    -------
    arr_standardized
        n_time × n_data × dim_data array standardized as described
    params
        dictionary of parameters for future use

    """
    if params is not None:
        arr_mn = params["arr_mn"]
        arr_mx = params["arr_mx"]
    else:
        arr_mn = np.nanmin(arr, axis=(0, 1), keepdims=True)
        arr_mx = np.nanmax(arr, axis=(0, 1), keepdims=True)
    arr_standardized = 0.9 * np.divide(arr - arr_mn, arr_mx - arr_mn) + 0.1
    if return_params:
        return arr_standardized, {"arr_mn": arr_mn, "arr_mx": arr_mx}
    else:
        return arr_standardized


def unstandardize(
    arr: np.array,
    params: dict[str, np.array],
) -> np.array:
    """inverse of standardize on array arr

    Parameters
    ----------
    arr
        n_time × n_data × dim_data array
    params
        dictionary of arr_mn and arr_mx used by standardize

    Returns
    -------
    arr_unstandardized
        the inverse of applying standardize to the array arr

    See Also
    --------
    standardize
        the inverse of this function

    """
    arr_unstandardized = (params["arr_mx"] - params["arr_mn"]) / 0.9 * (
        arr - 0.1
    ) + params["arr_mn"]
    assert np.allclose(standardize(arr_unstandardized, params=params), arr)
    return arr_unstandardized


def unstandardize_mean_and_cov(
    mean: np.array, cov: np.array, params: dict[str, np.array]
) -> tuple[np.array, np.array]:
    """calculate unstandardized mean and covariance of a Gaussian distribution
    with standardized mean and covariance

    Parameters
    ----------
    mean
        standardized mean
    cov
        standardized cov
    params
        parameters used to perform standardisation

    Returns
    -------
    mean_uns, cov_uns
        unstandardized mean and covariance

    """

    mean_uns = unstandardize(mean.reshape((1, 1, -1)), params=params).reshape(
        mean.shape
    )
    coeff = np.diag(((params["arr_mx"] - params["arr_mn"]) / 0.9).ravel())
    cov_uns = coeff @ cov @ coeff.T

    return mean_uns, cov_uns


def unstandardize_mean_and_cov_diffs(
    mean_diff: np.array, cov_diff: np.array, params: dict[str, np.array]
) -> tuple[np.array, np.array]:
    """calculate unstandardized mean and covariance of a Gaussian distribution
    with standardized mean and covariance

    Parameters
    ----------
    mean_diff
        standardized mean of difference
    cov_diff
        standardized cov of difference
    params
        parameters used to perform standardisation

    Returns
    -------
    mean_diff_uns, cov_diff_uns
        unstandardized mean_diff and covariance_diff

    """
    coeff = np.diag(((params["arr_mx"] - params["arr_mn"]) / 0.9).ravel())
    mean_diff_uns = coeff @ mean_diff
    cov_diff_uns = coeff @ cov_diff @ coeff.T

    return mean_diff_uns, cov_diff_uns


def normalize(
    arr: np.array,
    eps: float = np.finfo(float).eps,
    *,
    params: dict[str, np.array] = None,
    return_params: bool = False,
):
    """normalize array elements to have mean 0 & stddev ~1 along the 3rd axis
    in a safe manner

    Parameters
    ----------
    arr
        n_time1 × n_data1 × dim_data array
    eps
        safety parameter for dimensions having no variance
    params
        supply mean and std to be used by this function
    return_params
        should we return parameters that can be used to apply this function
        to future data?

    Returns
    -------
    arr_normalized
        n_time1 × n_data1 × dim_data array normalized as described

    """
    if params is not None:
        arr_mean = params["arr_mean"]
        arr_std = params["arr_std"]
    else:
        arr_mean = np.nanmean(arr, axis=(0, 1), keepdims=True)
        arr_std = np.nanstd(arr, axis=(0, 1), keepdims=True) + eps
    arr -= arr_mean
    arr /= arr_std
    if return_params:
        return arr, {"arr_mean": arr_mean, "arr_std": arr_std}
    else:
        return arr


def unnormalize(
    arr: np.array,
    params: dict[str, np.array],
) -> np.array:
    """inverse of normalize on array arr

    Parameters
    ----------
    arr
        n_time × n_data × dim_data array
    params
        dictionary of arr_mn and arr_mx used by normalize

    Returns
    -------
    arr_unnormalized
        the inverse of applying normalize to the array arr

    See Also
    --------
    normalize
        the inverse of this function

    """
    arr_unnormalized = params["arr_std"] * arr + params["arr_mean"]
    assert np.allclose(normalize(arr_unnormalized, params=params), arr)
    return arr_unnormalized


def take_finite_along_axis(arr: np.array, axis: int = 0) -> np.array:
    """take the n-1 dimensional slices along axis `axis` only for slices
    where every element is finite

    Parameters
    ----------
    arr
        array on which to operate
    axis
        axis on which to operate; axis < arr.ndim

    Example
    -------
    ```
    eg_arr = np.concatenate(
        [np.arange(7), np.repeat(np.nan, 3), np.arange(2)]
    ).reshape(6, 2)
    np.array_equal(
        take_finite_along_axis(eg_arr, axis=0),
        array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.0, 1.0]]),
    )  # True
    ```

    Returns
    -------
    arr_finite_along_axis
        the result

    """
    new_shape = list(arr.shape)
    new_shape[axis] = -1
    return np.take(
        arr,
        np.argwhere(
            np.isfinite(arr).all(
                axis=tuple(a for a in range(arr.ndim) if a != axis)
            )
        ),
        axis,
    ).reshape(tuple(new_shape))


def mask_all_but_time_i(arr: np.array, i: int) -> np.array:
    """takes a standardized n_time × n_data × dim_data array and returns an
    n_time × n_data × dim_data array with only data from time i and all else
    set to nan

    Parameters
    ----------
    arr
        (n_time × n_data × dim_data) input array
    i
        index 0<=i<n_time

    Returns
    -------
    arr_m
        a masked version of arr

    """

    arr_m = np.nan * np.ones_like(arr)
    arr_m[i] = arr[i].copy()
    return arr_m


def mask_all_but_time_i_vect(arr: np.array, i: np.array) -> np.array:
    """vectorized form of mask_all_but_time_i"""
    assert arr.shape[1] == len(i)
    arr_m = np.nan * np.ones_like(arr)
    for i_data, i_time in enumerate(i):
        arr_m[i_time, i_data] = arr[i_time, i_data].copy()
    return arr_m


def parcellate_arrays(*args) -> np.array:
    """parcellates all inputted arrays

    Parameters
    ----------
    args
        3-d arrays in the typical format

    Returns
    -------
    tuple of arrays parcellated with parcellate_array_to_snapshots

    See Also
    --------
    parcellate_array_to_snapshots

    """

    return (
        np.concatenate(
            [mask_all_but_time_i(arr, i) for i in range(arr.shape[0])], axis=1
        )
        for arr in args
    )


def weighted_mean_and_covariance(values: np.array, weights: np.array):
    """calculated the weighted mean and covariance of `values` with weights
    `weights`

    Parameters
    ----------
    values
        n_time × n_data × dim_data array of values
    weights
        n_data array of weights

    Returns
    -------
    mean
        n_time × dim_data weighted mean
    cov
        n_time × dim_data × dim_data weighted covariance

    """
    assert (weights >= 0).all()

    # weighted mean
    m_c_num = np.einsum(
        "ijk,j->ik", np.nan_to_num(values), weights, optimize=True
    )
    m_c_denom = np.einsum(
        "ijk,j->ik", np.isfinite(values), weights, optimize=True
    )
    m_c = m_c_num / m_c_denom

    # weighted covariance
    v_centered = values - np.expand_dims(m_c, axis=1)
    v_c_num = np.einsum(
        "ijk,j,ijl->ikl",
        np.nan_to_num(v_centered),
        weights,
        np.nan_to_num(v_centered),
        optimize=True,
    )
    v_c_denom = np.einsum(
        "ijk,j,ijl->ikl",
        np.isfinite(v_centered),
        weights,
        np.isfinite(v_centered),
        optimize=True,
    )
    v_c = v_c_num / v_c_denom

    return m_c, v_c


def plot_metric_vs_clusters_over_time(
    metric: np.array,
    assignments: np.array,
    metric_name: str,
    *,
    savename: str | os.PathLike,
    title: str,
    xticks: np.array = None,
    xlabel: str = "Time steps",
    xlim: tuple = None,
    ylim: tuple = None,
    legend_loc: str = None,
    legend_bbox_to_anchor=(1.5, 1),
    colors: tuple = (
        "#0072CE",
        "#E87722",
        "#64A70B",
        "#93328E",
        "#A81538",
        "#4E5B31",
    ),
    show: bool = False,
) -> None:
    """plot the mean likelihood +/- 1sem
    of the metric for each cluster (as assigned)
    vs. time
    """
    n_timesteps, n_data = metric.shape
    assert n_data == len(assignments)
    n_clusters = len(set(assignments))

    fig, ax = plt.subplots()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for i, c in enumerate(string.ascii_uppercase[:n_clusters]):
        v_c = metric[:, c == assignments]
        c_mean = np.nanmean(
            v_c,
            axis=1,
        )
        c_sem = sp_stats.sem(v_c, axis=1, nan_policy="omit")

        plt.errorbar(
            x=xticks
            if xticks is not None
            else np.arange(n_timesteps) + 0.025 * (i - int(n_clusters / 2)),
            y=c_mean.T,
            yerr=c_sem.T,
            color=colors[i],
            linestyle=(
                "solid",
                "dotted",
                "dashed",
                "dashdot",
                "densely dashdotted",
                "loosely dashdotted",
            )[i],
            label=f"cluster {c}",
            capsize=5,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = dict(zip(labels, handles))
    ax.legend(
        unique_labels_dict.values(),
        unique_labels_dict.keys(),
        fontsize="large",
        loc=legend_loc
        if legend_loc is not None
        else plt.rcParams["legend.loc"],
        bbox_to_anchor=legend_bbox_to_anchor,
    )
    plt.xticks(
        ticks=xticks if xticks is not None else range(n_timesteps),
        labels=xticks if xticks is not None else range(1, n_timesteps + 1),
    )
    if len(title) > 0:
        plt.title(title, fontsize="large")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    ax.set_xlabel(xlabel, fontsize="large")
    ax.set_ylabel(metric_name, fontsize="large")
    plt.savefig(savename, bbox_inches="tight", transparent=True)
    if show:
        plt.show(bbox_inches="tight")


def histograms_by_cluster(
    *,
    savename: str | os.PathLike = "",
    title: str = "Histograms by cluster",
    metrics: np.array = None,
    metric_names: list = None,
    clusters: np.array = None,
    cluster_ordering: np.array = None,
    show: bool = False,
    nrows: int = 2,
    ncols: int = 3,
    nbins: int = 20,
    density: bool = True,
    verbose: bool = False,
    mean_overlay: bool = True,
    normal_overlay: bool = False,
    μσ_overlay=None,
    tighten=True,
    colors: tuple = (
        "#0072CE",
        "#E87722",
        "#64A70B",
        "#93328E",
        "#A81538",
        "#4E5B31",
    ),
    alpha: float = 0.5,
) -> None:
    """creates subplots of overlapping histograms by cluster assignment

    Parameters
    ----------
    savename
        filename for saved plot
    title
        suptitle (overall) title
    metrics
        n_data x n_metric_names array
    metric_names
        list of metric names
    clusters
        n_data-length vector of cluster assignments
    cluster_ordering
        in what order should the clusters be plotted?
    show
        should we show the result when we're done?
    nrows
        number of rows of sub-plots
    ncols
        number of columns of sub-plots
    nbins
        number of bins for individual histograms
    density
        boolean: return density-normalized histograms?
    verbose
        should we be verbose?
    mean_overlay
        should we have vertical lines for the means?
    normal_overlay
        should we superimpose gaussian densities over the plots?
    μσ_overlay
        optionally provide a dictionary {cluster letter: {μ: _}, {σ: _}} where
        the _ are vectors of length len(metric_names) describing the means and
        standard deviations of gaussian overlays -- allows us to illustrate
        model predictions with real data
    tighten
        should we apply tight_layout() after creating the plot?
    colors
        tuple for coloring the clusters
    alpha
        standard translucency parameter

    Returns
    -------
    plot with more plots

    """

    c_labels = cluster_ordering or sorted(list(np.unique(clusters)))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout="constrained")
    axs = np.atleast_2d(axs).reshape(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            m_num = ncols * i + j
            if m_num >= len(metric_names):
                fig.delaxes(axs[i, j])
                continue
            for k, c in enumerate(c_labels):
                if np.sum(clusters == c) == 0:
                    continue
                # mn = np.nanmin(metrics[clusters == c])
                # mx = np.nanmax(metrics[clusters == c])
                # wd = mx - mn
                # axs[i, j].set_xlim(left=mn - 0.05 * wd, right=mx + 0.05 * wd)
                axs[i, j].hist(
                    x=metrics[clusters == c, m_num],
                    label=f"cluster {c}"
                    if not verbose
                    else "cluster {c} (μ={μ},σ={σ})".format(
                        c=c,
                        μ=np.nanmean(metrics[clusters == c, m_num]).round(2),
                        σ=np.nanstd(metrics[clusters == c, m_num]).round(2),
                    ),
                    bins=nbins,
                    color=colors[k],
                    alpha=alpha,
                    density=density,
                    # rwidth=1,
                    # range=(mn, mx),
                )
                if mean_overlay:
                    axs[i, j].axvline(
                        np.nanmean(metrics[clusters == c, m_num]),
                        color=colors[k],
                    )
                if normal_overlay:
                    μ = np.nanmean(metrics[clusters == c, m_num])
                    σ = np.nanstd(metrics[clusters == c, m_num])
                    mn, mx = axs[i, j].get_xlim()
                    pts = np.linspace(mn, mx, 1000)
                    axs[i, j].plot(
                        pts,
                        sp_stats.norm.pdf(pts, loc=μ, scale=σ),
                        color=colors[k],
                    )
                if μσ_overlay is not None:
                    mn, mx = axs[i, j].get_xlim()
                    pts = np.linspace(mn, mx, 1000)
                    axs[i, j].plot(
                        pts,
                        sp_stats.norm.pdf(
                            pts,
                            loc=μσ_overlay[c]["μ"][m_num],
                            scale=μσ_overlay[c]["σ"][m_num],
                        ),
                        color=colors[k],
                    )

                axs[i, j].set_title(metric_names[m_num], fontsize="large")
                axs[i, j].spines["right"].set_visible(False)
                axs[i, j].spines["top"].set_visible(False)
                if verbose:
                    axs[i, j].legend(fontsize="large")

    if len(c_labels) > 1 and not verbose:
        handles, labels = axs[0, 0].get_legend_handles_labels()
        unique_labels_dict = dict(zip(labels, handles))
        fig.legend(
            unique_labels_dict.values(),
            unique_labels_dict.keys(),
            fontsize="large",
            loc="upper right",
            bbox_to_anchor=(1.3, 1.0),
        )
    # fig.set_size_inches(18.5, 10.5, forward=True)
    if tighten:
        plt.tight_layout()
    if len(title) > 0:
        fig.suptitle(title, size=30)
    if len(savename) > 0:
        fig.savefig(savename, bbox_inches="tight", transparent=True)
    if show:
        plt.show(bbox_inches="tight")


def histogram(
    metrics: np.array = None,
    *,
    savename: str | os.PathLike = "",
    show: bool = False,
    title: str = None,
    density: bool = True,
    nbins: int = 25,
    figsize: tuple = (6.4, 4.8),
) -> None:
    """plot a simple histogram

    Parameters
    ----------
    metrics
        vector of distribution for the creation of the histogram
    savename
        filename in which to save the result
    show
        should we show the plot after we've created it?
    title
        for the plot?
    density
        matplotlib option for histogram creation
    nbins
        number of bins to use
    figsize
        size of figure to create

    """

    fig, axs = plt.subplots(layout="constrained", figsize=figsize)
    axs.hist(
        x=metrics.ravel(),
        bins=nbins,
        color="#0072CE",
        alpha=1.0,
        density=density,
    )
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    if title is not None:
        plt.title(title)
    if len(savename) > 0:
        fig.savefig(savename, transparent=True)
    if show:
        fig.show()


def pies_by_cluster(
    *,
    savename: str | os.PathLike = "",
    title: str = "",
    categories: np.array = None,
    category_ordering: np.array = None,
    category_legend_names: dict = None,
    clusters: np.array = None,
    cluster_ordering: np.array = None,
    show: bool = False,
    nrows: int = None,
    ncols: int = None,
    slice_colors: tuple[str] = None,
    legend_bbox_to_anchor=(0.0, 0.0),
    fig_length: float = None,
    fig_width: float = None,
    halo_colors: tuple[str] = None,
) -> None:
    """creates subplots of pie charts

    Parameters
    ----------
    savename
        filename for saved plot
    title
        suptitle (overall) title
    categories
        n_data-length vector of category assignments
    category_ordering
        in what order should the categories be plotted?
    category_legend_names
        dictionary with categories as keys and what they should be called
        in the legend as values; e.g. {'CN': Cognitively normal}
    clusters
        n_data-length vector of cluster assignments
    cluster_ordering
        in what order should the clusters be plotted?
    show
        should we show the result when we're done?
    nrows
        number of rows of sub-plots
    ncols
        number of columns of sub-plots
    cmap
        color map to use
    slice_colors
        colors for the slices of the pie
    legend_bbox_to_anchor
        change the positioning of the legend
    fig_length
        figure length
    fig_width
        figure width
    halo_colors
        colors for the halos

    Returns
    -------
    plot with more plots

    """

    cluster_labels = cluster_ordering or sorted(list(np.unique(clusters)))
    category_labels = category_ordering or sorted(list(np.unique(categories)))
    nrows = nrows or len(cluster_labels)
    ncols = ncols or 1

    if slice_colors is None:
        slice_colors = plt.colormaps["cividis"].colors
        slice_colors = np.flipud(
            np.array(slice_colors)[
                np.linspace(
                    0, len(slice_colors) - 1, len(category_labels)
                ).astype(int)
            ]
        ).tolist()

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout="constrained")
    axs = axs.reshape(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            m_num = ncols * i + j
            if m_num >= len(cluster_labels):
                break
            # prev_hatch_color = plt.rcParams['hatch.color']
            # plt.rcParams['hatch.color'] = 'white'
            patches, _ = axs[i, j].pie(
                [
                    np.logical_and(
                        categories == cat,
                        clusters == cluster_labels[m_num],
                    ).sum()
                    for cat in category_labels
                ],
                colors=slice_colors,
                # hatch=['//', '++', 'oo', '**', '..', 'xx']
            )
            # plt.rcParams['hatch.color'] = prev_hatch_color
            if halo_colors is not None:
                axs[i, j].add_patch(
                    plt.Circle(
                        (0, 0),
                        1.0,
                        color=halo_colors[m_num],
                        linewidth=1.5,
                        fill=False,
                    )
                )
            axs[i, j].set_title(cluster_labels[m_num])

    axs[0, 0].legend(
        patches,
        category_labels
        if category_legend_names is None
        else list(map(category_legend_names.__getitem__, category_labels)),
        loc="upper right",
        bbox_to_anchor=legend_bbox_to_anchor,
    )
    if fig_length is None or fig_width is None:
        fig.set_size_inches(ncols, nrows + 1, forward=True)
    else:
        fig.set_size_inches(fig_width, fig_length, forward=True)
    if len(title) > 0:
        fig.suptitle(title, size=30)
    if len(savename) > 0:
        fig.savefig(savename, bbox_inches="tight", transparent=True)
    if show:
        plt.show(bbox_inches="tight")


def pie(
    assignments: np.array,
    *,
    savename: str | os.PathLike = None,
    title: str = "",
    cluster_ordering: np.array = None,
    show: bool = False,
    legend_bbox_to_anchor=(1.2, 1.0),
    colors: tuple[str] = (
        "#0072CE",
        "#E87722",
        "#64A70B",
        "#93328E",
        "#A81538",
        "#4E5B31",
    ),
):
    """creates a pie chart

    Parameters
    ----------
    assignments
        cluster assignments
    savename
        name / location to save result
    title
        title for plot
    cluster_ordering
        an array of the unique elements of assignments
    show
        should the plot be shown after its creation?
    legend_bbox_to_anchor
        where to place the legend
    colors
        colors to use

    """
    cluster_labels = cluster_ordering or sorted(list(np.unique(assignments)))
    assert set(cluster_labels) == set(assignments)
    fig, axs = plt.subplots(layout="constrained")
    patches, _ = axs.pie(
        [(assignments == lbl).sum() for lbl in cluster_labels],
        colors=list(colors),
        explode=[0.03] * len(cluster_labels),
    )
    axs.legend(
        patches,
        cluster_labels,
        loc="upper right",
        bbox_to_anchor=legend_bbox_to_anchor,
    )
    if title is not None:
        axs.set_title(title, fontsize="large")
    if savename is not None:
        fig.savefig(savename, bbox_inches="tight", transparent=True)
    if show:
        plt.show(bbox_inches="tight")


def summarize_metric_vs_cluster(
    metric: np.array,
    cluster_assignment: np.array,
    names: list = None,
    cluster_ordering: np.array = None,
) -> None:
    """creates aggregate summary statistics grouped by cluster

    Parameters
    ----------
    metric
        n_data × n_names array of data to aggregate
    cluster_assignment
        n_data-length vector of assignments
    names
        n_names list of variable names
    cluster_ordering
        (optional) supply an ordered list of clusters

    Returns
    -------
    means of metric by cluster and p-values from 2-sided t-tests

    """
    n_clusters = len(set(cluster_assignment))
    ordered_clusters = cluster_ordering or string.ascii_uppercase[:n_clusters]

    assert metric.shape[0] == len(cluster_assignment)
    if names is not None:
        assert metric.shape[1] == len(names)

    m_x_c = {a: metric[cluster_assignment == a] for a in ordered_clusters}

    print(
        pd.DataFrame.from_records(
            [(a, *np.nanmean(ts, axis=0)) for a, ts in m_x_c.items()],
            columns=["cluster", *[name + "_mean" for name in names]],
        ).set_index("cluster")
    )

    print(
        pd.DataFrame.from_records(
            [
                (
                    f"{a}_vs_{b}",
                    *sp_stats.ttest_ind(
                        m_x_c[a],
                        m_x_c[b],
                        nan_policy="omit",
                        alternative="two-sided",
                    ).pvalue,
                )
                for a, b in itertools.combinations(m_x_c.keys(), 2)
            ],
            columns=[
                "comparison",
                *[name + "_pval" for name in names],
            ],
        ).set_index("comparison")
    )


def get_finite_length(arr: np.array) -> np.array:
    """takes a standardized n_time × n_data × dim_data array and returns an
    n_data array containing the length of the longest trajectory with all
    full data starting from index 0 / initial time

    Parameters
    ----------
    arr
        (n_time × n_data × dim_data) input array

    Returns
    -------
    lengths
        n_data output array containing lengths

    Example
    -------
    ```
    arr = np.arange(24).reshape((2, 3, 4)).astype(float)
    arr[0,1] = arr[1,2] = np.nan
    assert all(get_finite_length(arr) == np.array([2, 0, 1]))
    ```
    """

    arr_fin = np.all(np.isfinite(arr), axis=-1)
    return np.where(
        np.all(arr_fin, axis=0), arr_fin.shape[0], np.argmin(arr_fin, axis=0)
    )


def take_final_finite(arr: np.array) -> np.array:
    """takes a standardized n_time × n_data × dim_data array and returns an
    n_data × dim_data array with the last full set of finite measurements for
    each person in the array `arr`

    Parameters
    ----------
    arr
        (n_time × n_data × dim_data) input array

    Returns
    -------
    arr_ff
        (n_data × dim_data) output array containing final finite data

    Example
    -------
    ```
    arr = np.arange(24).reshape(2, 3, 4).astype(float)
    arr[1,1:2] = np.nan
    assert all(take_final_finite(arr) == np.array([2, 0, 1]))
    ```
    """

    final_idx = get_finite_length(arr) - 1
    assert np.all(final_idx >= 0)
    return np.stack([arr[i, j] for i, j in zip(final_idx, range(len(arr[0])))])


def add_constant_where_finite(arr: np.array) -> np.array:
    """takes a standardized n_time × n_data × dim_data array and returns an
    n_time × n_data × (dim_data+1) array with a constant appended

    Parameters
    ----------
    arr
        (n_time × n_data × dim_data) input array

    Returns
    -------
    arr1
        (n_time × n_data × (dim_data+1)) array with constant appended

    """

    return np.concatenate(
        [
            arr,
            np.where(np.isfinite(arr).any(axis=-1)[..., None], 1.0, np.nan),
        ],
        axis=-1,
    )


def today_str() -> str:
    """ISO 8601 date string"""
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def make_str_nice(s: str) -> str:
    """
    make the string nice
    'Hello Wor#rld' -> 'hello_wor_rld'
    '.fooBar' -> 'foobar'
    """
    s = re.sub("[^0-9a-zA-Z_]", "_", s.lower())
    s = re.sub("_+", "_", s).strip(" _")
    return s


def format_names(n_list: list[str], elide_at: int = 42) -> list[str]:
    """take a list of names and return a list of nice names"""
    return [n.replace("_", " ")[:elide_at] for n in n_list]


def plot_weighted_means_2d_trajectories(
    weights: np.array,
    values: np.array,
    colors: list,
    saveloc: str | os.PathLike,
    *,
    xlabel: str = "β-amyloid",
    ylabel: str = "Gray matter density",
    xlim: tuple[float, float] = (-50.0, 230.0),
    ylim: tuple[float, float] = (-0.275, 0.025),
    arrow_width: float = 0.6,
    soft_assignment: bool = True,
    cov_alpha: float = 0.05,
    conf_thresh: float = 0.68,
    elide_at: list = None,
) -> None:
    """

    Parameters
    ----------
    weights
        (n_data × n_clusters) array of weights for calculating the
        weighted mean and covariance
    values
        (n_time × n_data × 2) array of trajectories for averaging over
    colors
        list of colors to use
    saveloc
        where to save plot
    xlabel
        name for first value
    ylabel
        name for second value
    xlim
        2-tuple of limits for first value
    ylim
        2-tuple of limits for second value
    arrow_width
        float for controlling the size of arrows
    soft_assignment
        should soft assignment be used for averaging?
    cov_levels
        levels for covariance contours; if None, not printed
    cov_alpha
        alpha level for covariance contours
    conf_thresh
        probabilistic cutoff in (0,1) for confidence region
    elide_at
        list of length number of clusters, specifies the 0-based index to stop
        printing;

    """
    # plot out hidden trajectories colored by cluster
    # for training data
    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    n_clusters = weights.shape[1]

    for i, c in enumerate(string.ascii_uppercase[:n_clusters]):
        prob_c = (
            weights[:, i]
            if soft_assignment
            else (weights.argmax(axis=1) == i).astype(int)
        )
        m_c, v_c = weighted_mean_and_covariance(values, prob_c)

        if elide_at is not None and elide_at[i] is not None:
            m_c, v_c = m_c[: elide_at[i]], v_c[: elide_at[i]]

        plt.quiver(
            m_c[:-1, 0].ravel(),
            m_c[:-1, 1].ravel(),
            np.diff(m_c[:, 0]).ravel(),
            np.diff(m_c[:, 1]).ravel(),
            color=colors[i],
            linestyle="solid",
            units="xy",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=arrow_width,
            headwidth=2,
            headlength=2,
            headaxislength=1,
            # zorder=-i,
            alpha=1.0,
        )

        plt.scatter(
            m_c[:, 0].ravel(),
            m_c[:, 1].ravel(),
            c=colors[i],
            marker=("o", "v", "^", "s", "+", "x")[i],
            label=f"cluster {c}",
            s=4,
        )
        pos = np.dstack(
            np.meshgrid(
                np.linspace(*ax.get_xlim(), num=1000),
                np.linspace(*ax.get_ylim(), num=1000),
            )
        )

        for t in range(
            values.shape[0] if not elide_at or not elide_at[i] else elide_at[i]
        ):
            zval = sp_stats.multivariate_normal(mean=m_c[t], cov=v_c[t]).pdf(
                pos
            )
            res = sp_opt.minimize(
                lambda thr: np.square(
                    zval[zval >= thr].sum() / zval.sum() - conf_thresh
                ),
                0.9 * zval.max(),
                method="Nelder-Mead",
                tol=1e-6,
            )
            ax.contourf(
                pos[..., 0],
                pos[..., 1],
                zval,
                colors=(
                    "#0072CE",
                    "#E87722",
                    "#64A70B",
                    "#93328E",
                    "#A81538",
                    "#4E5B31",
                )[i],
                linewidths=1,
                levels=[float(res.x), np.inf],
                alpha=cov_alpha,
            )

    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = dict(zip(labels, handles))
    ax.legend(
        unique_labels_dict.values(),
        unique_labels_dict.keys(),
        fontsize="large",
        bbox_to_anchor=(1.3, 1),
        markerscale=3,
    )
    ax.set_xlabel(xlabel, fontsize="large")
    ax.set_ylabel(ylabel, fontsize="large")
    plt.tight_layout()
    fig.savefig(
        saveloc,
        bbox_inches="tight",
        transparent=True,
    )
