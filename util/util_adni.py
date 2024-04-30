#!/usr/bin/env python3

"""
Utilities for loading and plotting the ADNI dataset
"""

import functools
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from framework import marginalizable_mixture_model as mixmodel

pd.options.display.float_format = "{:,.3f}".format

plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["font.family"] = "serif"

name = "ADNI"

hidden_columns = [
    "amyloid_centiloid",
    "gm_score_standardized",
]

observed_columns = [
    "ADNI_MEM",
    "ADNI_EF",
    "ADAS13",
    "MOCA",
]

diagnosis_column = ["diagnosis"]
diagnosis_list = ["CN", "sMCI", "pMCI", "AD"]

cluster_colors = (
    "#0072CE",
    "#E87722",
    "#64A70B",
    "#93328E",
    "#A81538",
    "#4E5B31",
)

diagnosis_colors = tuple(
    np.array(plt.colormaps["cividis"].reversed().colors)[
        np.linspace(
            0,
            plt.colormaps["cividis"].N - 1,
            len(diagnosis_list),
        ).astype(int)
    ]
)


def get_final_diagnoses(diagnoses: np.array) -> np.array:
    """extracts final available diagnoses from diagnostic trajectories

    Parameters
    ----------
    diagnoses
        array of diagnoses as supplied by get_data()

    Returns
    -------
        an n_data-length array of final diagnoses for each person

    See Also
    -------
    get_data
        used to generate the array `diagnoses`
    """
    diagnosis_paths = [
        "->".join(l).replace("->nan", "")
        for l in diagnoses.squeeze().transpose().astype(str).tolist()
    ]
    return np.array([s.split("->")[-1] for s in diagnosis_paths])


def plot_2d_trajectories(
    model,
    savename: str | os.PathLike,
    *,
    title: str = "Latent trajectories by cluster (training)",
    states: np.array = None,
    inferred_clusters: np.array = None,
    intensities: np.array = None,
    std_param: dict[str, np.array] = None,
    drop_superimposed_model: bool = True,
    cluster_ordering: np.array = None,
    xlabel: str = "β-amyloid",
    ylabel: str = "Gray matter density",
    xlim: tuple[float, float] = (-50.0, 230.0),
    ylim: tuple[float, float] = (-0.275, 0.025),
    arrow_width: float = 0.4,
    show: bool = False,
) -> None:
    """plots 2d trajectories and colors by inferred cluster membership

    Parameters
    ----------
    model
        mixture model instance
    savename: str
        filename for output figure
    title: str
        title for figure
    states
        (optional) states to plot
        defaults to model's training data
    inferred_clusters
        (optional) inferred cluster membership for coloring the trajectories
        defaults to inferred clusters for training data
    std_param
        (optional) plot initial state densities for clusters as a sanity check
    cluster_ordering
        orders clusters according to provided array
    xlabel
        label for x-axis
    ylabel
        label for y-axis
    xlim
        limits for x-axis
    ylim
        limits for y-axis
    arrow_width
        width of arrows for trajectories
    show
        open the resulting figure?

    Returns
    -------
    plots of trajectories colored by inferred cluster membership
    """
    if states is None:
        states = model.states
    if inferred_clusters is None:
        inferred_clusters = np.array(
            [model.correspondence[c] for c in model.cluster_assignment]
        )
    n_clusters = (
        len(set(inferred_clusters).intersection(set(string.ascii_letters)))
        if model is None
        else model.n_clusters
    )
    c_labels = cluster_ordering or string.ascii_uppercase[:n_clusters]

    assert states.ndim == 3 and states.shape[-1] == 2
    assert states.shape[1] == len(inferred_clusters)

    # plot out hidden trajectories colored by cluster
    # for training data
    fig, ax = plt.subplots()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for i, c in enumerate(c_labels):
        if np.sum(inferred_clusters == c) == 0:
            continue
        xcinit = states[:-1, inferred_clusters == c, 0].ravel()
        ycinit = states[:-1, inferred_clusters == c, 1].ravel()
        if intensities is not None:
            intc0 = np.tile(
                intensities[inferred_clusters == c], (1, states.shape[0] - 1)
            ).ravel()
            intc = np.tile(
                intensities[inferred_clusters == c], (1, states.shape[0])
            ).ravel()
        xcdir = np.diff(states[:, inferred_clusters == c, 0], axis=0).ravel()
        ycdir = np.diff(states[:, inferred_clusters == c, 1], axis=0).ravel()
        assert len(xcinit) == len(ycinit) == len(xcdir) == len(ycdir)
        cids = functools.reduce(
            np.logical_and,
            map(np.isfinite, [xcinit, ycinit, xcdir, ycdir]),
        )
        plt.quiver(
            xcinit[cids],
            ycinit[cids],
            xcdir[cids],
            ycdir[cids],
            color=[
                cluster_colors[i]
                + hex(round(255 * (2 * intens + 1) / 3))[2:].upper().zfill(2)
                for intens in intc0[cids]
            ]
            if intensities is not None
            else cluster_colors[i],
            linestyle="solid",
            units="xy",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=arrow_width,
            headwidth=5,
            headlength=7,
            headaxislength=6,
            zorder=-i,
            alpha=0.5,
        )
        plt.scatter(
            states[:, inferred_clusters == c, 0].ravel(),
            states[:, inferred_clusters == c, 1].ravel(),
            c=[
                cluster_colors[i]
                + hex(round(255 * (2 * intens + 1) / 3))[2:].upper().zfill(2)
                for intens in intc
            ]
            if intensities is not None
            else cluster_colors[i],
            marker=("o", "v", "^", "s", "+", "x")[i],
            label=f"cluster {c}" if cluster_ordering is None else c,
            s=2,
            alpha=0.6667,
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
    if len(title) > 0:
        plt.title(title)
    ax.set_xlabel(xlabel, fontsize="large")
    ax.set_ylabel(ylabel, fontsize="large")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    plt.tight_layout()
    if std_param is not None and not drop_superimposed_model:
        model.superimpose_model_on_plot(ax, std_param=std_param)
    fig.savefig(savename, bbox_inches="tight", transparent=True)
    if show:
        plt.show(bbox_inches="tight")


def get_outcomes_by_cluster(
    model: mixmodel.MMLinGaussSS_marginalizable,
    cluster_assignments: np.array,
    final_diagnoses: np.array,
) -> None:
    """print a table of outcomes vs cluster

    Parameters
    ----------
    model
        trained model on ADNI data
    cluster_assignments
        n_data array of cluster assignments
    final_diagnoses
        n_data array of diagnoses

    Returns
    -------
    prints out breakdowns of the outcomes for each cluster in a format that
    can be copied easily into Excel

    See Also
    -------
    report_model_outcomes
        pretty-printed results that are less amenable to copy-and-paste
    """

    assert final_diagnoses.size == cluster_assignments.size

    cluster_records = []
    for c in string.ascii_uppercase[: model.n_clusters]:
        cluster_pct = np.mean(
            cluster_assignments == model.inverse_correspondence[c]
        )
        outcome_pct = [
            np.mean(
                final_diagnoses[
                    cluster_assignments == model.inverse_correspondence[c]
                ]
                == d
            )
            for d in diagnosis_list
        ]
        cluster_records.append((c, cluster_pct, *outcome_pct))

    tbl = pd.DataFrame.from_records(
        cluster_records,
        columns=pd.MultiIndex.from_tuples(
            [
                ("", "cluster"),
                ("overall", "prevalence"),
                *[("within-cluster", d) for d in diagnosis_list],
            ]
        ),
    ).fillna("---")

    print(tbl)


def generate_outcome_table(
    model: mixmodel.MMLinGaussSS_marginalizable,
    diagnoses: np.array,
) -> None:
    """print a table of outcomes on trained model for evaluative purposes

    Parameters
    ----------
    model
        trained model on ADNI data
    diagnoses
        n_timesteps × n_data array of diagnoses

    Returns
    -------
    prints out breakdowns of the outcomes for each cluster in a format that
    can be copied easily into Excel

    See Also
    -------
    report_model_outcomes
        pretty-printed results that are less amenable to copy-and-paste
    """

    final_diagnoses = get_final_diagnoses(diagnoses)
    get_outcomes_by_cluster(model, model.cluster_assignment, final_diagnoses)


def set_model_correspondence(
    mdl: mixmodel.MMLinGaussSS_marginalizable, diagnoses: np.array
) -> None:
    """takes a trained model and sets the cluster correspondence according
    to final diagnostic outcome -- note that clusters are only determined up to
    relabelling; i.e. the labels themselves don't matter

    Parameters
    ----------
    mdl
        trained model on ADNI data
    diagnoses
        n_timesteps × n_data array of diagnoses

    """

    mdl.correspondence = dict(
        zip(
            np.argsort(
                [
                    np.mean(
                        get_final_diagnoses(diagnoses)[
                            mdl.cluster_assignment == c
                        ]
                        == "AD"
                    )
                    for c in range(mdl.n_clusters)
                ]
            ),
            string.ascii_uppercase,
        )
    )


def get_trajectories(
    return_mmse: bool = False,
    return_approx_age: bool = False,
) -> tuple:
    """does a standard pull of trajectories w/ lengths 2,3,4 and returns a
    tuple of the standard data

    Returns
    -------
        tuple of hidden states, observed states, diagnoses, ids, times,
        mmse if return_mmse, tau_cols if return_tau_columns, approx_age if
        return_approx_age
    """

    z, x, d, ids, time, mmse, age = map(
        np.load(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "adni-trajectories.npz",
            ),
            allow_pickle=True,
        ).__getitem__,
        ["z", "x", "d", "ids", "time", "mmse", "age"],
    )

    match return_mmse, return_approx_age:
        case True, True:
            return z, x, d, ids, time, mmse, age
        case True, False:
            return z, x, d, ids, time, mmse
        case _:
            return z, x, d, ids, time


def return_profiling_dataframe(ids):
    return pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "adni-profiling.csv",
        ),
        index_col="RID",
    ).loc[ids.ravel()]


# print off some summary stats if called as a script
if __name__ == "__main__":
    print(f"Generating {name} summary stats...")

    (
        z,
        x,
        d,
        ids,
        time,
        mmse,
    ) = get_trajectories(
        return_mmse=True,
    )
    dz, dx = z.shape[-1], x.shape[-1]
    length = np.argmin(
        np.all(np.isfinite(np.concatenate([x, z], axis=-1)), axis=-1), axis=0
    )
    length[length == 0] = x.shape[0]
    print("lengths: ", dict(zip(*np.unique(length, return_counts=True))))

    final_diagnoses = get_final_diagnoses(d).ravel()

    amyloid, gm = np.split(z, 2, axis=-1)
    print(f"{np.nanmean(amyloid)=:.2f}, {np.nanstd(amyloid)=:.2f}")
    print(f"{np.nanmean(gm)=:.3f}, {np.nanstd(gm)=:.2f}")

    print("Breakdown by diagnosis|".ljust(79, "-"))
    print(
        pd.Series(final_diagnoses, name="diagnosis")
        .value_counts(normalize=True)
        .loc[diagnosis_list]
    )

    length_records = []
    for ell in set(length):
        length_pct = np.mean(length == ell)
        outcome_pct = [
            np.sum(final_diagnoses[length == ell] == d) for d in diagnosis_list
        ]
        length_records.append((ell, length_pct, *outcome_pct))

    tbl = pd.DataFrame.from_records(
        length_records,
        columns=pd.MultiIndex.from_tuples(
            [
                ("", "length"),
                ("overall", "prevalence"),
                *[("within-cluster", d) for d in diagnosis_list],
            ]
        ),
    ).fillna("---")

    print("Diagnoses by length|".ljust(79, "-"))
    print(tbl)

    init = pd.DataFrame(
        data={
            "dx": final_diagnoses,
            "amyloid": z[0, :, 0],
            "gm": z[0, :, 1],
            "adni_mem": x[0, :, 0],
            "adni_ef": x[0, :, 1],
            "adas_13": x[0, :, 2],
            "moca": x[0, :, 3],
        }
    )

    print("init|".ljust(79, "-"))
    print(f"{init.mean()}")

    print("init x dx|".ljust(79, "-"))
    dx_list = [d for d in diagnosis_list if d != "MCI_tbd"]
    print(f"{init.groupby('dx').agg('mean').loc[dx_list]}")

    lookup = return_profiling_dataframe(ids[0])

    print("Breakdown by features|".ljust(79, "-"))
    print(lookup.agg(["mean", "std"]).T)


"""
Generating ADNI summary stats...
lengths:  {2: 337, 3: 194, 4: 40}
np.nanmean(amyloid)=36.59, np.nanstd(amyloid)=43.62
np.nanmean(gm)=-0.069, np.nanstd(gm)=0.03
Breakdown by diagnosis|--------------------------------------------------------
CN     0.410
sMCI   0.392
pMCI   0.033
AD     0.165
Name: diagnosis, dtype: float64
Diagnoses by length|-----------------------------------------------------------
            overall within-cluster              
  length prevalence             CN sMCI pMCI  AD
0      2      0.590            142  122   13  60
1      3      0.340             72   93    6  23
2      4      0.070             20    9    0  11
init|--------------------------------------------------------------------------
amyloid    34.611
gm         -0.064
adni_mem    0.637
adni_ef     0.596
adas_13    12.769
moca       24.208
dtype: float64
init x dx|---------------------------------------------------------------------
      amyloid     gm  adni_mem  adni_ef  adas_13   moca
dx                                                     
CN     19.765 -0.054     1.111    0.951    8.701 25.756
sMCI   29.473 -0.062     0.574    0.578   12.460 24.290
pMCI   72.561 -0.079     0.098    0.485   18.789 22.789
AD     76.139 -0.088    -0.283   -0.224   22.415 20.447
Breakdown by features|---------------------------------------------------------
                            mean     std
is_female                  0.475   0.500
edu_yrs                   16.378   2.624
apoe4_pos                  0.399   0.490
mmse_less_age              0.003   2.107
ad_outcome                 0.165   0.371
days_to_ad_or_last_obs 1,101.137 558.593
"""
