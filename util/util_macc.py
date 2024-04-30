#!/usr/bin/env python3

"""
Utilities for loading and plotting with the MACC dataset
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.float_format = "{:,.3f}".format

name = "MACC"
diagnosis_list = ["NCI", "CIND", "VCIND", "AD"]

cluster_colors = (
    "#0072CE",
    "#E87722",
    "#64A70B",
    "#93328E",
    "#A81538",
    "#4E5B31",
)

diagnosis_colors = tuple(
    np.flipud(
        np.array(plt.colormaps["cividis"].colors)[
            np.linspace(
                0,
                plt.colormaps["cividis"].N - 1,
                len(diagnosis_list),
            ).astype(int)
        ]
    )
)


def get_data() -> tuple:
    return tuple(
        map(
            np.load(
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    ),
                    "data",
                    "macc-trajectories.npz",
                ),
                allow_pickle=True,
            ).__getitem__,
            ["z", "x", "d", "mmse_cdrsum_cdrglobal", "ell", "ids", "ages"],
        )
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
        "->".join(l).replace("->None", "")
        for l in diagnoses.squeeze().transpose().astype(str).tolist()
    ]
    return np.array([s.split("->")[-1] for s in diagnosis_paths])


def get_annualized_mmse_changes() -> np.array:
    annualized_mmse_changes = []
    *_, mmse_cdrsum_cdrglobal, ell, _, ages = get_data()
    for i, i_ell in enumerate(ell):
        annualized_mmse_changes.append(
            (
                mmse_cdrsum_cdrglobal[i_ell - 1, i, 0]
                - mmse_cdrsum_cdrglobal[0, i, 0]
            )
            / (ages[i_ell - 1, i] - ages[0, i])
        )
    return np.array(annualized_mmse_changes)


def get_demographics(ids):
    return pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "macc-profiling.csv",
        ),
        index_col="Harmy_ID",
    ).loc[ids.ravel()]


# print off some summary stats if called as a script
if __name__ == "__main__":
    print(f"Generating {name} summary stats...")

    z, x, d, _, length, ids, ages = get_data()

    print("lengths: ", dict(zip(*np.unique(length, return_counts=True))))
    print(
        "n. GM scores: ",
        dict(
            zip(
                *np.unique(
                    (~np.isnan(z[..., 1])).astype(int).sum(axis=0),
                    return_counts=True,
                )
            )
        ),
    )
    print("ages (mean): ", ages.mean(axis=1).round(2))
    print("ages (std.): ", ages.std(axis=1).round(2))

    final_diagnoses = get_final_diagnoses(d).ravel()

    print(f"{np.nanmean(z[0,:,0])=:.2f}, {np.nanstd(z[0,:,0])=:.2f}")
    print(f"{np.nanmean(z[:,:,1])=:.3f}, {np.nanstd(z[:,:,1])=:.2f}")

    print("Breakdown by diagnosis|".ljust(79, "-"))
    print(
        pd.Series(final_diagnoses, name="diagnosis")
        .value_counts()
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

    print("Breakdown by features|".ljust(79, "-"))
    print(
        get_demographics(ids)
        .select_dtypes(include=np.number)
        .agg(["mean", "std"])
        .T
    )


"""
Generating MACC summary stats...
lengths:  {2: 21, 3: 137}
n. GM scores:  {1: 5, 2: 52, 3: 101}
ages (mean):  [72.95 74.91 76.92]
ages (std.):  [7.35 7.34 7.33]
np.nanmean(z[0,:,0])=52.96, np.nanstd(z[0,:,0])=41.03
np.nanmean(z[:,:,1])=-0.041, np.nanstd(z[:,:,1])=0.06
Breakdown by diagnosis|--------------------------------------------------------
NCI      36
CIND     50
VCIND    18
AD       54
Name: diagnosis, dtype: int64
Diagnoses by length|-----------------------------------------------------------
            overall within-cluster               
  length prevalence            NCI CIND VCIND  AD
0      2      0.133              4    9     2   6
1      3      0.867             32   41    16  48
Breakdown by features|---------------------------------------------------------
           mean   std
is_female 0.551 0.499
edu_yrs   7.854 4.764
apoe4_pos 0.304 0.461
"""
