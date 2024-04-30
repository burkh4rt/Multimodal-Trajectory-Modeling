#!/usr/bin/env python3

"""
plot posterior-informed average cluster trajectories
"""

import pathlib
import sys

import numpy as np
import pandas as pd

whereami = pwd = pathlib.Path(__file__).absolute().parent
sys.path.append(str(pwd.parent))

from util import util_adni as data
from util import util_state_space as st_sp

n_splits, n_clusters = 10, 4
soft_assignment = True


def main():
    z, x, d, ids, time = data.get_trajectories()

    for mdl in ["mtm", "mtm_init", "gmm_init"]:
        df_post = pd.read_csv(
            whereami.joinpath(
                "results",
                f"ADNI_r7_xval{n_splits}_{n_clusters}clusters_{mdl}_post.csv",
            ),
            index_col="id",
        )

        st_sp.plot_weighted_means_2d_trajectories(
            weights=df_post.values,
            values=z,
            colors=data.cluster_colors,
            saveloc=whereami.parent.joinpath(
                "figures",
                f"ADNI_r7_xval{n_splits}_{n_clusters}cl_{mdl}_posterior.pdf",
            ),
            xlabel="β-amyloid burden (centiloid)",
            ylabel="Gray matter density",
            xlim=(-50.0, 230.0),
            ylim=(-0.275, 0.025),
            soft_assignment=soft_assignment,
            arrow_width=0.5,
            elide_at=[None, None, None, 3],
        )

        st_sp.plot_weighted_means_2d_trajectories(
            weights=df_post.values,
            values=np.stack((x[..., 0], z[..., -1]), axis=-1),
            colors=data.cluster_colors,
            saveloc=whereami.parent.joinpath(
                "figures",
                f"ADNI_r7_xval{n_splits}_{n_clusters}cl_{mdl}_"
                f"gm_vs_adnimem_posterior.pdf",
            ),
            xlabel="ADNI-Mem",
            xlim=(np.nanmin(x[..., 0]) - 0.2, np.nanmax(x[..., 0]) + 0.2),
            ylabel="Gray matter density",
            ylim=(-0.275, 0.025),
            soft_assignment=soft_assignment,
            arrow_width=0.01,
            elide_at=[None, None, None, 3],
        )


if __name__ == "__main__":
    main()
