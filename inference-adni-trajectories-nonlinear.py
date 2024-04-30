#!/usr/bin/env python3

"""
Compares mixtures of state space models with nonlinear component models
"""

import os
import textwrap

from framework import marginalizable_mixture_model as mixmodel
from framework_extended import state_space_model_knn as ssm_knn
from framework_extended import (
    state_space_model_linear_trans_knn_meas as ssm_hybrid,
)
from framework_extended import state_space_model_mixture as ssmm
from util import util_adni as data
from util import util_state_space as util

alpha = 1.0
n_cluster_list = [3, 4]


def main():
    (
        z_orig,
        x,
        d,
        ids,
        time,
        mmse,
        approx_age,
    ) = data.get_trajectories(return_mmse=True, return_approx_age=True)
    z, std_param = util.standardize(z_orig, return_params=True)

    for n_clusters in n_cluster_list:
        print(f"|{n_clusters=}|".upper().center(79, "="))

        print("Mixture of k-NN-based State Space models".ljust(79, "-"))
        best_mdl_knn = ssmm.StateSpaceMixtureModel(
            n_clusters=n_clusters,
            data=(z, x),
            component_model=ssm_knn.StateSpaceKNN,
            component_model_hyperparams={"n_neighbors": [5, 10, 15]},
        ).fit(n_restarts=1000)
        data.set_model_correspondence(best_mdl_knn, d)
        data.plot_2d_trajectories(
            best_mdl_knn,
            savename=os.path.join(
                "figures", f"r7_adni_knn{n_clusters}cluster.pdf"
            ),
            title="",
            states=z_orig,
            xlabel="β-amyloid burden (centiloid)",
        )
        data.generate_outcome_table(best_mdl_knn, d)
        print(f"{best_mdl_knn.hex_hash=}")

        print("-" * 79)
        print("Mixture of state space models with linear transition model and")
        print("k-NN-based measurement model".ljust(79, "-"))
        best_mdl_hybrid = ssmm.StateSpaceMixtureModel(
            n_clusters=n_clusters,
            data=(z, x),
            component_model=ssm_hybrid.StateSpaceHybrid,
            component_model_hyperparams={
                "n_neighbors": [5, 10, 15],
                "alpha": 1.0,
            },
        ).fit(n_restarts=1000)
        data.set_model_correspondence(best_mdl_hybrid, d)
        data.plot_2d_trajectories(
            best_mdl_hybrid,
            savename=os.path.join(
                "figures", f"r7_adni_hybrid{n_clusters}cluster.pdf"
            ),
            title="",
            states=z_orig,
            xlabel="β-amyloid burden (centiloid)",
        )
        data.generate_outcome_table(best_mdl_hybrid, d)
        print(f"{best_mdl_hybrid.hex_hash=}")

        print("-" * 79)
        print("Mixture of linear Gaussian State Space models".ljust(79, "-"))
        best_mdl_lg0 = mixmodel.MMLinGaussSS_marginalizable(
            n_clusters=n_clusters,
            states=z,
            observations=x,
            random_seed=0,
            init="kmeans",
            alpha=1.0,
        ).train_with_multiple_random_starts(n_starts=1000)
        data.set_model_correspondence(best_mdl_lg0, d)
        best_mdl_lg0.to_pickle()  # update correspondence
        data.generate_outcome_table(best_mdl_lg0, d)
        data.plot_2d_trajectories(
            best_mdl_lg0,
            savename=os.path.join(
                "figures", f"r7_adni_lg{n_clusters}cluster.pdf"
            ),
            title="",
            states=z_orig,
            xlabel="β-amyloid burden (centiloid)",
        )
        print(f"{best_mdl_lg0.hex_hash=}")

    print("\n".join(textwrap.wrap(f"{std_param=}")))


if __name__ == "__main__":
    main()


"""
=================================|N_CLUSTERS=3|================================
Mixture of k-NN-based State Space models---------------------------------------
             overall within-cluster                  
  cluster prevalence             CN  sMCI  pMCI    AD
0       A      0.448          0.547 0.438 0.004 0.012
1       B      0.152          0.425 0.310 0.057 0.207
2       C      0.399          0.250 0.373 0.057 0.320
best_mdl_knn.hex_hash='56f3377fc9722c4235de40c97418399e'
-------------------------------------------------------------------------------
Mixture of state space models with linear transition model and
k-NN-based measurement model---------------------------------------------------
             overall within-cluster                  
  cluster prevalence             CN  sMCI  pMCI    AD
0       A      0.357          0.632 0.363 0.005 0.000
1       B      0.417          0.420 0.479 0.034 0.067
2       C      0.226          0.039 0.279 0.078 0.605
best_mdl_hybrid.hex_hash='076acda7c691dc59834c29d751f1779b'
-------------------------------------------------------------------------------
Mixture of linear Gaussian State Space models----------------------------------
             overall within-cluster                  
  cluster prevalence             CN  sMCI  pMCI    AD
0       A      0.534          0.557 0.423 0.007 0.013
1       B      0.340          0.325 0.438 0.077 0.160
2       C      0.126          0.014 0.139 0.028 0.819
best_mdl_lg0.hex_hash='49cb9120d3a1713763a0bdbdb97779bb'
=================================|N_CLUSTERS=4|================================
Mixture of k-NN-based State Space models---------------------------------------
             overall within-cluster                  
  cluster prevalence             CN  sMCI  pMCI    AD
0       A      0.299          0.690 0.310 0.000 0.000
1       B      0.151          0.500 0.465 0.012 0.023
2       C      0.413          0.309 0.508 0.055 0.127
3       D      0.137          0.000 0.141 0.064 0.795
best_mdl_knn.hex_hash='78753183e9dab8da210f3c2504eb00d0'
-------------------------------------------------------------------------------
Mixture of state space models with linear transition model and
k-NN-based measurement model---------------------------------------------------
             overall within-cluster                  
  cluster prevalence             CN  sMCI  pMCI    AD
0       A      0.366          0.636 0.364 0.000 0.000
1       B      0.207          0.415 0.458 0.068 0.059
2       C      0.294          0.310 0.488 0.042 0.161
3       D      0.133          0.000 0.158 0.053 0.789
best_mdl_hybrid.hex_hash='d92577bb16d0e66428f3060b4b8c0a9e'
-------------------------------------------------------------------------------
Mixture of linear Gaussian State Space models----------------------------------
             overall within-cluster                  
  cluster prevalence             CN  sMCI  pMCI    AD
0       A      0.349          0.623 0.377 0.000 0.000
1       B      0.196          0.420 0.509 0.027 0.045
2       C      0.320          0.328 0.437 0.077 0.158
3       D      0.135          0.039 0.156 0.026 0.779
best_mdl_lg0.hex_hash='2283204edb95b4aba3619c6427e992fa'
std_param={'arr_mn': array([[[-36.7379    ,  -0.24768382]]]),
'arr_mx': array([[[2.2007877e+02, 9.7658022e-04]]])}
"""
