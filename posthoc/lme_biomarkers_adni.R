#!/usr/bin/env Rscript

library(lme4)
library(lmerTest)
library(emmeans)
library(magrittr)
library(dplyr)

options(warn = -1)

file.path("posthoc", "results", "biomarkers_by_cluster_over_time.csv") %>%
  read.csv() %>%
  data.frame() %>%
  mutate_at(c("ids", "cluster"), as.factor) %>%
  {
    (.) %>%
      lmer(gm_diff ~ amyl_prev:cluster + cluster + (1 | ids), data = .) %>%
      {
        (.) %>%
          anova() %>%
          print()
        (.) %>%
          summary() %>%
          print()
        (.) %>%
          emtrends(., "cluster",
            var = "amyl_prev",
            lmer.df = "satterthwaite"
          ) %>%
          print()
      }
    (.) %>%
      lmer(adni_mem_diff ~ gm_diff:cluster + cluster + (1 | ids), data = .) %>%
      {
        (.) %>%
          anova() %>%
          print()
        (.) %>%
          summary() %>%
          print()
        (.) %>%
          emtrends(., "cluster",
            var = "gm_diff",
            lmer.df = "satterthwaite"
          ) %>%
          print()
      }
  }

##########################################
###     gm_diff ~ amyl_prev:cluster    ###
##########################################

# Type III Analysis of Variance Table with Satterthwaite's method
#                      Sum Sq    Mean Sq NumDF  DenDF F value    Pr(>F)
# cluster           0.0027556 0.00091853     3 585.44  12.034 1.181e-07 ***
# amyl_prev:cluster 0.0034847 0.00087118     4 625.00  11.414 6.014e-09 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
# Formula: gm_diff ~ amyl_prev:cluster + cluster + (1 | ids)
#    Data: df

# REML criterion at convergence: -5404.8

# Scaled residuals:
#     Min      1Q  Median      3Q     Max
# -6.1262 -0.4457  0.0143  0.4818  5.1113

# Random effects:
#  Groups   Name        Variance  Std.Dev.
#  ids      (Intercept) 8.637e-06 0.002939
#  Residual             7.633e-05 0.008736
# Number of obs: 845, groups:  ids, 571

# Fixed effects:
#                      Estimate Std. Error         df t value Pr(>|t|)
# (Intercept)        -3.389e-03  6.583e-04  5.755e+02  -5.149 3.61e-07 ***
# clusterB           -1.039e-05  1.066e-03  5.541e+02  -0.010 0.992230
# clusterC           -3.136e-03  1.076e-03  5.633e+02  -2.914 0.003713 **
# clusterD           -1.080e-02  2.003e-03  6.486e+02  -5.391 9.82e-08 ***
# amyl_prev:clusterA  4.506e-05  7.238e-05  6.810e+02   0.623 0.533794
# amyl_prev:clusterB -5.415e-05  1.497e-05  5.919e+02  -3.618 0.000323 ***
# amyl_prev:clusterC -5.619e-05  1.466e-05  5.739e+02  -3.832 0.000141 ***
# amyl_prev:clusterD -8.813e-05  2.107e-05  6.669e+02  -4.183 3.26e-05 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Correlation of Fixed Effects:
#             (Intr) clstrB clstrC clstrD amy_:A amy_:B amy_:C
# clusterB    -0.618
# clusterC    -0.612  0.378
# clusterD    -0.329  0.203  0.201
# amyl_prv:cA -0.291  0.179  0.178  0.096
# amyl_prv:cB  0.000 -0.556  0.000  0.000  0.000
# amyl_prv:cC  0.000  0.000 -0.552  0.000  0.000  0.000
# amyl_prv:cD  0.000  0.000  0.000 -0.831  0.000  0.000  0.000
#  cluster amyl_prev.trend       SE  df  lower.CL  upper.CL
#  A              4.51e-05 7.24e-05 681 -9.70e-05  1.87e-04
#  B             -5.42e-05 1.50e-05 592 -8.36e-05 -2.48e-05
#  C             -5.62e-05 1.47e-05 574 -8.50e-05 -2.74e-05
#  D             -8.81e-05 2.11e-05 667 -1.29e-04 -4.68e-05

##########################################
###   adni_mem_diff ~ gm_diff:cluster  ###
##########################################

# Type III Analysis of Variance Table with Satterthwaite's method
#                 Sum Sq Mean Sq NumDF DenDF F value    Pr(>F)
# cluster         3.3378  1.1126     3   837  6.1725 0.0003767 ***
# gm_diff:cluster 5.8478  1.4620     4   837  8.1106 2.036e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
# Formula: adni_mem_diff ~ gm_diff:cluster + cluster + (1 | ids)
#    Data: df

# REML criterion at convergence: 946.2

# Scaled residuals:
#     Min      1Q  Median      3Q     Max
# -4.0692 -0.5937  0.0101  0.6529  4.3011

# Random effects:
#  Groups   Name        Variance Std.Dev.
#  ids      (Intercept) 0.0000   0.0000
#  Residual             0.1803   0.4246
# Number of obs: 845, groups:  ids, 571

# Fixed effects:
#                   Estimate Std. Error        df t value Pr(>|t|)
# (Intercept)        0.07556    0.03140 837.00000   2.406 0.016344 *
# clusterB          -0.07795    0.04466 837.00000  -1.745 0.081319 .
# clusterC          -0.08047    0.04957 837.00000  -1.623 0.104898
# clusterD          -0.30971    0.07260 837.00000  -4.266 2.22e-05 ***
# gm_diff:clusterA  -0.30627    4.33598 837.00000  -0.071 0.943706
# gm_diff:clusterB   5.10147    3.21711 837.00000   1.586 0.113178
# gm_diff:clusterC  12.92726    3.09517 837.00000   4.177 3.27e-05 ***
# gm_diff:clusterD   8.74414    2.47531 837.00000   3.533 0.000434 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Correlation of Fixed Effects:
#             (Intr) clstrB clstrC clstrD gm_d:A gm_d:B gm_d:C
# clusterB    -0.703
# clusterC    -0.634  0.445
# clusterD    -0.433  0.304  0.274
# gm_dff:clsA  0.445 -0.313 -0.282 -0.193
# gm_dff:clsB  0.000  0.398  0.000  0.000  0.000
# gm_dff:clsC  0.000  0.000  0.547  0.000  0.000  0.000
# gm_dff:clsD  0.000  0.000  0.000  0.712  0.000  0.000  0.000
# optimizer (nloptwrap) convergence code: 0 (OK)
# boundary (singular) fit: see help('isSingular')

#  cluster gm_diff.trend   SE  df lower.CL upper.CL
#  A              -0.306 4.34 837    -8.82      8.2
#  B               5.101 3.22 837    -1.21     11.4
#  C              12.927 3.10 837     6.85     19.0
#  D               8.744 2.48 837     3.89     13.6
