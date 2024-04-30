#!/usr/bin/env Rscript

library(lme4)
library(lmerTest)
library(emmeans)
library(magrittr)
library(dplyr)

file.path("posthoc", "results", "mmse_by_cluster_over_time.csv") %>%
  read.csv() %>%
  data.frame() %>%
  mutate_at(c("id", "cluster"), as.factor) %>%
  lmer(mmse_age_adjusted ~ cluster * time_in_years + (1 | id), data = .) %>%
  {
    (.) %>%
      anova() %>%
      print()
    (.) %>%
      summary() %>%
      print()
    (.) %>%
      contest(., list(
        `A_init vs. D_init` = c(1, 0, 0, -1, 0, 0, 0, 0)
      )) %>%
      print()
    (.) %>%
      emtrends(.,
        pairwise ~ "cluster",
        var = "time_in_years",
        lmer.df = "satterthwaite"
      ) %>%
      print()
  }

# Type III Analysis of Variance Table with Satterthwaite's method
#                       Sum Sq Mean Sq NumDF  DenDF F value    Pr(>F)
# cluster               452.45  150.82     3 874.18  61.848 < 2.2e-16 ***
# time_in_years         383.08  383.08     1 932.24 157.094 < 2.2e-16 ***
# cluster:time_in_years 469.12  156.37     3 929.09  64.127 < 2.2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
# Formula: mmse_age_adjusted ~ cluster * time_in_years + (1 | id)
#    Data: .

# REML criterion at convergence: 6015.3

# Scaled residuals:
#     Min      1Q  Median      3Q     Max
# -5.2986 -0.3892  0.1227  0.4708  3.6545

# Random effects:
#  Groups   Name        Variance Std.Dev.
#  id       (Intercept) 2.565    1.601
#  Residual             2.439    1.562
# Number of obs: 1415, groups:  id, 571

# Fixed effects:
#                         Estimate Std. Error        df t value Pr(>|t|)
# (Intercept)              1.13797    0.17088 884.84358   6.660 4.82e-11 ***
# clusterB                -0.41122    0.23725 876.33363  -1.733 0.083397 .
# clusterC                -0.47972    0.24027 878.46576  -1.997 0.046179 *
# clusterD                -3.74447    0.29212 876.93062 -12.818  < 2e-16 ***
# time_in_years            0.04423    0.05463 922.39442   0.810 0.418340
# clusterB:time_in_years  -0.12752    0.07335 923.26135  -1.738 0.082460 .
# clusterC:time_in_years  -0.28432    0.07476 921.58593  -3.803 0.000152 ***
# clusterD:time_in_years  -1.24457    0.09451 937.76729 -13.169  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Correlation of Fixed Effects:
#             (Intr) clstrB clstrC clstrD tm_n_y clB:__ clC:__
# clusterB    -0.720
# clusterC    -0.711  0.512
# clusterD    -0.585  0.421  0.416
# time_in_yrs -0.472  0.340  0.335  0.276
# clstrB:tm__  0.351 -0.471 -0.250 -0.205 -0.745
# clstrC:tm__  0.345 -0.248 -0.471 -0.202 -0.731  0.544
# clstrD:tm__  0.273 -0.196 -0.194 -0.462 -0.578  0.430  0.422
#                       Sum Sq  Mean Sq NumDF   DenDF   F value       Pr(>F)
# A_init vs. D_init   336.1514 336.1514     1 880.933 137.85158 1.121408e-29
# $emtrends
#  cluster time_in_years.trend     SE  df lower.CL upper.CL
#  A                    0.0442 0.0546 922   -0.063   0.1514
#  B                   -0.0833 0.0490 924   -0.179   0.0128
#  C                   -0.2401 0.0510 921   -0.340  -0.1399
#  D                   -1.2003 0.0771 946   -1.352  -1.0490

# Degrees-of-freedom method: satterthwaite
# Confidence level used: 0.95

# $contrasts
#  contrast estimate     SE  df t.ratio p.value
#  A - B       0.128 0.0734 923   1.738  0.3043
#  A - C       0.284 0.0748 922   3.803  0.0009
#  A - D       1.245 0.0945 938  13.169  <.0001
#  B - C       0.157 0.0707 922   2.217  0.1193
#  B - D       1.117 0.0913 939  12.229  <.0001
#  C - D       0.960 0.0925 938  10.383  <.0001

# Degrees-of-freedom method: satterthwaite
# P value adjustment: tukey method for comparing a family of 4 estimates
