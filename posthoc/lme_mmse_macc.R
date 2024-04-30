#!/usr/bin/env Rscript

library(lme4)
library(lmerTest)
library(emmeans)
library(magrittr)
library(dplyr)

file.path("posthoc", "results", "mmse_by_cluster_over_time_macc.csv") %>%
  read.csv() %>%
  data.frame() %>%
  mutate_at(c("index", "cluster"), as.factor) %>%
  lmer(mmse_age_adjusted ~ cluster:time_in_years + cluster + (1 | index),
    data = .
  ) %>%
  {
    (.) %>%
      anova() %>%
      print()
    (.) %>%
      summary() %>%
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
# cluster               124.79  62.396     2 195.11 10.8059 3.542e-05 ***
# cluster:time_in_years 159.87  53.290     3 294.97  9.2288 7.467e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
# Formula: mmse_age_adjusted ~ cluster:time_in_years + cluster + (1 | index)
#    Data: .

# REML criterion at convergence: 2454.9

# Scaled residuals:
#     Min      1Q  Median      3Q     Max
# -3.3681 -0.4455  0.0675  0.5088  2.4794

# Random effects:
#  Groups   Name        Variance Std.Dev.
#  index    (Intercept) 20.262   4.501
#  Residual              5.774   2.403
# Number of obs: 453, groups:  index, 158

# Fixed effects:
#                         Estimate Std. Error        df t value Pr(>|t|)
# (Intercept)              4.02161    0.88590 195.03968   4.540 9.85e-06 ***
# clusterB                -2.05583    1.45173 195.18560  -1.416    0.158
# clusterC                -4.54141    1.00967 194.99979  -4.498 1.18e-05 ***
# clusterA:time_in_years   0.08411    0.15947 294.92655   0.527    0.598
# clusterB:time_in_years  -0.09169    0.21174 295.79139  -0.433    0.665
# clusterC:time_in_years  -0.44700    0.08567 294.20057  -5.217 3.43e-07 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Correlation of Fixed Effects:
#             (Intr) clstrB clstrC clA:__ clB:__
# clusterB    -0.610
# clusterC    -0.877  0.535
# clstrA:tm__ -0.333  0.203  0.292
# clstrB:tm__  0.000 -0.262  0.000  0.000
# clstrC:tm__  0.000  0.000 -0.161  0.000  0.000
# $emtrends
#  cluster time_in_years.trend     SE  df lower.CL upper.CL
#  A                    0.0841 0.1595 295   -0.230    0.398
#  B                   -0.0917 0.2118 296   -0.508    0.325
#  C                   -0.4470 0.0857 294   -0.616   -0.278

# Degrees-of-freedom method: kenward-roger
# Confidence level used: 0.95

# $contrasts
#  contrast estimate    SE  df t.ratio p.value
#  A - B       0.176 0.265 295   0.663  0.7850
#  A - C       0.531 0.181 295   2.934  0.0101
#  B - C       0.355 0.228 295   1.555  0.2668

# Degrees-of-freedom method: kenward-roger
# P value adjustment: tukey method for comparing a family of 3 estimates
