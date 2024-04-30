#!/usr/bin/env Rscript

library(DiffXTables)
library(magrittr)
library(dplyr)
library(broom)

list(
  our_table = file.path("posthoc", "results", "contingency_table_ours_test.csv") %>%
    read.csv(., row.names = "cluster") %>%
    data.matrix(),
  gmm_table = file.path("posthoc", "results", "contingency_table_gmm_init.csv") %>%
    read.csv(., row.names = "cluster") %>%
    data.matrix()
) %>%
  {
    (.) %>%
      .$our_table %>%
      chisq.test() %>%
      print()
    (.) %>%
      .$gmm_table %>%
      chisq.test() %>%
      print()
    (.) %>%
      sharma.song.test(., compensated = FALSE) %>%
      print()
    (.) %>%
      sharma.song.test(., compensated = TRUE) %>%
      print()
  }


# 	Pearson's Chi-squared test
# data:  .
# X-squared = 308.01, df = 9, p-value < 2.2e-16


# 	Pearson's Chi-squared test
# data:  .
# X-squared = 38.736, df = 9, p-value = 1.285e-05


# 	Sharma-Song Test for Second-Order Differential Contingency Tables
#                        Null table marginal is observed
# data:  .
# X-squared = 93.633, df = 9, p-value = 3.031e-16


# 	Sharma-Song Test for Second-Order Differential Contingency Tables
#                        Null table marginal is observed
# data:  .
# X-squared = 93.278, df = 9, p-value = 3.572e-16
