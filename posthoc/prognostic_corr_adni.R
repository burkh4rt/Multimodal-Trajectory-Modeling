#!/usr/bin/env Rscript

library(cocor)
library(magrittr)
library(dplyr)
library(broom)

drop_outliers <- function(df, var) {
  df %>%
    filter(abs({{ var }} - mean({{ var }}, na.rm = TRUE))
    < 3 * sd({{ var }}, na.rm = TRUE))
}

file.path("posthoc", "results", "prognostics_all.csv") %>%
  read.csv() %>%
  data.frame() %>%
  {
    (.) %>%
      drop_outliers(., mmse_init) %>%
      drop_outliers(., our_index_snapshot_init) %>%
      cocor(
        ~ ann_mmse_change_age_adjusted + mmse_init |
          ann_mmse_change_age_adjusted + our_index_snapshot_init,
        .,
        test = "steiger1980"
      ) %>%
      print()
    (.) %>%
      drop_outliers(., moca_init) %>%
      drop_outliers(., our_index_snapshot_init) %>%
      cocor(
        ~ ann_mmse_change_age_adjusted + moca_init |
          ann_mmse_change_age_adjusted + our_index_snapshot_init,
        .,
        test = "steiger1980"
      ) %>%
      print()
    (.) %>%
      drop_outliers(., our_index_snapshot_init) %>%
      do(tidy(cor.test(
        .$our_index_snapshot_init,
        .$ann_mmse_change_age_adjusted
      ))) %>%
      print()
    (.) %>%
      drop_outliers(., mmse_init) %>%
      do(tidy(cor.test(
        .$mmse_init,
        .$ann_mmse_change_age_adjusted
      ))) %>%
      print()
    (.) %>%
      drop_outliers(., moca_init) %>%
      do(tidy(cor.test(
        .$moca_init,
        .$ann_mmse_change_age_adjusted
      ))) %>%
      print()
  }

#   Results of a comparison of two overlapping correlations based on dependent groups

# Comparison between r.jk (ann_mmse_change_age_adjusted, mmse_init) = 0.0228 and r.jh (ann_mmse_change_age_adjusted, our_index_snapshot_init) = 0.3522
# Difference: r.jk - r.jh = -0.3294
# Related correlation: r.kh = 0.3718
# Data: .: j = ann_mmse_change_age_adjusted, k = mmse_init, h = our_index_snapshot_init
# Group size: n = 553
# Null hypothesis: r.jk is equal to r.jh
# Alternative hypothesis: r.jk is not equal to r.jh (two-sided)
# Alpha: 0.05

# steiger1980: Steiger's (1980) modification of Dunn and Clark's z (1969) using average correlations
#   z = -7.1345, p-value = 0.0000
#   Null hypothesis rejected


#   Results of a comparison of two overlapping correlations based on dependent groups

# Comparison between r.jk (ann_mmse_change_age_adjusted, moca_init) = 0.2493 and r.jh (ann_mmse_change_age_adjusted, our_index_snapshot_init) = 0.3395
# Difference: r.jk - r.jh = -0.0902
# Related correlation: r.kh = 0.4082
# Data: .: j = ann_mmse_change_age_adjusted, k = moca_init, h = our_index_snapshot_init
# Group size: n = 553
# Null hypothesis: r.jk is equal to r.jh
# Alternative hypothesis: r.jk is not equal to r.jh (two-sided)
# Alpha: 0.05

# steiger1980: Steiger's (1980) modification of Dunn and Clark's z (1969) using average correlations
#   z = -2.0660, p-value = 0.0388
#   Null hypothesis rejected

# # A tibble: 1 × 8
#   estimate statistic  p.value parameter conf.low conf.high method                               alternative
#      <dbl>     <dbl>    <dbl>     <int>    <dbl>     <dbl> <chr>                                <chr>
# 1    0.396      10.2 1.36e-22       560    0.324     0.464 Pearson's product-moment correlation two.sided
# # A tibble: 1 × 8
#   estimate statistic p.value parameter conf.low conf.high method                               alternative
#      <dbl>     <dbl>   <dbl>     <int>    <dbl>     <dbl> <chr>                                <chr>
# 1   0.0570      1.35   0.177       560  -0.0259     0.139 Pearson's product-moment correlation two.sided
# # A tibble: 1 × 8
#   estimate statistic  p.value parameter conf.low conf.high method                               alternative
#      <dbl>     <dbl>    <dbl>     <int>    <dbl>     <dbl> <chr>                                <chr>
# 1    0.282      6.96 9.54e-12       560    0.204     0.357 Pearson's product-moment correlation two.sided
