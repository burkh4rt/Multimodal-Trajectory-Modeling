#!/usr/bin/env Rscript

library(tidyverse)
library(broom)

file.path("posthoc", "results", "paired_prognostic_aucs_from_baseline_meas.csv") %>%
  read.csv() %>%
  data.frame() %>%
  {
    (.) %>%
      group_by(feature) %>%
      summarise(mean = mean(auc)) %>%
      arrange(mean) %>%
      print()
    (.) %>%
      with(., pairwise.t.test(auc, feature,
        paired = TRUE,
        p.adjust.method = "none"
      )) %>%
      tidy() %>%
      filter(., group1 == "('our_in',)") %>%
      print(n = nrow(.))
  }

# # A tibble: 10 × 2
# feature                  mean
# <chr>                   <dbl>
# 1 gm_init                 0.741
# 2 adni_ef_init            0.796
# 3 amyloid_init            0.810
# 4 mmse_init               0.814
# 5 moca_init               0.850
# 6 gm_amyloid              0.850
# 7 gm_mmse                 0.852
# 8 adas13_init             0.876
# 9 adni_mem_init           0.878
# 10 our_index_snapshot_init 0.878
# # A tibble: 9 × 3
# group1                  group2        p.value
# <chr>                   <chr>           <dbl>
# 1 our_index_snapshot_init adas13_init   0.917
# 2 our_index_snapshot_init adni_ef_init  0.00617
# 3 our_index_snapshot_init adni_mem_init 0.992
# 4 our_index_snapshot_init amyloid_init  0.00618
# 5 our_index_snapshot_init gm_amyloid    0.198
# 6 our_index_snapshot_init gm_init       0.00129
# 7 our_index_snapshot_init gm_mmse       0.348
# 8 our_index_snapshot_init mmse_init     0.0878
# 9 our_index_snapshot_init moca_init     0.166
