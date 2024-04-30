#!/usr/bin/env Rscript

library(tidyverse)
library(broom)

file.path("posthoc", "results", "paired_concordances_from_baseline_meas_multiv.csv") %>%
  read.csv() %>%
  data.frame() %>%
  {
    (.) %>%
      group_by(variables) %>%
      summarise(mean = mean(concordance)) %>%
      arrange(mean) %>%
      print()
    (.) %>%
      with(., pairwise.t.test(concordance, variables,
        paired = TRUE,
        p.adjust.method = "none"
      )) %>%
      tidy() %>%
      filter("('our_in',)" == group1) %>%
      print(n = nrow(.))
  }

# # A tibble: 10 × 2
# variables                                                     mean
# <chr>                                                        <dbl>
# 1 ('gm_ini',)                                                  0.703
# 2 ('adni_e',)                                                  0.749
# 3 ('moca_i',)                                                  0.801
# 4 ('amyloi',)                                                  0.807
# 5 ('adni_m',)                                                  0.829
# 6 ('adas13',)                                                  0.830
# 7 ('amyloi', 'gm_ini')                                         0.833
# 8 ('our_in',)                                                  0.836
# 9 ('adni_m', 'adni_e', 'moca_i', 'adas13')                     0.851
# 10 ('adni_m', 'adni_e', 'moca_i', 'adas13', 'amyloi', 'gm_ini') 0.887
# # A tibble: 9 × 3
# group1      group2                                                       p.value
# <chr>       <chr>                                                          <dbl>
# 1 ('our_in',) ('adas13',)                                                  0.875
# 2 ('our_in',) ('adni_e',)                                                  0.0768
# 3 ('our_in',) ('adni_m', 'adni_e', 'moca_i', 'adas13', 'amyloi', 'gm_ini') 0.0543
# 4 ('our_in',) ('adni_m', 'adni_e', 'moca_i', 'adas13')                     0.662
# 5 ('our_in',) ('adni_m',)                                                  0.816
# 6 ('our_in',) ('amyloi', 'gm_ini')                                         0.772
# 7 ('our_in',) ('amyloi',)                                                  0.0134
# 8 ('our_in',) ('gm_ini',)                                                  0.00289
# 9 ('our_in',) ('moca_i',)                                                  0.339
