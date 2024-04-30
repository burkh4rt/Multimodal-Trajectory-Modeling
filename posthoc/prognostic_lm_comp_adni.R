#!/usr/bin/env Rscript

library(broom)
library(tidyverse)


file.path("posthoc", "results", "paired_prognostic_mse_from_baseline.csv") %>%
  read.csv() %>%
  data.frame() %>%
  {
    (.) %>%
      group_by(variables) %>%
      summarise(mean = mean(MSE)) %>%
      arrange(mean) %>%
      print()
    (.) %>%
      with(., pairwise.t.test(MSE, variables,
        paired = TRUE,
        p.adjust.method = "none"
      )) %>%
      tidy() %>%
      filter("('our_in',)" == group1) %>%
      print(n = nrow(.))
  }

# # A tibble: 11 × 2
# variables                                                     mean
# <chr>                                                        <dbl>
# 1 ('adni_m', 'adni_e', 'moca_i', 'adas13', 'amyloi', 'gm_ini') 0.880
# 2 ('adni_m', 'adni_e', 'moca_i', 'adas13')                     0.893
# 3 ('adas13',)                                                  0.900
# 4 ('our_in',)                                                  0.916
# 5 ('adni_m',)                                                  0.982
# 6 ('moca_i',)                                                  0.998
# 7 ('adni_e',)                                                  1.00
# 8 ('amyloi', 'gm_ini')                                         1.00
# 9 ('amyloi',)                                                  1.02
# 10 ('gm_ini',)                                                 1.10
# 11 ('mmse_i',)                                                 1.14
# # A tibble: 10 × 3
# group1      group2                                                       p.value
# <chr>       <chr>                                                          <dbl>
# 1 ('our_in',) ('adas13',)                                                   0.745
# 2 ('our_in',) ('adni_e',)                                                   0.139
# 3 ('our_in',) ('adni_m', 'adni_e', 'moca_i', 'adas13', 'amyloi', 'gm_ini')  0.463
# 4 ('our_in',) ('adni_m', 'adni_e', 'moca_i', 'adas13')                      0.683
# 5 ('our_in',) ('adni_m',)                                                   0.178
# 6 ('our_in',) ('amyloi', 'gm_ini')                                          0.109
# 7 ('our_in',) ('amyloi',)                                                   0.0663
# 8 ('our_in',) ('gm_ini',)                                                   0.0449
# 9 ('our_in',) ('mmse_i',)                                                   0.0134
# 10 ('our_in',) ('moca_i',)                                                  0.135
