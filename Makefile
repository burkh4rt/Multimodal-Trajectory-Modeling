# if using docker, requires docker daemon to be running
# first: docker build -t thistly-cross .
# then: uncomment line 9 to define `docker` variable

# otherwise, make sure, your conda or pip and 
# R environments are activated / the libraries
# are installed & on your path

# docker = true

ifdef docker
	options := --rm -ti -v $(shell pwd):/home/felixity
	run-python := docker run $(options) thistly-cross python3
	run-r := docker run $(options) thistly-cross Rscript
else
	run-python := python3
	run-r := Rscript
endif

experiments: $(wildcard *.py)
	$(run-python) model-selection.py
	$(run-python) inference-adni-xval.py
	$(run-python) inference-adni-xval-collate-results.py
	$(run-python) inference-adni-trajectories-nonlinear.py
	# $(run-python) inference-train-adni-test-macc.py

statistics: experiments $(wildcard posthoc/*.R posthoc/*.py)
	$(run-r) posthoc/lme_biomarkers_adni.R
	$(run-r) posthoc/lme_mmse_adni.R
	# $(run-r) posthoc/lme_mmse_macc.R
	$(run-r) posthoc/contingency_tbl_comp_gmm_adni.R
	$(run-r) posthoc/prognostic_auc_comp_adni.R
	$(run-python) posthoc/prognostic_survival_models_adni.py
	$(run-python) posthoc/plot_posterior_trajectories_adni.py

clean:
	-rm -rf figures/
	-rm -rf results/
	-rm -rf posthoc/results/

all: statistics
.PHONY: all statistics experiments clean
.DEFAULT: all
