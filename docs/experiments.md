# Experiments and reports

This document gives a short overview of the experimental pipeline used in the project.

## Purpose

The experiment code is used to compare several robust estimators for cryo-EM image averaging. The simulations generate image stacks with inliers and outliers, run different estimators on the same input data, compute quantitative metrics, and optionally generate plots or LaTeX reports with figures and tables.

## Main components

The comparison pipeline is organized around the following tasks:

1. Build or load an image dataset.
2. Run a collection of estimators.
3. Store the estimated averages and final weights.
4. Compute classification metrics using the known inlier/outlier labels.
5. Compute reconstruction metrics when a ground-truth image is available.
6. Generate plots, tables, and LaTeX report components.

## Metrics

The basic metrics include:

- Classification metrics based on the final weights assigned by each method and the known image labels. These include soft precision and soft recall (as defined in Huang and Tagare, 2016), average precision and ROC-AUC.
- Reconstruction metrics comparing the estimated average with the known ground truth. These include RMSE and cross-correlation.
- Half-set reconstruction metrics based on splitting the input image stack.

Some parts of the code also compute Fourier Ring Correlation and metrics by Fourier ring. These are used for more detailed analysis, but they are not required for the simplest test suite.

## Reports

The report-generation code creates LaTeX output containing figures and tables for the comparison experiments. These reports are mainly intended as an internal tool for inspecting the results and producing material for the thesis.

## Running the simulations

The simulations are implemented through the `scripts/estimator_runs/run_simulation.py` script. The script takes a required `--config` argument flag with the path to the config file, which contains some parameters for the simulations. Some example config files can be found in the `configs` directory.
To run a simulation, run

```bash
python -m scripts.estimator_runs.run_simulation --config path-to-config
```

A number of additional argument flags can be added to modify several parameters of the simulations and their output. Some important ones are:

- `--snr`. Must be followed by a space-separated list of float values, used to specify the signal-to-noise ratios that will be used in the experiments. If more than one SNR value is given, multiple experiments will be run in sequence, and the reports will contain plots of estimator performance versus SNR level. Default is 0.01.
- `--report`. Generate a LaTeX report with figures and tables of the results of the simulations. Must be followed by the path to the directory where the report should be generated. If absent, no report will be generated.
- `--n-runs`. Number of simulations to run per experiment. Default is 1.

Example usage:

```bash
python -m scripts.estimator_runs.run_simulation --config configs/example_gmm --snr 0.05 0.02 0.01 0.0067 0.005 --report results/gmm_report --n-runs 10
```

For a complete list of optional arguments, run

```bash
python -m scripts.estimator_runs.run_simulation -h
```
