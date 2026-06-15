# cryo-robust

Code for robust estimation experiments in cryo-EM image processing, developed as part of a Master's thesis on robust estimation methods in cryo-EM.

The repository contains implementations of several estimators, simulation and evaluation tools, and utilities for generating LaTeX reports with figures and tables. The main focus is the comparison of robust estimators for noisy cryo-EM image data in the presence of outliers.

## Overview

Cryo-electron microscopy image processing often requires estimating representative averages from very noisy particle images. In this project, different robust estimation strategies are implemented and compared under controlled simulation settings.

The code includes:

- Robust M-estimators based on Iteratively Reweighted Least Squares (IRLS).
- Real-space and Fourier-space estimator variants.
- An ADMM-based estimator combining real and Fourier updates.
- GMM-based reweighting.
- Simulation and evaluation tools.
- Classification and reconstruction metrics calculation.
- LaTeX report generation utilities.

The repository is intended as research code accompanying the thesis, not as a general-purpose cryo-EM software package.

## Repository structure

```text
.
├── src/cryo_robust/
│   ├── estimators/          # Robust estimators, weights, distances and data containers
│   ├── comparison/          # Simulation, evaluation, metrics, plotting and report utilities
│   └── utils/               # Shared helper utilities
├── scripts/                 # Scripts used to run experiments and processing utilities
├── tests/                   # Basic pytest test suite
├── docs/                    # Short documentation about estimators, experiments and tests
├── data/                    # Small input data files used by the simulation setup
├── pyproject.toml
└── README.md
```

- `src/cryo_robust/estimators`: robust estimators and related data structures.
- `src/cryo_robust/comparison`: simulation, evaluation, metrics, visualization and LaTeX report generation.
- `src/cryo_robust/utils`: shared helper utilities, such as real- or fourier-space masks.
- `scripts`: command-line scripts used to run the experiments.
- `data`: small input data required to reproduce the simulated experiments.
- `docs`: short documentation about estimators, experiments and tests.
- `tests`: basic pytest test suite.

## Installation

From the root of the repository:

```bash
pip install -e .
```

For development, install the package together with the testing dependencies, included in in the ".[dev]" package:

```bash
pip install -e ".[dev]"
```

To be able to view the images produced by the estimation methods in the `napari` viewer (through the `--show-images` argument flag in the estimator runs scripts), install the package with visualization dependencies:

```bash
pip install -e ".[visualization]"
```

## Running tests

The repository includes a small pytest suite covering the basic behavior of the weight functions, distance functions, estimators, and core metrics.

```bash
pytest -q
```

*Note: this requires installing the package with the development dependencies, `pip install -e ".[dev]"`.*

These tests are not intended to reproduce all thesis experiments. Their purpose is to detect implementation errors in the most central components of the code.

## Data

The repository includes only small input files required to generate the simulated datasets used in the experiments.

Generated datasets, large intermediate files, experiment outputs, and compiled reports are not tracked in the repository. They should be regenerated locally using the experiment scripts.

## Experiments

The experiment pipeline is used to:

1. Build or load image datasets.
2. Run several estimators on the same data.
3. Compute classification and reconstruction metrics.
4. Generate plots, tables and LaTeX report components.

## Documentation

Additional documentation is available in the `docs/` directory:

- `docs/estimators.md`: overview of the estimator classes and their design.
- `docs/experiments.md`: brief description of the experiment and report-generation pipeline.
- `docs/testing.md`: explanation of the test suite and a development smoke check.

## Main estimator components

The estimator code is organized around a few central abstractions:

- `ImageBatch`: canonical container for real-space and Fourier-space image data.
- `WeightSet`: container for real-space and Fourier-space weights.
- `EstimatorResult`: standard output object returned by estimators.
- `IRLSSolver`: single-space IRLS solver.
- `IRLSFourier`, `JointIRLSFourier` and `FlatteningIRLSFourier`: Fourier-space IRLS variants.
- `RecursiveGMMEstimator`: recursive GMM-based robust averaging estimator.
- `ADMMSolver`: experimental estimator coupling real-space and Fourier-space updates.

See `docs/estimators.md` for more details.

## Scope and limitations

This repository contains research code developed for a Master's thesis. The code is designed to make the experiments reproducible and the implemented estimators inspectable, but it is not intended to be a production-ready cryo-EM processing package.

In particular:

- The APIs may still change.
- Only selected components are covered by tests.
- Some scripts are specific to the thesis experiments.
- Generated reports should be inspected manually when used for final analysis.

## Thesis context

This code accompanies the Master's thesis:
**Estimación robusta en el procesamiento de imagen de criomicroscopía electrónica**

The main goal of the project is to study robust estimation methods for cryo-EM image averaging, with emphasis on statistical modeling, robustness to outliers, Fourier-space methods, and quantitative comparison of estimator performance.

Code for robust estimation experiments for cryo-EM class averaging, developed as part of a Master's thesis on robust estimation in cryo-EM image processing.

The repository contains implementations of robust estimators, simulation and evaluation pipelines, and tools for generating LaTeX reports with figures and tables.

***

Author: Andrés Contreras de Santos
