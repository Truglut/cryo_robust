# Testing

The repository includes a small pytest-based test suite.

The tests are intentionally simple. Their goal is not to validate every numerical detail of the full experimental pipeline, but to detect implementation mistakes that would affect the main thesis experiments.

## What is tested

The current tests cover:

- Basic behavior of weight and distance functions.
- Consistency of weighted averages.
- Simple expected behavior of the estimators.
- Basic classification and reconstruction metrics.

The tests use small synthetic tensors with known expected behavior. For example, IRLS estimators with constant unit weights should reduce to the ordinary sample mean.

## What is not tested

The basic tests do not cover:

- Full end-to-end thesis experiments.
- Large real-data experiments.
- LaTeX report visual inspection.
- Fourier Ring Correlation details.
- Metrics computed by Fourier ring.

Some of these parts can be better checked through the development smoke simulation and by inspecting generated reports.

## Running tests

From the root of the repository:

```bash
pip install -e ".[dev]"
pytest -q
```

## Smoke check

In addition to the tests, the project can be checked with a small end-to-end simulation configuration. This configuration runs one representative estimator of each type, computes the evaluation metrics, and can optionally generate a LaTeX report.

This smoke check is not meant to replace unit tests or to serve as a full scientific validation. Instead, the smoke check should verify that:

- The dataset generation or loading step works.
- All representative estimators can run.
- Metrics are computed without errors.
- Optional report generation still works.

The configuration file for this smoke check can be found in `configs/test.yaml`. To run it, do

```bash
python -m scripts.estimator_runs.run_simulation --config configs/test.yaml
```

To generate a LaTeX report, add a `--report` flag with an argument indicating the directory where the new report should be stored:

```bash
python -m scripts.estimator_runs.run_simulation --config configs/test.yaml --report path-to-new-report
```

For a more detailed explanation on how the simulation script works and how to run it with different arguments, see `docs/experiments.md`.
