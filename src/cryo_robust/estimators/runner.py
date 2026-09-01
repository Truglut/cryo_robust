from pathlib import Path
from typing import Iterable, Mapping, Any

import mrcfile
import torch

from cryo_robust.comparison.domain.runs import AVERAGE_NAME, MethodRun
from cryo_robust.comparison.domain.runs import MEDIAN_NAME
from cryo_robust.estimators.admm import ADMMSolver
from cryo_robust.estimators.base import Estimator
from cryo_robust.estimators.construction import build_estimator
from cryo_robust.estimators.data import ImageBatch
from cryo_robust.estimators.gmm import RecursiveGMMEstimator
from cryo_robust.estimators.results import EstimatorResult, WeightSet


def load_reference(
    path: str | Path | None, device: str | torch.device
) -> torch.Tensor | None:
    """
    Loads the starting reference from the given path and on the given device, 
    or returns None if the path is None.
    """
    if path is None:
        return None
    return torch.as_tensor(mrcfile.read(path), dtype=torch.float32, device=device)


def fit_estimator(
    estimator: Estimator,
    image_batch: ImageBatch,
    reference: torch.Tensor | None = None,
    *,
    plot_gmm: bool = False,
    method_name: str = "GMM",
) -> EstimatorResult:
    """
    Fits the estimator on the given image batch, starting from the specified
    reference.

    Parameters
    ----------
    estimator : Estimator
        Estimator to fit on the images. Must have a `fit()` method that returns
        a `EstimatorResult` object containing an `average` field.
    image_batch: ImageBatch
        Set of images to perform the estimation (i.e. robust averaging) on.
    reference: torch.Tensor, optional
        Initial reference for the estimation process. If set to None, the estimator
        will choose its own default (generally the mean of all images). Default is None.
    plot_gmm : bool
        If ``plot_gmm`` is True and ``estimator`` is of type ``RecursiveGMMEstimator``,
        then plots of the GMM fits will be generated and shown during the estimation
        process. Default is False.
    method_name : str
        Name of the method, used for the GMM plots in case they are requested.
    """
    # Handle ADMM separately because it needs two references
    if isinstance(estimator, ADMMSolver):
        return estimator.fit(
            image_batch,
            initial_reference_real=reference,
            initial_reference_fourier=(
                None
                if reference is None
                else torch.fft.rfft2(reference, norm=image_batch.norm)
            ),
        )

    # Handle GMM separately because it has plotting logic
    elif isinstance(estimator, RecursiveGMMEstimator):
        return estimator.fit(
            image_batch,
            reference=reference,
            plot_fits=plot_gmm,
            plot_title=method_name,
        )

    # Otherwise just fit the estimator
    return estimator.fit(image_batch, reference=reference)


def run_estimators(
    method_configs: Iterable[Mapping[str, Any]],
    image_batch: ImageBatch,
    *,
    plot_gmm: bool = False,
    add_avg: bool = False,
    add_median: bool = False,
) -> dict[str, MethodRun]:
    """
    Build and run the configured estimation methods.

    Parameters
    ----------
    method_configs : Iterable[Mapping[str, Any]]
        Estimator configuration blocks.
    image_batch : ImageBatch
        Images on which the estimators are fitted.
    plot_gmm : bool, optional
        Whether to show GMM fit diagnostics.
    add_average : bool, optional
        Include the sample average as a baseline.
    add_median : bool, optional
        Include the sample median as a baseline.

    Returns
    -------
    dict[str, MethodRun]
        Estimation runs keyed by method name.
    """
    results = {}

    # Iterate over methods to run them and save results
    for method_cfg in method_configs:
        method_name = method_cfg["name"]
        print(f"Running {method_name}...")

        # Build and fit the estimator
        estimator = build_estimator(method_cfg, image_batch)
        reference = load_reference(
            method_cfg.get("initial_reference"), image_batch.device
        )

        estimator_result = fit_estimator(
            estimator,
            image_batch,
            reference,
            plot_gmm=plot_gmm,
            method_name=method_name,
        )

        results[method_name] = MethodRun(
            estimator=estimator, result=estimator_result, initial_reference=reference
        )

    # Add results of sample average and median if requested
    if add_avg:
        average_result = EstimatorResult(
            average=image_batch.ensure_real().mean(dim=0),
            weights=WeightSet(
                real=torch.ones((image_batch.n_images, 1, 1), device=image_batch.device)
            ),
        )
        results[AVERAGE_NAME] = MethodRun(estimator=None, result=average_result)
    if add_median:
        median_result = EstimatorResult(
            average=image_batch.ensure_real().median(dim=0).values,
        )
        results[MEDIAN_NAME] = MethodRun(estimator=None, result=median_result)

    return results
