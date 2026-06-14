import yaml
from pathlib import Path
from argparse import Namespace

import numpy as np
import mrcfile
import torch

from cryo_robust.estimators import build_estimator
from cryo_robust.estimators.data import ImageBatch
from cryo_robust.estimators.base import Estimator
from cryo_robust.estimators.admm import ADMMSolver
from cryo_robust.estimators.irls import (
    IRLSFourier,
    JointIRLSFourier,
    FlatteningIRLSFourier,
)
from cryo_robust.estimators.gmm import RecursiveGMMEstimator

from cryo_robust.comparison.domain.enums import ImageSpace, AggregationStrategy
from cryo_robust.comparison.evaluation.aggregation import aggregate_weights
from cryo_robust.comparison.visualization.plotting import AVERAGE_NAME, MEDIAN_NAME

from cryo_robust.utils.masks import create_circular_mask


def load_config(config_path: str | Path, snr: float | None = None):
    """
    Loads the config file and overrides the config SNR specification if the ``snr``
    argument is not ``None``.
    """
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    if snr is not None:
        cfg.setdefault("noise", {})["snr"] = snr
    return cfg


def load_reference(
    path: str | Path | None, device: str | torch.device
) -> torch.Tensor | None:
    """
    Loads the starting reference from the given path, or returns None if the path
    is None.
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
) -> None:
    """
    Fits the estimator on the given image batch, starting from the specified
    reference
    """
    if isinstance(estimator, ADMMSolver):
        estimator.fit(
            image_batch,
            initial_reference_real=reference,
            initial_reference_fourier=(
                None
                if reference is None
                else torch.fft.rfft2(reference, norm=image_batch.norm)
            ),
        )
    elif isinstance(estimator, RecursiveGMMEstimator):
        estimator.fit(
            image_batch,
            reference=reference,
            plot_fits=plot_gmm,
            plot_title=method_name,
        )
    else:
        estimator.fit(image_batch, reference=reference)


def run_estimators(
    cfg: dict,
    image_batch: ImageBatch,
    args: Namespace,
    add_avg: bool = False,
    add_median: bool = False,
) -> dict:
    """
    Builds all of the estimators that are specified in ``cfg["experiment"]["methods"]``
    and runs them on the image batch.
    Stores and returns their results as a dict with the following keys:
    - ``"estimator"``. The ``Estimator`` object that implements the estimation method.
    - ``"reference"``. The initial reference the method used.
    - ``"avg"``. The final, real-space estimate given by the estimator.
    - ``"weights"``. A dictionary mapping every ImageSpace to the set of weights the
        estimator produced in said space, or None if the estimator does not operate
        in that space.
    """
    results = {}

    # Iterate over methods to run them and save results
    for method_cfg in cfg["experiment"]["methods"]:
        method_name = method_cfg["name"]
        print(f"Running {method_name}...")

        # Build and fit the estimator
        estimator = build_estimator(method_cfg, image_batch, device=args.device)
        reference = load_reference(method_cfg.get("initial_reference"), args.device)
        fit_estimator(
            estimator,
            image_batch,
            reference,
            plot_gmm="gmm" in args.plot,
            method_name=method_name,
        )

        # Save results in dict
        results[method_name] = {
            "estimator": estimator,
            "reference": reference,
            "avg": estimator.avg,
            "weights": estimator.final_weights,
        }

    # Add results of sample average and median if requested
    if add_avg:
        results[AVERAGE_NAME] = {
            "avg": image_batch.ensure_real().mean(dim=0),
            "weights": {
                space: torch.ones((image_batch.n_images, 1, 1), device=args.device)
                for space in ImageSpace
            },
            "reference": None,
            "estimator": AVERAGE_NAME,
        }
    if add_median:
        results[MEDIAN_NAME] = {
            "avg": image_batch.ensure_real().median(dim=0).values,
            "weights": {space: None for space in ImageSpace},
            "reference": None,
            "estimator": MEDIAN_NAME,
        }

    return results


def apply_mask(images_tensor: torch.Tensor, mask_radius: float, inplace: bool = False):
    """
    Applies a circular mask to a batch of images, optionally modifying the 
    input tensor in-place
    """
    # Create mask on device
    image_shape = tuple(images_tensor.shape[1:])
    mask_np = create_circular_mask(image_shape, mask_radius)
    mask_tensor = torch.from_numpy(mask_np).to(device=images_tensor.device)

    masked_images = images_tensor if inplace else images_tensor.clone()

    # Apply mask to images
    masked_images *= mask_tensor

    return masked_images, mask_tensor


def canonical_image_weights(
    estimator: Estimator, final_weights: dict[ImageSpace, torch.Tensor]
):
    """
    Gets a canonical set of image weights for each estimator type, used for 
    identifying 'good' and 'bad' image subsets
    """
    if isinstance(estimator, IRLSFourier):
        weights = 0.5 * (
            final_weights[ImageSpace.FOURIER_REAL]
            + final_weights[ImageSpace.FOURIER_IMAG]
        )
    elif isinstance(estimator, (JointIRLSFourier, FlatteningIRLSFourier)):
        weights = final_weights[ImageSpace.FOURIER_REAL]
    else:
        weights = final_weights[ImageSpace.REAL]

    if weights is None:
        raise ValueError(
            f"canonical_image_weights: extracted None for estimator of type {type(estimator)}"
        )

    return aggregate_weights(weights, AggregationStrategy.MEAN)


def process_and_save_subsets(
    results: dict,
    image_path: Path,
    images_save: np.ndarray,
    args: Namespace,
    snr: float | None = None,
) -> None:
    """
    For each of the provided quantiles and weight thresholds (through the 
    command-line arguments stored in ``args``), extracts the subsets of images with
    highest and lowest weights.
    Saves these subsets to a file if requested.
    """
    # Initialize quantiles and thresholds arrays from args
    quantiles = np.array(args.quantiles) if args.quantiles else np.array([])
    fixed_thresholds = np.array(args.thresholds) if args.thresholds else np.array([])

    # Create subsets directory if saves requested
    if args.save_quantiles or args.save_thresholds:
        if snr is not None:
            subsets_dir = image_path.parent / f"subsets_snr_{snr:.3f}"
        else:
            subsets_dir = image_path.parent / "subsets"
        subsets_dir.mkdir(exist_ok=True)

    # Iterate over methods to identify subsets and save if requested
    for method_name, data in results.items():
        # Skip the average or the median if they are included in `results`
        if data["estimator"] in [AVERAGE_NAME, MEDIAN_NAME]:
            continue
        # Get aggregated weights according to estimator type
        weights = canonical_image_weights(data["estimator"], data["weights"])

        # Initialize indices dicts
        idx_good = {"quantile": {}, "fixed_threshold": {}}
        idx_bad = {"quantile": {}, "fixed_threshold": {}}

        # Quantile subsets
        if quantiles.size > 0:
            p_low = np.quantile(weights, quantiles)
            p_high = np.quantile(weights, 1 - quantiles)

            for i, q in enumerate(quantiles):
                # Identify good and bad subset indices for this quantile
                subset_good = weights >= p_high[i]
                subset_bad = weights < p_low[i]

                # Print diagnostic info to terminal
                print(f"\nCalculated images for quantile {q}.")
                print(f"Number of good images: {subset_good.sum()}")
                print(f"Number of bad images:  {subset_bad.sum()}\n")

                # Save subset info to results dict for later processing
                idx_good["quantile"][q] = subset_good
                idx_bad["quantile"][q] = subset_bad

                # Save image subsets to file if requested
                if args.save_quantiles:
                    mrcfile.write(
                        str(subsets_dir / f"{method_name}_{100*q:.0f}pct_best.mrcs"),
                        data=images_save[subset_good],
                        overwrite=False,
                    )
                    mrcfile.write(
                        str(subsets_dir / f"{method_name}_{100*q:.0f}pct_worst.mrcs"),
                        data=images_save[subset_bad],
                        overwrite=False,
                    )

        # Threshold subsets
        for thr in fixed_thresholds:
            # Identify good and bad subset indices for this threshold
            subset_good = weights >= thr
            subset_bad = weights < thr

            # Print diagnostic info to terminal
            print(f"\nCalculated good and bad images for weight threshold {thr}")
            print(f"Good images: weight >= threshold. Bad images: weight < threshold")
            print(f"Number of good images: {subset_good.sum()}")
            print(f"Number of bad images:  {subset_bad.sum()}\n")

            # Save subset info to dict for later processing
            idx_good["fixed_threshold"][thr] = subset_good
            idx_bad["fixed_threshold"][thr] = subset_bad

            # Save good and bad images for this threshold
            if args.save_thresholds:
                mrcfile.write(
                    str(subsets_dir / f"{method_name}_weight_geq_{thr}.mrcs"),
                    data=images_save[subset_good],
                    overwrite=False,
                )
                mrcfile.write(
                    str(subsets_dir / f"{method_name}_weight_lt_{thr}.mrcs"),
                    data=images_save[subset_bad],
                    overwrite=False,
                )

        # Save subset data in results dict
        data["idx_good"] = idx_good
        data["idx_bad"] = idx_bad

        # Save weights to file if requested
        if args.save_weights:
            weights_dir = image_path.parent / "weights"
            weights_dir.mkdir(exist_ok=True)
            np.save(str(weights_dir / f"{method_name}_weights.npy"), weights)
