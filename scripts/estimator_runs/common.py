import yaml
from pathlib import Path
from argparse import Namespace

import numpy as np
import mrcfile
import torch

from cryo_robust.estimators.results import WeightSet

from cryo_robust.comparison.domain.enums import AggregationStrategy
from cryo_robust.comparison.domain.runs import MethodRun

from cryo_robust.comparison.evaluation.aggregation import aggregate_weights
from cryo_robust.comparison.visualization.plotting import AVERAGE_NAME, MEDIAN_NAME

from cryo_robust.utils.masks import create_circular_mask


def load_config(
    config_path: str | Path,
    snr: float | None = None,
    reference_image_path: Path | None = None,
    misclassified_path: Path | None = None,
):
    """
    Loads the config file and overrides the config SNR specification if the ``snr``
    argument is not ``None``.
    """
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    if snr is not None:
        cfg.setdefault("noise", {})["snr"] = snr
    cfg.setdefault("data", {})
    if reference_image_path is not None:
        cfg["data"]["reference_image_path"] = reference_image_path
    if misclassified_path is not None:
        cfg["data"]["misclassified_path"] = misclassified_path
    return cfg


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


def canonical_image_weights(weights: WeightSet) -> np.ndarray:
    """
    Return one scalar weight per image from an estimator WeightSet.

    Parameters
    ----------
    weights : WeightSet
        Estimator weights.

    Returns
    -------
    np.ndarray
        One scalar score per image.

    Raises
    ------
    ValueError
        If the estimator did not produce any usable weights.
    """
    canonical = weights.canonical_weights()

    if canonical is None:
        raise ValueError("No canonical image weights are available.")

    return aggregate_weights(
        canonical,
        AggregationStrategy.MEAN,
    )


def process_and_save_subsets(
    results: dict[str, MethodRun],
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
    if not args.quantiles and not args.thresholds and not args.save_weights:
        return

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
    for method_name, run in results.items():
        # Skip the average or the median if they are included in `results`
        if (
            method_name in [AVERAGE_NAME, MEDIAN_NAME]
            or run.result.weights.canonical_weights() is None
        ):
            continue

        weights = canonical_image_weights(run.result.weights)

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

        # Save weights to file if requested
        if args.save_weights:
            weights_dir = image_path.parent / "weights"
            weights_dir.mkdir(exist_ok=True)
            np.save(str(weights_dir / f"{method_name}_weights.npy"), weights)
