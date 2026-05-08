import yaml
import argparse
from pathlib import Path

import numpy as np
import mrcfile
import torch

from estimators import build_estimator
from estimators.base import Estimator
from estimators.admm import ADMMSolver
from estimators.irls import IRLSFourier
from estimators.gmm import GMMEstimator, RecursiveGMMEstimator

from method_comparison.evaluator import aggregate_weights

from utils.masks import create_circular_mask
from utils.space import Space

AVERAGE_NAME = "Average"


def load_config(config_path: str, snr: float | None = None):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
        if snr is not None:
            cfg["noise"]["snr"] = snr
        return cfg
    

def build_base_parser():
    """Parses the config from the command line"""
    parser = argparse.ArgumentParser(description="Run robust estimators on real data")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device for PyTorch"
    )
    parser.add_argument(
        "--view_images",
        default=False,
        action="store_true",
        help="If True, show generated images",
    )
    parser.add_argument(
        "--gmm_evaluation",
        default=False,
        action="store_true",
        help="If True, show a general overview of gmm models",
    )
    parser.add_argument(
        "--plot_weights",
        default=False,
        action="store_true",
        help="If True, show plots of image weights",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        help="Quantiles for which to show (and optionally save with --save_quantiles) best and worst images",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        help="Weight thresholds for which to show good and bad images",
    )
    parser.add_argument(
        "--save_quantiles",
        default=False,
        action="store_true",
        help="If True, save images with highest and lowest weights for each quantile in --quantiles",
    )
    parser.add_argument(
        "--save_thresholds",
        default=False,
        action="store_true",
        help="If True, save images with weights higher and lower than each given threshold",
    )
    parser.add_argument(
        "--save_original",
        default=False,
        action="store_true",
        help="If True, any images saved will be the original, unaligned images",
    )
    parser.add_argument(
        "--save_weights",
        default=False,
        action="store_true",
        help="If True, final weights for every estimation method will be saved as a .npy file",
    )
    return parser


def apply_mask(images_tensor: torch.Tensor, mask_radius: float, inplace: bool = False):
    """Applies a circular mask to a batch of images, optionally modifying the input tensor in-place"""
    # Create mask on device
    image_shape = tuple(images_tensor.shape[1:])
    mask_np = create_circular_mask(image_shape, mask_radius)
    mask_tensor = torch.from_numpy(mask_np).to(device=images_tensor.device)

    masked_images = images_tensor if inplace else images_tensor.clone()

    # Apply mask to images
    masked_images *= mask_tensor

    return masked_images, mask_tensor


def run_estimators(
    cfg: dict, images_dict: dict[Space, torch.Tensor], args, add_avg: bool = False
) -> dict:
    device = args.device

    results = {}
    for method_cfg in cfg["experiment"]["methods"]:
        method_name = method_cfg["name"]
        print(f"Running {method_name}...")

        estimator = build_estimator(method_cfg, images_dict, device=device)

        # Handle optional reference
        reference = None
        if method_cfg.get("initial_reference"):
            ref_np = mrcfile.read(method_cfg["initial_reference"])
            reference = torch.tensor(ref_np, dtype=torch.float32, device=device)

        if isinstance(estimator, (GMMEstimator, RecursiveGMMEstimator)):
            estimator.fit(
                images_dict,
                reference=reference,
                plot_fits=args.gmm_evaluation,
                plot_title=method_name,
            )
        else:
            estimator.fit(images_dict, reference=reference)

        results[method_name] = {
            "estimator": estimator,
            "reference": reference,
            "avg": estimator.avg,
            "weights": estimator.final_weights,
        }

    if add_avg:
        results[AVERAGE_NAME] = {
            "avg": images_dict[Space.REAL].mean(dim=0),
            "weights": {
                space: torch.ones(
                    size=(images_dict[space].shape[0], 1, 1),
                    dtype=torch.float32,
                    device=args.device,
                )
                for space in Space
            },
            "reference": None,
            "estimator": None,
        }
    return results


def get_weights(estimator: Estimator, final_weights: dict[Space, torch.Tensor]):
    if isinstance(estimator, ADMMSolver):
        weights = final_weights[Space.REAL]
    elif isinstance(estimator, IRLSFourier):
        weights = 0.5 * (
            final_weights[Space.FOURIER_REAL] + final_weights[Space.FOURIER_IMAG]
        )
    else:
        weights = final_weights[estimator.space]

    return aggregate_weights(weights, "mean")


def process_and_save_subsets(
    results: dict, image_path: Path, images_save: np.ndarray, args
) -> None:
    # Initialize quantiles and thresholds arrays from args
    quantiles = np.array(args.quantiles) if args.quantiles else np.array([])
    fixed_thresholds = np.array(args.thresholds) if args.thresholds else np.array([])

    # Create subsets directory if saves requested
    if args.save_quantiles or args.save_thresholds:
        subsets_dir = image_path.parent / "subsets"
        subsets_dir.mkdir(exist_ok=True)

    # Iterate over methods to identify subsets and save if requested
    for method_name, data in results.items():
        # Skip the average if it is included in `results`
        if data["estimator"] is None:
            continue
        # Get aggregated weights according to estimator type
        weights = get_weights(data["estimator"], data["weights"])

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
