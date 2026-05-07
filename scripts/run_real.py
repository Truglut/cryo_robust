import argparse
from pathlib import Path

import torch
import mrcfile

from method_comparison.evaluator import report_unlabeled

from utils.space import Space

from scripts.common import (
    load_config,
    apply_mask,
    run_estimators,
    process_and_save_subsets,
)
from scripts.napari_visualization import visualize_results


def parse_arguments():
    """Parses the config from the command line"""
    parser = argparse.ArgumentParser(description="Run robust estimators on real data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
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
    return parser.parse_args()


def load_and_preprocess(cfg: dict, args) -> tuple:
    """Loads images, applies masks on the target device"""
    data_cfg = cfg["data"]
    image_path = Path(data_cfg["image_path"])

    # Read aligned images
    aligned_images_np = mrcfile.read(str(image_path))

    # Determine which images to save later
    images_save = aligned_images_np
    if args.save_original:
        orig_path = data_cfg.get("original_particles_path")
        if orig_path is None:
            raise Exception(
                "Requested to save original images, but config file does not contain path to original images"
            )
        if not Path(orig_path).exists():
            raise Exception("Original images were not found in the config file path")
        images_save = mrcfile.read(orig_path)

    # Move images to device
    tensor_images = torch.from_numpy(aligned_images_np).to(
        dtype=torch.float32, device=args.device
    )

    return tensor_images, images_save, image_path


def main():
    args = parse_arguments()

    # Load configurations
    cfg = load_config(args.config, None)

    # Read images from file path
    tensor_images, images_save, image_path = load_and_preprocess(cfg, args)

    # Apply mask to images
    mask_radius = cfg["mask"]["params"]["radius"]
    tensor_images, _ = apply_mask(tensor_images, mask_radius, inplace=True)

    # Compute fourier transform and save images in dict
    fourier_images = torch.fft.rfft2(tensor_images, norm="ortho")
    images_dict = {
        Space.REAL: tensor_images,
        Space.FOURIER_REAL: fourier_images.real,
        Space.FOURIER_IMAG: fourier_images.imag,
    }

    # Run Estimators
    results = run_estimators(cfg, images_dict, args)

    # Show report of results (currently just plots weight distributions)
    report_unlabeled(results, args.plot_weights)
    process_and_save_subsets(
        results, image_path=image_path, images_save=images_save, args=args
    )

    # Show images (averages and original images) with napari
    if args.view_images:
        visualize_results(results, tensor_images, args)


if __name__ == "__main__":
    main()
