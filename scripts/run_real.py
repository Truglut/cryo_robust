from pathlib import Path

import torch
import mrcfile

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.evaluation.report_building import compute_report_labeled
from method_comparison.visualization.printing import print_report
from method_comparison.visualization.plotting import plot_report

from scripts.cli import build_experimental_parser, parse_arguments
from scripts.common import (
    load_config,
    apply_mask,
    run_estimators,
    process_and_save_subsets,
)
from scripts.napari_visualization import visualize_results


def load_and_preprocess(cfg: dict, args) -> tuple:
    """Loads images on the target device"""
    data_cfg = cfg["data"]
    image_path = Path(data_cfg["image_path"])

    # Read aligned images
    aligned_images_np = mrcfile.read(str(image_path))

    # Determine which images to save later
    images_save = aligned_images_np
    if args.save_unaligned:
        orig_path = data_cfg.get("unaligned_particles_path")
        if orig_path is None:
            raise Exception(
                "Requested to save unaligned images, but config file does not contain path to unaligned images"
            )
        if not Path(orig_path).exists():
            raise Exception("Unaligned images were not found in the config file path")
        images_save = mrcfile.read(orig_path)

    # Move images to device
    tensor_images = torch.from_numpy(aligned_images_np).to(
        dtype=torch.float32, device=args.device
    )

    if args.standardize:
        global_image_std = tensor_images.std()
        tensor_images /= global_image_std

    return tensor_images, images_save, image_path


def main():
    args = parse_arguments(build_experimental_parser())

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
    process_and_save_subsets(
        results, image_path=image_path, images_save=images_save, args=args
    )

    # Show report of results (currently just plots weight distributions)
    report = compute_report_labeled(
        results,
        images_dict,
        real_agg_strategies=(AggregationStrategy.MEAN,),
        fourier_agg_strategies=(AggregationStrategy.MEAN,),
        energy_reference="global_avg",
    )
    plot_report(
        report,
        plot_weights="weights" in args.plot,
        density=False,
        plot_frc="fsc" in args.plot,
        max_subplots=args.max_subplots,
    )

    # Show images (averages and original images) with napari
    if args.show_images:
        visualize_results(results, tensor_images, args)


if __name__ == "__main__":
    main()
