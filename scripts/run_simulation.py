from pathlib import Path

import numpy as np
import torch

from method_comparison.domain.enums import Space
from method_comparison.dataset_builder import create_evaluation_dataset
from method_comparison.evaluation import (
    print_report,
    plot_report,
)
from method_comparison.evaluation.report_building import compute_report_labeled

from scripts.common import (
    load_config,
    apply_mask,
    run_estimators,
    process_and_save_subsets,
    build_base_parser,
)
from scripts.napari_visualization import visualize_results

FSC_THRESHOLD = 0.143
RECALL_METHODS = ["huang_tagare", "inlier_avg", "global_avg"]


def parse_arguments():
    # Parse the config file path from the command line
    parser = build_base_parser()
    parser.add_argument(
        "--snr",
        default=None,
        type=float,
        help="Target signal to noise ratio in image generation. Overrides snr in config file",
    )
    parser.add_argument(
        "--normalize",
        default=False,
        action="store_true",
        help="If True, images will be normalized to [0,1] before adding noise/rotating",
    )
    parser.add_argument(
        "--reapply_mask",
        default=False,
        action="store_true",
        help="If True, the mask will be reapplied to the estimations from every method",
    )
    parser.add_argument(
        "--plot_fsc",
        default=False,
        action="store_true",
        help="If True, plot FSC resolution for all methods (overlayed on one figure)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load configurations
    cfg = load_config(args.config, args.snr)

    # rng seed for reproducibility
    seed = cfg.get("seed", None)
    rng = np.random.default_rng(seed=seed)

    # Generate the data (good copies, rotated outliers, misclassified outliers + noise)
    print("Generating data...")
    images, ground_truth, labels = create_evaluation_dataset(
        cfg, rng, normalize=args.normalize
    )
    tensor_images = torch.from_numpy(images).to(dtype=torch.float32, device=args.device)

    # Apply mask to images
    mask_radius = cfg["mask"]["params"]["radius"]
    tensor_images, mask_tensor = apply_mask(tensor_images, mask_radius, inplace=True)
    mask = mask_tensor.detach().cpu().numpy()
    ground_truth *= mask

    # Prepare image dict for estimation models
    fourier_images = torch.fft.rfft2(tensor_images, norm="ortho")
    images_dict = {
        Space.REAL: tensor_images,
        Space.FOURIER_REAL: fourier_images.real,
        Space.FOURIER_IMAG: fourier_images.imag,
    }
    del fourier_images

    # Run the Estimation Methods
    results = run_estimators(cfg, images_dict, args, add_avg=True)

    # Identify and save requested subsets
    image_path = Path(cfg["data"]["reference_image_path"])
    process_and_save_subsets(results, image_path, images_save=images, args=args)

    report = compute_report_labeled(
        results=results,
        images_dict=images_dict,
        ground_truth_img=ground_truth,
        labels=labels,
        reapply_mask=args.reapply_mask,
        mask=mask,
        fsc_threshold=FSC_THRESHOLD,
        recall_methods=RECALL_METHODS,
        real_agg_strategies=("mean",),
        fourier_agg_strategies=("energy",),
        energy_reference="ground_truth",
    )
    print_report(report)
    plot_report(
        report,
        max_subplots=True,
        plot_weights=args.plot_weights,
        density=False,
        plot_fsc=args.plot_fsc,
    )

    # Show images (averages and original images) with napari
    if args.view_images:
        visualize_results(results, tensor_images, args, ground_truth, labels)


if __name__ == "__main__":
    main()
