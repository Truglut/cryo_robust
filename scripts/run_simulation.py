from pathlib import Path

import numpy as np
import torch

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.dataset_builder import create_evaluation_dataset
from method_comparison.evaluation.report_building import compute_report_labeled
from method_comparison.visualization.printing import print_report
from method_comparison.visualization.plotting import plot_report
from method_comparison.visualization.latex import generate_latex_report

from scripts.cli import build_simulation_parser, parse_arguments
from scripts.common import (
    load_config,
    apply_mask,
    run_estimators,
    process_and_save_subsets,
)
from scripts.napari_visualization import visualize_results

FSC_THRESHOLD = 0.143
RECALL_METHODS = ["huang_tagare", "inlier_avg", "global_avg"]


def main():
    args = parse_arguments(build_simulation_parser())

    # Load configurations
    cfg = load_config(args.config, args.snr)

    # rng seed for reproducibility
    seed = cfg.get("seed", None)
    rng = np.random.default_rng(seed=seed)

    # Generate the data (good copies, rotated outliers, misclassified outliers + noise)
    print("Generating data...")
    images, ground_truth, labels = create_evaluation_dataset(
        cfg, rng, standardize=args.standardize
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

    # Calculate complete report with classification and reconstruction metrics
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

    # Print report to terminal
    print_report(report)

    # Optionally plot the report
    plot_report(
        report,
        max_subplots=True,
        plot_weights="weights" in args.plot,
        density=False,
        plot_fsc="fsc" in args.plot,
    )

    # Optionally save the report
    if args.report is not None:
        generate_latex_report(report, output_path=args.report)

    # Show images (averages and original images) with napari
    if args.show_images:
        visualize_results(results, tensor_images, args, ground_truth, labels)


if __name__ == "__main__":
    main()
