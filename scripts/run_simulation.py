from pathlib import Path

import numpy as np
import mrcfile
import torch

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.domain.reports import EvaluationReport
from method_comparison.dataset_builder import create_evaluation_dataset
from method_comparison.evaluation.frc import FRCThreshold
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

FRC_THRESHOLDS = [FRCThreshold.ONE_HALF, FRCThreshold.HALF_BIT]
RECALL_METHODS = ["huang_tagare", "inlier_avg", "global_avg"]


def run_experiment(cfg, args, snr) -> EvaluationReport:
    # rng seed for reproducibilty
    seed = cfg.get("seed", None)
    rng = np.random.default_rng(seed=seed)

    # Generate the data
    images, ground_truth, labels = create_evaluation_dataset(
        cfg, rng, snr, standardize_before_noise=args.standardize == "before"
    )

    if args.standardize == "after":
        images = (images - images.mean(axis=(1, 2), keepdims=True)) / images.std(
            axis=(1, 2), keepdims=True
        )
        ground_truth = (ground_truth - ground_truth.mean()) / ground_truth.std()

        # global_image_std = images.std()
        # images = images / (global_image_std + 1.0e-8)
        # ground_truth = ground_truth / (global_image_std + 1.0e-8)

    # Move images to torch
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

    # # Undo standardization
    # if args.standardize:
    #     for _, data in results.items():
    #         data["avg"] = data["avg"] * (global_image_std + 1.0e-8)

    #     images *= global_image_std + 1.0e-8
    #     ground_truth *= global_image_std + 1.0e-8

    # Identify and save requested subsets
    image_path = Path(cfg["data"]["reference_image_path"])
    process_and_save_subsets(
        results, image_path, images_save=images, args=args, snr=snr
    )

    # Calculate complete report with classification and reconstruction metrics
    report = compute_report_labeled(
        results=results,
        images_dict=images_dict,
        ground_truth_img=ground_truth,
        labels=labels,
        reapply_mask=args.reapply_mask,
        mask=mask,
        frc_thresholds=FRC_THRESHOLDS,
        recall_methods=RECALL_METHODS,
        real_agg_strategies=(AggregationStrategy.MEAN,),
        fourier_agg_strategies=(AggregationStrategy.MEAN,),
        energy_reference="ground_truth",
        independent_half_sets=args.independent_half_sets,
    )

    # if args.standardize:
    #     tensor_images *= global_image_std + 1.0e-8

    #     # Re-sync the Fourier dictionaries to the unstandardized scale
    #     fourier_unstandardized = torch.fft.rfft2(tensor_images, norm="ortho")
    #     images_dict[Space.FOURIER_REAL] = fourier_unstandardized.real
    #     images_dict[Space.FOURIER_IMAG] = fourier_unstandardized.imag

    # Print report to terminal
    if args.print:
        print_report(report)

    # Optionally plot the report
    plot_report(
        report,
        max_subplots=args.max_subplots,
        plot_weights="weights" in args.plot,
        density=False,
        plot_frc="frc" in args.plot,
    )

    # Show images (averages and original images) with napari
    if args.show_images:
        visualize_results(results, tensor_images, args, ground_truth, labels)

    return report


def main():
    args = parse_arguments(build_simulation_parser())

    # Load configurations
    cfg = load_config(args.config, args.snr)
    ground_truth_image: np.ndarray = mrcfile.read(cfg["data"]["reference_image_path"])

    reports = dict()

    # Run simulations with every specified snr
    for snr in args.snr:
        print(f"Running experiment with SNR {snr:.3f}")

        snr_report = run_experiment(cfg, args, snr=snr)
        reports[snr] = snr_report

    # Optionally save the report
    if args.report is not None:
        generate_latex_report(
            snr_reports=reports,
            output_path=args.report,
            cfg=cfg,
            ground_truth_image=ground_truth_image,
            args=args
        )


if __name__ == "__main__":
    main()
