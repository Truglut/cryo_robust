from pathlib import Path
from argparse import Namespace

import numpy as np
import mrcfile
import torch

from estimators.data import ImageBatch

from method_comparison.domain.enums import ImageSpace, AggregationStrategy
from method_comparison.domain.reports import EvaluationReport, EvaluationStudy
from method_comparison.dataset_builder import create_evaluation_dataset
from method_comparison.evaluation.frc import FRCThreshold
from method_comparison.evaluation.report_building import compute_report
from method_comparison.visualization.printing import print_report
from method_comparison.visualization.plotting import plot_report
from method_comparison.latex import generate_latex_report

from scripts.cli import build_simulation_parser, parse_arguments
from scripts.common import (
    load_config,
    apply_mask,
    run_estimators,
    process_and_save_subsets,
)
from scripts.napari_visualization import visualize_results

from utils.masks import create_fourier_mask

FRC_THRESHOLDS = [FRCThreshold.ONE_HALF, FRCThreshold.HALF_BIT]
RECALL_METHODS = ["huang_tagare", "inlier_avg", "global_avg"]


def run_experiment(
    cfg: dict, args: Namespace, snr: float, rng: np.random.Generator
) -> EvaluationReport:
    """
    Runs one instance of the simulated experiments.
    Generates the image set, the runs all of the estimation methods specified through
    the ``cfg`` dict and returns a report with the results.
    """
    # Generate the data
    images, ground_truth, labels = create_evaluation_dataset(
        cfg=cfg,
        rng=rng,
        snr=snr,
        standardize_before_noise=args.standardize in ["before", "both"],
        per_image_noise_std=args.per_image_noise_std,
    )

    if args.standardize in ["after", "both"]:
        images = (images - images.mean(axis=(1, 2), keepdims=True)) / images.std(
            axis=(1, 2), keepdims=True
        )
        ground_truth = (ground_truth - ground_truth.mean()) / ground_truth.std()

    # Move images to torch
    tensor_images = torch.from_numpy(images).to(dtype=torch.float32, device=args.device)

    # Apply mask to images
    mask_radius = cfg["mask"]["params"]["radius"]
    tensor_images, mask_tensor = apply_mask(tensor_images, mask_radius, inplace=True)
    mask = mask_tensor.detach().cpu().numpy()
    ground_truth *= mask

    # Prepare image dict for estimation models
    image_batch = ImageBatch.from_real(tensor_images)

    # Run the Estimation Methods
    results = run_estimators(cfg, image_batch, args, add_avg=True, add_median=False)

    # Identify and save requested subsets
    image_path = Path(cfg["data"]["reference_image_path"])
    process_and_save_subsets(
        results, image_path, images_save=images, args=args, snr=snr
    )

    # Calculate fourier weight mask
    fourier_weight_mask = None
    if args.fourier_weight_mask != "none":
        fourier_weight_mask = create_fourier_mask(
            image_shape=ground_truth.shape, mask_type=args.fourier_weight_mask
        )
    weights_masks_dict = {
        ImageSpace.REAL: mask,
        ImageSpace.FOURIER_REAL: fourier_weight_mask,
        ImageSpace.FOURIER_IMAG: fourier_weight_mask,
    }

    # Calculate complete report with classification and reconstruction metrics
    report = compute_report(
        results=results,
        image_batch=image_batch,
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
        masks_dict=weights_masks_dict,
    )

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
    """
    Runs all of the requested simulated experiments.
    For each SNR level, repeats the same experiment the request number of times
    (through the ``n-runs`` command-line argument) and stores all of the results.
    """
    args = parse_arguments(build_simulation_parser())

    # Load configurations
    cfg = load_config(args.config, args.snr)
    ground_truth_image: np.ndarray = mrcfile.read(cfg["data"]["reference_image_path"])

    # rng seed for reproducibilty
    seed = cfg.get("seed", None)
    rng = np.random.default_rng(seed=seed)

    snr_results = dict()

    # Run simulations with every specified snr
    for snr in args.snr:
        if args.n_runs <= 1:
            print(f"Running experiment with SNR {snr:.3f}")

            snr_report = run_experiment(cfg, args, snr=snr, rng=rng)
            snr_results[snr] = snr_report
        else:
            reports_list = []

            for i in range(args.n_runs):
                print(f"Running experiment {i + 1} with SNR {snr:.3f}")

                reports_list.append(run_experiment(cfg, args, snr=snr, rng=rng))
            snr_results[snr] = EvaluationStudy(reports_list)

    # Optionally save the report
    if args.report is not None:
        generate_latex_report(
            results=snr_results,
            output_path=args.report,
            cfg=cfg,
            ground_truth_image=ground_truth_image,
            args=args,
        )


if __name__ == "__main__":
    main()
