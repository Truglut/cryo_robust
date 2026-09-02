from argparse import Namespace

import torch
import numpy as np
import napari

from cryo_robust.comparison.dataset_builder import LABEL_TYPES

from cryo_robust.comparison.domain.runs import AVERAGE_NAME, MethodRun


def visualize_results(
    results: dict[str, MethodRun],
    tensor_images: torch.Tensor,
    args: Namespace,
    ground_truth: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    subset_averages: bool = False,
) -> None:
    """Sets up and launches the napari viewer with all of the images"""
    # Initialize viewer
    viewer = napari.Viewer()

    # Show original image if there is one
    if ground_truth is not None:
        viewer.add_image(ground_truth, name="True image", visible=False)

    # Show regular average
    viewer.add_image(
        tensor_images.mean(dim=0).detach().cpu().numpy(),
        name="Average of all images",
        visible=True,
    )

    # Show average of good images if labels have been provided
    if labels is not None:
        good_images_mask = torch.from_numpy(labels == 0).to(tensor_images.device)
        viewer.add_image(
            tensor_images[good_images_mask].mean(dim=0).detach().cpu().numpy(),
            name="Average of only good images",
            visible=False,
        )

    # Iterate over methods to show: estimated average, reference and quantile subsets
    for method_name, run in results.items():
        # Skip the average
        if method_name == AVERAGE_NAME:
            continue

        # Show the estimated representative
        avg_np = run.result.average.detach().cpu().numpy()
        viewer.add_image(avg_np, name=f"Estimation with {method_name}", visible=False)

        # Show the reference if needed
        reference = run.initial_reference
        if reference is not None:
            ref_np = reference.detach().cpu().numpy()
            viewer.add_image(ref_np, name=f"{method_name}: reference", visible=False)

        # Show quantile and threshold-defined subset averages if requested
        if subset_averages:
            show_quantile_and_threshold_averages(
                tensor_images, args, viewer, method_name
            )

    # Adjust contrast limits
    global_min = float(min(layer.data.min() for layer in viewer.layers))
    global_max = float(max(layer.data.max() for layer in viewer.layers))
    viewer.layers.link_layers(viewer.layers[1:], attributes=["contrast_limits"])
    viewer.layers[0].contrast_limits = (global_min, global_max)

    # Show examples of all image types (good, very rotated, misclassified)
    if labels is not None:
        for label in LABEL_TYPES:
            max_show = 25
            n_show = min(max_show, (labels == label).sum())
            if n_show:
                labels_mask = torch.from_numpy(labels == label).to(tensor_images.device)
                viewer.add_image(
                    tensor_images[labels_mask][:n_show].detach().cpu().numpy(),
                    name=f"{n_show} first {LABEL_TYPES[label]}",
                    visible=False,
                )
    else:
        # Add random sample of the images
        max_show = 50
        n_show = min(max_show, tensor_images.shape[0])
        idx_show = torch.from_numpy(
            np.random.choice(tensor_images.shape[0], size=n_show, replace=False)
        ).to(tensor_images.device)
        viewer.add_image(
            tensor_images[idx_show].detach().cpu().numpy(),
            name=f"{n_show} random images",
            visible=False,
        )

    # Run the viewer
    napari.run()


def show_quantile_and_threshold_averages(
    tensor_images: torch.Tensor,
    args: Namespace,
    viewer: napari.Viewer,
    method_name: str,
    quantile_indices_good: dict[float, np.ndarray],
    quantile_indices_bad: dict[float, np.ndarray],
    threshold_indices_good: dict[float, np.ndarray],
    threshold_indices_bad: dict[float, np.ndarray],
):
    for q in args.quantiles or []:
        mask_good = torch.from_numpy(quantile_indices_good[q]).to(tensor_images.device)
        mask_bad = torch.from_numpy(quantile_indices_bad[q]).to(tensor_images.device)

        if mask_good.sum() > 0:
            good_mean = tensor_images[mask_good].mean(dim=0).detach().cpu().numpy()
            viewer.add_image(
                good_mean,
                name=f"{100*q}% best ({method_name})",
                visible=False,
            )
        if mask_bad.sum() > 0:
            bad_mean = tensor_images[mask_bad].mean(dim=0).detach().cpu().numpy()
            viewer.add_image(
                bad_mean,
                name=f"{100*q}% worst ({method_name})",
                visible=False,
            )

    # Show threshold-defined subsets if requested
    for thr in args.thresholds or []:
        mask_good = torch.from_numpy(threshold_indices_good[thr]).to(
            tensor_images.device
        )
        mask_bad = torch.from_numpy(threshold_indices_bad[thr]).to(tensor_images.device)

        if mask_good.sum() > 0:
            good_mean = tensor_images[mask_good].mean(dim=0).detach().cpu().numpy()
            viewer.add_image(
                good_mean,
                name=f"Average of good (weight >= {thr}) images (method: {method_name}).",
                visible=False,
            )

        if mask_bad.sum() > 0:
            bad_mean = tensor_images[mask_bad].mean(dim=0).detach().cpu().numpy()
            viewer.add_image(
                bad_mean,
                name=f"Average of bad (weight < {thr}) images (method: {method_name}).",
                visible=False,
            )
