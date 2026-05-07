from typing import Any

import torch
import numpy as np
import napari

from method_comparison.dataset_builder import LABEL_TYPES
from scripts.common import AVERAGE_NAME


def visualize_results(
    results: dict[str, Any],
    tensor_images: torch.Tensor,
    args,
    ground_truth: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> None:
    """Sets up and launches the napari viewer"""
    # Initialize viewer
    viewer = napari.Viewer()

    # Move images to cpu for napari
    masked_images_np = tensor_images.detach().cpu().numpy()

    # Show original image if there is one
    if ground_truth is not None:
        viewer.add_image(ground_truth, name="True image", visible=False)

    # Show regular average
    viewer.add(
        masked_images_np.mean(axis=0), name="Average of all images", visible=True
    )

    # Show average of good images if labels have been provided
    if labels is not None:
        viewer.add_image(
            masked_images_np[labels == 0].mean(axis=0),
            name="Average of only good images",
            visible=False,
        )
    
    # Iterate over methods to show: estimated average, reference and quantile subsets
    for method_name, data in results:
        # Skip the average
        if method_name == AVERAGE_NAME:
            continue

        # Show the estimated representative
        avg_np = data["avg"].detach().cpu().numpy()
        viewer.add_image(avg_np, name=f"Estimation with {method_name}", visible=False)

        # Show the reference if needed
        if data["reference"] is not None:
            ref_np = (
                data["reference"].detach().cpu().numpy()
                if isinstance(data["reference"], torch.Tensor)
                else data["reference"]
            )
            viewer.add_image(ref_np, name=f"{method_name}: reference", visible=False)

        # Show quantile-defined subsets if requested
        for q in args.quantiles:
            good_imgs = masked_images_np[data["idx_good"]["quantile"][q]]
            bad_imgs = masked_images_np[data["idx_bad"]["quantile"][q]]
            viewer.add_image(
                good_imgs.mean(axis=0),
                name=f"{100*q}% best ({method_name})",
                visible=False,
            )
            viewer.add_image(
                bad_imgs.mean(axis=0),
                name=f"{100*q}% worst ({method_name})",
                visible=False,
            )
        
        # Show threshold-defined subsets if requested
        for thr in args.thresholds:
            good_images = masked_images_np[data["idx_good"]["fixed_threshold"][thr]]
            bad_images = masked_images_np[data["idx_bad"]["fixed_threshold"][thr]]

            viewer.add_image(
                good_images.mean(axis=0),
                name=f"Average of good (weight >= {thr}) images (method: {method_name}).",
                visible=False,
            )

            viewer.add_image(
                bad_images.mean(axis=0),
                name=f"Average of bad (weight < {thr}) images (method: {method_name}).",
                visible=False,
            )
    
    # Adjust contrast limits
    global_min = float(min(layer.data.min() for layer in viewer.layers))
    global_max = float(max(layer.data.max() for layer in viewer.layers))
    viewer.layers.link_layers(viewer.layers, attributes=["contrast_limits"])
    viewer.layers[0].contrast_limits = (global_min, global_max)

    # Add random sample of the images
    n_show = min(50, masked_images_np.shape[0])
    idx_show = np.random.choice(masked_images_np.shape[0], size=n_show, replace=False)
    viewer.add_image(
        masked_images_np[idx_show], name=f"{n_show} random images", visible=False
    )

    # Show examples of all image types (good, very rotated, misclassified)
    if labels is not None:
        for label in LABEL_TYPES:
            max_show = 25
            n_show = min(max_show, (labels == label).sum())
            if n_show:
                viewer.add_image(
                    masked_images_np[labels == label][:n_show],
                    name=f"{n_show} first {LABEL_TYPES[label]}",
                    visible=False,
                )

    # Run the viewer
    napari.run()
