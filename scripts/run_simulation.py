import argparse
import yaml
import numpy as np
import torch
import mrcfile
import napari

from estimators import build_estimator
from estimators.gmm import GMMEstimator, RecursiveGMMEstimator
from method_comparison.dataset_builder import create_evaluation_dataset
from method_comparison.evaluator import compare_and_report

from utils.masks import create_circular_mask
from utils.space import Space

LABEL_TYPES = {
    0: "generated copies of reference",
    1: "very rotated copies of reference",
    2: "misclassified outliers",
}


def load_config(config_path: str, snr: float | None = None):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
        if snr:
            cfg["noise"]["snr"] = snr
        return cfg


def main():
    # Parse the config file path from the command line
    parser = argparse.ArgumentParser(description="Robust Estimation Comparator")
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
    args = parser.parse_args()

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
    # labels according to LABEL_TYPES

    # Create mask for images
    image_shape = images.shape[1:]
    mask = create_circular_mask(image_shape, cfg["mask"]["params"]["radius"])

    # Apply mask to images
    images = mask * images
    ground_truth = ground_truth * mask

    # Prepare images (real space and fourier) on pytorch
    tensor_images = torch.from_numpy(images).to(dtype=torch.float32, device=args.device)
    fourier_images = torch.fft.rfft2(tensor_images, norm="ortho")
    images_dict = {
        Space.REAL: tensor_images,
        Space.FOURIER_REAL: fourier_images.real,
        Space.FOURIER_IMAG: fourier_images.imag,
    }

    # Run the Estimation Methods
    results = {}
    estimators = {}
    for method_cfg in cfg["experiment"]["methods"]:
        method_name = method_cfg["name"]
        print(f"Running {method_name} on {args.device.upper()}...")

        # 1. Build the estimator, passing the command-line device
        estimator = build_estimator(method_cfg, images_dict, device=args.device)
        estimators[method_name] = estimator

        # Get initial reference for estimator
        if method_cfg.get("use_reference", False):
            reference = torch.tensor(
                mrcfile.read(method_cfg["use_reference"]),
                dtype=torch.float32,
                device=args.device,
            )
        else:
            reference = None

        # Run estimator on images
        if isinstance(estimator, GMMEstimator) or isinstance(
            estimator, RecursiveGMMEstimator
        ):
            estimator.fit(
                tensor_images,
                reference=reference,
                plot_fits=args.gmm_evaluation,
                plot_title=method_name,
            )
        else:
            estimator.fit(tensor_images)

        # 3. Store results (final weights and estimated average)
        results[method_name] = {
            "avg": estimator.avg,
            "weights": estimator.final_weights,
            "reference": reference,
            "estimator": estimator,
        }

    results["Average"] = {
        "avg": tensor_images.mean(dim=0),
        "weights": {
            space: torch.ones(
                size=(images.shape[0], 1, 1), dtype=torch.float32, device=args.device
            )
            for space in Space
        },
        "reference": None,
        "estimator": None,
    }

    compare_and_report(
        results=results,
        images_dict=images_dict,  # Pass images to allow baseline fits / global avg
        ground_truth_img=ground_truth,
        labels=labels,
        plot_weights=args.plot_weights,
        max_subplots=4,
        real_agg_strategies=["mean"],
        fourier_agg_strategies=["energy"],
        energy_reference="ground_truth",  # Or "global_avg"
        fsc_threshold=0.143,
        mask=mask,
        reapply_mask=args.reapply_mask,
    )

    # Show images (averages and original images) with napari
    if args.view_images:
        viewer = napari.Viewer()

        # Show original image
        viewer.add_image(ground_truth, name=f"True image", visible=True)

        # Show regular average
        viewer.add_image(
            images.mean(axis=0),
            name=f"Average of all images (equal weights)",
            visible=False,
        )

        # Show average of good images
        viewer.add_image(
            images[labels == 0].mean(axis=0),
            name="Average of only good images",
            visible=False,
        )

        # Show estimated average with every method
        for method_cfg in cfg["experiment"]["methods"]:
            method_name = method_cfg["name"]
            viewer.add_image(
                results[method_name]["avg"],
                name=f"Estimation with {method_name}",
                visible=False,
            )

        ## Adjust contrast limits to better compare images
        # Sweep through the data of all added layers to find the absolute extremes
        global_min = float(min(layer.data.min() for layer in viewer.layers))
        global_max = float(max(layer.data.max() for layer in viewer.layers))

        # Link and apply
        viewer.layers.link_layers(viewer.layers, attributes=["contrast_limits"])
        viewer.layers[0].contrast_limits = (global_min, global_max)

        # Show examples of all image types (good, very rotated, misclassified)
        for label in LABEL_TYPES:
            max_show = 25
            n_show = min(max_show, (labels == label).sum())
            if n_show:
                viewer.add_image(
                    images[labels == label][:n_show],
                    name=f"{n_show} first {LABEL_TYPES[label]}",
                    visible=False,
                )

        # Run the viewer with napari
        napari.run()


if __name__ == "__main__":
    main()
