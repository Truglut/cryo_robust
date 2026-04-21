import argparse
import yaml
import numpy as np
import torch
import napari
import mrcfile
from estimators import build_estimator
from estimators.admm import ADMMSolver
from estimators.gmm import GMMEstimator, RecursiveGMMEstimator
from method_comparison.evaluator import report_unlabeled, aggregate_weights
from method_comparison.gmm_evaluation import evaluate_gmm_fits_unlabeled
from utils.masks import create_circular_mask
from utils.space import Space
import matplotlib.pyplot as plt


def load_config(config_path: str, snr: float | None = None):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    # Parse the config file path from the command line
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
        "--show_good_images",
        default=False,
        action="store_true",
        help="If True, show sample of good images and bad images",
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
    args = parser.parse_args()

    # Load configurations
    cfg = load_config(args.config, args.snr)

    unaligned = mrcfile.read(
        "data/particles/experimental_with_outliers/original_particles.mrcs"
    )
    # Read images from file path
    original_images = mrcfile.read(cfg["data"]["image_path"])

    # Create mask and apply to images
    image_shape = original_images.shape[1:]
    mask = create_circular_mask(image_shape, cfg["mask"]["params"]["radius"])
    masked_images = mask * original_images

    # Convert to tensor for the models
    tensor_images = torch.from_numpy(masked_images).to(
        dtype=torch.float32, device=args.device
    )
    fourier_images = torch.fft.rfft2(tensor_images, norm="ortho")
    images_dict = {
        Space.REAL: tensor_images,
        Space.FOURIER_REAL: fourier_images.real,
        Space.FOURIER_IMAG: fourier_images.imag,
    }

    # Run Estimators
    estimators = {}
    results = {}
    estimator_names = []

    for method_cfg in cfg["experiment"]["methods"]:
        method_name = method_cfg["name"]
        estimator_names.append(method_name)
        print(f"Running {method_name}...")

        # Build estimator from params
        estimator = build_estimator(method_cfg, images_dict, device=args.device)

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

        # Save results
        estimators[method_name] = estimator
        results[method_name] = {
            "reference": reference,  # .cpu().numpy() if reference is not None else None,
            "avg": estimator.avg,  # .detach().cpu().numpy(),
            "weights": estimator.final_weights,
        }

    # Show report of results (currently just plots weight distributions)
    report_unlabeled(results)

    # Identify x% of images with lowest and highest weights for every estimator
    quantiles = np.array([])
    fixed_thresholds = np.array([0.50])
    for method_name, estimator in estimators.items():
        if isinstance(estimator, ADMMSolver):
            weights = estimator.final_weights[Space.REAL]
        else:
            weights = estimator.final_weights[estimator.space]

        weights = aggregate_weights(weights, "mean")

        p_low = np.quantile(weights, quantiles)
        p_high = np.quantile(weights, 1 - quantiles)

        idx_good = {"quantile": {}, "fixed_threshold": {}}
        idx_bad = {"quantile": {}, "fixed_threshold": {}}
        for i, q in enumerate(quantiles):
            idx_bad["quantile"][q] = weights < p_low[i]
            idx_good["quantile"][q] = weights >= p_high[i]

        for thr in fixed_thresholds:
            idx_bad["fixed_threshold"][thr] = weights < thr
            idx_good["fixed_threshold"][thr] = weights >= thr

        results[method_name]["idx_good"] = idx_good
        results[method_name]["idx_bad"] = idx_bad

    # Show images (averages and original images) with napari
    if args.view_images or args.show_good_images:
        viewer = napari.Viewer()

        # Show regular average
        viewer.add_image(
            masked_images.mean(axis=0),
            name=f"Average of all images (equal weights)",
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

            # If a reference was provided, show it too
            if results[method_name]["reference"] is not None:
                viewer.add_image(
                    results[method_name]["reference"],
                    name=f"{method_name}: reference",
                    visible=False,
                )

        # Show average of "good" (x% highest weights) and "bad" (x% highest weights)
        # images from every method
        for method_name in results:
            idx_good = results[method_name]["idx_good"]
            idx_bad = results[method_name]["idx_bad"]

            for i, q in enumerate(quantiles):
                weight_low = p_low[i]
                weight_high = p_high[i]
                good_images = masked_images[idx_good["quantile"][q]]
                bad_images = masked_images[idx_bad["quantile"][q]]

                print(f"\nShowing good and bad images for quantile {q}")
                print(f"High weight threshold: {weight_high:.4f}")
                print(f"Low weight threshold:  {weight_low:.4f}")
                print(f"Number of good images: {good_images.shape[0]}")
                print(f"Number of bad images:  {bad_images.shape[0]}\n")

                viewer.add_image(
                    good_images.mean(axis=0),
                    name=f"Average of {100 * q}% best images (method: {method_name}). Weight > {weight_high}",
                    visible=False,
                )

                viewer.add_image(
                    bad_images.mean(axis=0),
                    name=f"Average of {100 * q}% worst images (method: {method_name}). Weight < {weight_low}",
                    visible=False,
                )

            for thr in fixed_thresholds:
                good_images = masked_images[idx_good["fixed_threshold"][thr]]
                bad_images = masked_images[idx_bad["fixed_threshold"][thr]]

                print(f"\nShowing good and bad images for weight threshold {thr}")
                print(
                    f"Good images: weight >= threshold. Bad images: weight < threshold"
                )
                print(f"Number of good images: {good_images.shape[0]}")
                print(f"Number of bad images:  {bad_images.shape[0]}\n")

                viewer.add_image(
                    good_images.mean(axis=0),
                    name=f"Average of good images (method: {method_name}). Weight >= {thr}",
                    visible=False,
                )

                viewer.add_image(
                    bad_images.mean(axis=0),
                    name=f"Average of bad images (method: {method_name}). Weight < {thr}",
                    visible=False,
                )

        ## Adjust contrast limits to better compare images
        # Sweep through the data of all added layers to find the absolute extremes
        global_min = float(min(layer.data.min() for layer in viewer.layers))
        global_max = float(max(layer.data.max() for layer in viewer.layers))

        # Link and apply
        viewer.layers.link_layers(viewer.layers, attributes=["contrast_limits"])
        viewer.layers[0].contrast_limits = (global_min, global_max)

        # Show 50 random images examples
        n_show = min(50, masked_images.shape[0])
        idx_show = np.random.choice(masked_images.shape[0], size=n_show, replace=False)
        viewer.add_image(
            masked_images[idx_show],
            name=f"{n_show} random images from the sample",
            visible=False,
        )

        # Run the viewer with napari
        napari.run()


if __name__ == "__main__":
    main()
