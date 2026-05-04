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
import os


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
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        help="Quantiles for which to show (and optionally save with --save_quantiles) best and worst images",
    )
    parser.add_argument(
        "--save_quantiles",
        default=False,
        action="store_true",
        help="If True, save images with highest and lowest weights for each quantile in --quantiles",
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
    args = parser.parse_args()

    # Load configurations
    cfg = load_config(args.config, args.snr)

    # Read images from file path
    aligned_images = mrcfile.read(cfg["data"]["image_path"])
    images_dir = os.path.dirname(cfg["data"]["image_path"])

    if args.save_original:
        try:
            images_save = mrcfile.read(cfg["data"]["original_particles_path"])
            succesful_read = True
        except KeyError:
            print("Warning: config file did not contain path to original images")
            succesful_read = False
        except FileNotFoundError:
            print("Warning: original images were not found in the config file path")
            succesful_read = False
        if not succesful_read:
            print("Using aligned images for saving")
            images_save = mrcfile.read(cfg["data"]["images_path"])
    else:
        images_save = aligned_images

    # Create mask and apply to images
    image_shape = aligned_images.shape[1:]
    mask = create_circular_mask(image_shape, cfg["mask"]["params"]["radius"])
    masked_images = mask * aligned_images

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
    report_unlabeled(results, args.plot_weights)

    # Identify x% of images with lowest and highest weights for every estimator
    fixed_thresholds = np.array([0.50])
    quantiles = np.array(args.quantiles)
    for method_name, estimator in estimators.items():
        if isinstance(estimator, ADMMSolver):
            weights = estimator.final_weights[Space.REAL]
        else:
            weights = estimator.final_weights[estimator.space]

        weights = aggregate_weights(weights, "mean")

        if quantiles.size > 0:
            if args.save_quantiles:
                subsets_dir = images_dir + "/subsets/"
                os.makedirs(subsets_dir, exist_ok=True)

            # Calculate weight percentiles
            p_low = np.quantile(weights, quantiles)
            p_high = np.quantile(weights, 1 - quantiles)

            # Initialize indices dictionaries
            idx_good = {"quantile": {}, "fixed_threshold": {}}
            idx_bad = {"quantile": {}, "fixed_threshold": {}}

            # Iterate over quantiles to find good and bad images
            for i, q in enumerate(quantiles):
                weight_low = p_low[i]
                weight_high = p_high[i]
                idx_bad["quantile"][q] = weights < weight_low
                idx_good["quantile"][q] = weights >= weight_high

                # Print diagnostics to terminal
                print(f"\nCalculated good and bad images for quantile {q}.")
                print(f"High weight threshold: {weight_high:.4f}")
                print(f"Low weight threshold:  {weight_low:.4f}")
                print("Good images: weight >= high thr. Bad images: weight < low thr")
                print(f"Number of good images: {idx_good["quantile"][q].sum()}")
                print(f"Number of bad images:  {idx_bad["quantile"][q].sum()}\n")

                # Save good and bad images for this quantile
                if args.save_quantiles:
                    mrcfile.write(
                        subsets_dir + f"{method_name}_{100*q:.0f}pct_best.mrcs",
                        data=images_save[idx_good["quantile"][q]],
                        overwrite=False,
                    )
                    mrcfile.write(
                        subsets_dir + f"{method_name}_{100*q:.0f}pct_worst.mrcs",
                        data=images_save[idx_bad["quantile"][q]],
                        overwrite=False,
                    )

        # Iterate over fixed weight thresholds to find good and bad images
        for thr in fixed_thresholds:
            idx_bad["fixed_threshold"][thr] = weights < thr
            idx_good["fixed_threshold"][thr] = weights >= thr

            # Print diagnostic info to terminal
            print(f"\nCalculated good and bad images for weight threshold {thr}")
            print(f"Good images: weight >= threshold. Bad images: weight < threshold")
            print(f"Number of good images: {idx_good["fixed_threshold"][thr].sum()}")
            print(f"Number of bad images:  {idx_bad["fixed_threshold"][thr].sum()}\n")

        # Save the good and bad indices for this estimation method
        results[method_name]["idx_good"] = idx_good
        results[method_name]["idx_bad"] = idx_bad

        # Save weights file for this estimation method if needed
        if args.save_weights:
            weights_dir = images_dir + "weights/"
            np.save(weights_dir + f"{method_name}_weights.npy", weights)

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
                good_images = masked_images[idx_good["quantile"][q]]
                bad_images = masked_images[idx_bad["quantile"][q]]

                viewer.add_image(
                    good_images.mean(axis=0),
                    name=f"Average of {100 * q}% best images (method: {method_name})",
                    visible=False,
                )

                viewer.add_image(
                    bad_images.mean(axis=0),
                    name=f"Average of {100 * q}% worst images (method: {method_name})",
                    visible=False,
                )

            for thr in fixed_thresholds:
                good_images = masked_images[idx_good["fixed_threshold"][thr]]
                bad_images = masked_images[idx_bad["fixed_threshold"][thr]]

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
