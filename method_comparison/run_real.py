import argparse
import yaml
import numpy as np
import torch
import napari
import mrcfile
from estimators import build_estimator
from method_comparison.evaluator import report_unlabeled
from method_comparison.gmm_evaluation import evaluate_gmm_fits_unlabeled
from utils.masks import create_circular_mask


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

    # Read images from file path
    images = mrcfile.read(cfg["data"]["image_path"])

    # Create mask and apply to images
    image_shape = images.shape[1:]
    mask = create_circular_mask(image_shape, cfg["mask"]["params"]["radius"])
    images = mask * images

    # Convert to tensor for the models
    tensor_images = torch.from_numpy(images).to(dtype=torch.float32, device=args.device)

    # Run Estimators
    estimators = {}
    results = {}
    
    for method_cfg in cfg["experiment"]["methods"]:
        method_name = method_cfg["name"]
        print(f"Running {method_name}...")
        
        estimator = build_estimator(method_cfg, images, device=args.device) # Use your builder function
        estimator.fit(tensor_images)
        
        estimators[method_name] = estimator
        results[method_name] = {
            "avg": estimator.avg,
            "weights": estimator.final_weights
        }

    # Show report of results
    report_unlabeled(results)

    # Evaluate gmm fits
    if args.gmm_evaluation:
        evaluate_gmm_fits_unlabeled(results, estimators, tensor_images)

    # Show images (averages and original images) with napari
    if args.view_images:
        viewer = napari.Viewer()

        # Show regular average
        viewer.add_image(
            images.mean(axis=0),
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

        ## Adjust contrast limits to better compare images
        # Sweep through the data of all added layers to find the absolute extremes
        global_min = float(min(layer.data.min() for layer in viewer.layers))
        global_max = float(max(layer.data.max() for layer in viewer.layers))

        # Link and apply
        viewer.layers.link_layers(viewer.layers, attributes=["contrast_limits"])
        viewer.layers[0].contrast_limits = (global_min, global_max)

        # Show 100 random images examples
        n_show = min(100, images.shape[0])
        idx_show = np.random.choice(images.shape[0], size=n_show, replace=False)
        viewer.add_image(
            images[idx_show],
            name=f"{n_show} random images from the sample",
            visible=False,
        )

        # Run the viewer with napari
        napari.run()


if __name__ == "__main__":
    main()
