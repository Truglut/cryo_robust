import argparse
import yaml
import numpy as np
import torch
import napari
from dataset_builder import create_evaluation_dataset
from weights import get_weight_function
from admm import admm_scheme
from evaluator import compare_and_report
from masks import create_circular_mask


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
        "--plot_weights",
        default=False,
        action="store_true",
        help="If True, plot IRLS weights at last ADMM iteration",
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

    # rng seed for reproducibility
    seed = cfg.get("seed", None)
    rng = np.random.default_rng(seed=seed)

    # Generate the data (good copies, rotated outliers, misclassified outliers + noise)
    print("Generating data...")
    images, ground_truth, labels = create_evaluation_dataset(
        cfg, rng, normalize=args.normalize
    )
    # labels according to LABEL_TYPES
    n_images = images.shape[0]

    # Create and apply circular mask
    image_shape = images.shape[1:]
    mask = create_circular_mask(image_shape, cfg["mask"]["params"]["radius"])
    images = mask * images
    ground_truth = ground_truth * mask

    # Convert images to tensor and calculate fft
    tensor_images = torch.from_numpy(images).to(dtype=torch.float32, device=args.device)
    tensor_truth = torch.from_numpy(ground_truth).to(
        dtype=torch.float32, device=args.device
    )
    fourier_images = torch.fft.rfft2(tensor_images, norm="ortho")
    fourier_truth = torch.fft.rfft2(tensor_truth, norm="ortho")

    # admm initialization
    avg = torch.mean(tensor_images, dim=0)
    fourier_avg = torch.mean(fourier_images, dim=0)
    initial_ref_real = avg
    initial_ref_fourier = fourier_avg

    weight_function_real = get_weight_function("global", params={"beta": 1e-8})
    weight_function_fourier = get_weight_function("smooth", params={"delta": 1})

    # Run the admm estimation method
    results = admm_scheme(
        tensor_images,
        fourier_images,
        ctf=torch.tensor(1),
        initial_ref_real=initial_ref_real,
        initial_ref_fourier=initial_ref_fourier,
        mu=1.0,
        C=1.0,
        weight_function_real=weight_function_real,
        weight_function_fourier=weight_function_fourier,
        max_iter=50,
        atol=1e-5,
        rtol=1e-5,
    )
    print(f"{results.converged = }")
    print(f"{results.iterations = }")

    # Simple weight evaluation
    compare_and_report(
        {
            "ADMM Scheme": {
                "avg": results.estimation_real,
                "weights": results.last_weights["real"],
            }
        },
        ground_truth,
        labels,
        plot_weights=args.plot_weights,
    )
    # Weight evaluation in fourier space
    compare_and_report(
        {
            "ADMM Scheme (Fourier Space - Real Part)": {
                "avg": results.estimation_fourier.real,
                "weights": results.last_weights["fourier_real"][:, 0, 0].reshape(
                    n_images, 1, 1
                ),
            }
        },
        ground_truth_img=fourier_truth.real,
        labels=labels,
        plot_weights=args.plot_weights,
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

        # Show estimated average with admm method
        viewer.add_image(
            results.estimation_real,
            name="Estimated signal using ADMM method",
            visible=False,
        )

        # Show difference between average and admm estimation
        viewer.add_image(
            results.estimation_real - avg,
            name="Difference between estimated signal with ADMM and mean",
            visible=False,
        )

        ## Adjust contrast limits to better compare images
        # Sweep through the data of all added layers to find the absolute extremes
        global_min = float(min(layer.data.min() for layer in viewer.layers))
        global_max = float(max(layer.data.max() for layer in viewer.layers))

        # Link and apply
        viewer.layers.link_layers(viewer.layers, attributes=["contrast_limits"])
        viewer.layers[0].contrast_limits = (global_min, global_max)

        # Show relative difference between average and admm estimation
        viewer.add_image(
            torch.clamp((results.estimation_real - avg) / (avg + 0.01), min=-1, max=1),
            name="Relative difference between estimated signal with ADMM and mean",
            visible=False,
        )

        # Show examples of all image types (good, very rotated, misclassified)
        for label in LABEL_TYPES:
            n_show = min(25, (labels == label).sum())
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
