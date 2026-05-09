import numpy as np
import mrcfile
import scipy
from typing import Tuple

LABEL_TYPES = {
    0: "generated copies of reference",
    1: "very rotated copies of reference",
    2: "misclassified outliers",
}

STANDARDIZE_TYPES = ["global", "per_particle"]


def add_noise(
    images: np.ndarray,
    rng: np.random.Generator,
    signal_var: float,
    snr: float | None = None,
    noise_std: float | None = None,
) -> np.ndarray:
    """Adds Gaussian noise to a batch of images based on target SNR or explicit noise_std."""
    if noise_std is not None:
        # Calculate signal-to-noise ratio
        snr = signal_var / noise_std**2
    elif snr is not None:
        # Calculate noise standard deviation
        noise_var = signal_var / snr
        noise_std = np.sqrt(noise_var)
    else:
        raise ValueError("Must provide either 'snr' or 'noise_std' in the config.")

    print("Adding noise to images:")
    print(
        f"\t- Average image std:  {np.sqrt(signal_var):.4f}\tVariance:  {signal_var:6f}"
    )
    print(f"\t- Noise std:          {noise_std:.4f}\tVariance:  {noise_std**2:.6f}")
    print(f"\t- SNR:                {snr:.4f}\n")

    return images + rng.normal(0, noise_std, size=images.shape)


def generate_rotated_copies(
    image: np.ndarray,
    n_copies: int,
    min_angle: float,
    max_angle: float,
    rng: np.random.Generator,
    interpolation_order: int = 3,
) -> np.ndarray:
    """Generates rotated copies of a single 2D reference image."""
    output_images = np.zeros(
        (n_copies, image.shape[0], image.shape[1]), dtype=image.dtype
    )

    # Pre-generate all random angles for slight efficiency gain
    angles = rng.uniform(min_angle, max_angle, size=n_copies)

    for i, angle in enumerate(angles):
        scipy.ndimage.rotate(
            image,
            angle,
            order=interpolation_order,
            reshape=False,
            output=output_images[i],
        )

    return output_images


def load_misclassified_images(
    image_path: str, n_copies: int, rng: np.random.Generator
) -> np.ndarray:
    """Loads a file of outlier images and samples the requested number."""
    images = mrcfile.read(image_path)

    # Handle the one image case
    if images.ndim == 2:
        h, w = images.shape
        images = images.reshape(1, h, w)

    n_available = images.shape[0]

    replace = n_available < n_copies

    # Sample indices
    indices = rng.choice(n_available, size=n_copies, replace=replace)

    return images[indices]


def create_evaluation_dataset(
    cfg: dict, rng: np.random.Generator, standardize: str | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an evaluation dataset with rotated inliers, rotated outliers,
    misclassified outliers, and applies Gaussian noise.
    """
    data_cfg = cfg["data"]
    gen_cfg = cfg["generation"]
    noise_cfg = cfg["noise"]

    # Load Reference
    ref_image = mrcfile.read(data_cfg["reference_image_path"])
    h, w = ref_image.shape
    true_signal_var = ref_image.var()

    n_good = gen_cfg["n_copies"]
    n_rot_bad = gen_cfg["n_copies_rotated"]
    n_misc = gen_cfg["n_misclassified"]
    total_copies = n_good + n_rot_bad + n_misc

    # Pre-allocate the dataset array
    dataset = np.zeros((total_copies, h, w), dtype=ref_image.dtype)

    # labels array: 0 for inliers, 1 for rotated outliers, 2 for misclassified outliers
    labels = np.zeros(total_copies, dtype=int)

    current_idx = 0

    # Fill "good" copies
    if n_good > 0:
        dataset[current_idx : current_idx + n_good] = generate_rotated_copies(
            ref_image,
            n_good,
            min_angle=-gen_cfg["max_rotation_reference"],
            max_angle=gen_cfg["max_rotation_reference"],
            rng=rng,
        )
        labels[current_idx : current_idx + n_good] = 0
        current_idx += n_good

    # Fill "very rotated" outliers
    if n_rot_bad > 0:
        dataset[current_idx : current_idx + n_rot_bad] = generate_rotated_copies(
            ref_image,
            n_rot_bad,
            min_angle=gen_cfg["min_rotation_very_rotated"],
            max_angle=gen_cfg["max_rotation_very_rotated"],
            rng=rng,
        )
        labels[current_idx : current_idx + n_rot_bad] = 1
        current_idx += n_rot_bad

    # Fill misclassified images
    if n_misc > 0:
        dataset[current_idx : current_idx + n_misc] = load_misclassified_images(
            data_cfg["misclassified_path"], n_misc, rng=rng
        )
        labels[current_idx : current_idx + n_misc] = 2

    # Add noise to the array
    final_dataset = add_noise(
        dataset,
        rng=rng,
        signal_var=true_signal_var,
        snr=noise_cfg.get("snr"),
        noise_std=noise_cfg.get("noise_std"),
    )

    if standardize == "global":
        global_mean = final_dataset.mean()
        global_std = final_dataset.std()
        final_dataset = (final_dataset - global_mean) / (global_std + 1e-8)
    elif standardize == "per_particle":
        means = final_dataset.mean(axis=(1, 2), keepdims=True)
        stds = final_dataset.std(axis=(1, 2), keepdims=True)
        final_dataset = (final_dataset - means) / (stds + 1e-8)

    return final_dataset, ref_image, labels
