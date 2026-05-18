import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

from estimators.base import Estimator

from method_comparison.domain.enums import Space
from method_comparison.domain.metrics import ReconstructionMetrics
from method_comparison.evaluation.frc import (
    FRCThreshold,
    FRCData,
    compute_frc,
    get_resolution,
    area_under_frc,
)

### Fourier ring correlation ###


# def compute_frc(
#     image1: np.ndarray, image2: np.ndarray, pixel_size: float = 1.0
# ) -> FRCData:
#     """
#     Computes the 2D Fourier Ring Correlation between two images.
#     Returns spatial resolutions (Å), spatial frequencies (1/Å), and the FRC curve.
#     """
#     if image1.shape != image2.shape:
#         raise ValueError("Images must have the same shape to compute FRC.")

#     # Compute 2D FFTs and shift zero frequency to center
#     F1 = np.fft.fftshift(np.fft.fft2(image1))
#     F2 = np.fft.fftshift(np.fft.fft2(image2))

#     # Create radial distance map
#     shape = image1.shape
#     center = (shape[0] // 2, shape[1] // 2)
#     y, x = np.indices(shape)
#     r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
#     r = np.round(r).astype(int)

#     # Calculate max radius and effective diameter for frequency scaling
#     max_r = int(np.min([center[0], center[1]]))
#     box_size = 2 * max_r

#     frc = np.zeros(max_r)
#     # Spatial frequencies: k = i / (box_size * pixel_size)
#     freqs = np.arange(max_r) / (box_size * pixel_size)

#     # Calculate resolutions: d = 1 / k
#     resolutions = np.zeros(max_r)
#     resolutions[0] = np.inf  # DC component (0 frequency) represents infinite resolution
#     resolutions[1:] = 1.0 / freqs[1:]

#     for i in range(max_r):
#         mask = r == i
#         if np.sum(mask) == 0:
#             continue

#         f1_shell = F1[mask]
#         f2_shell = F2[mask]

#         # Cross-correlation numerator
#         num = np.real(np.sum(f1_shell * np.conj(f2_shell)))

#         # Normalization denominator
#         den = np.sqrt(np.sum(np.abs(f1_shell) ** 2) * np.sum(np.abs(f2_shell) ** 2))

#         frc[i] = num / den if den > 0 else 0.0

#     return FRCData(resolutions=resolutions, freqs=freqs, frc=frc)


# def get_resolution_from_frc(frc_data: FRCData, threshold: float = 0.5) -> float:
#     """
#     Finds the spatial resolution (in Å) where the FRC curve first drops below the threshold.
#     Interpolates in frequency space for accuracy, then inverts to return resolution.
#     """
#     freqs = frc_data.freqs
#     frc = frc_data.frc
#     drop_idx = np.where(frc < threshold)[0]

#     # Correlation never drops below threshold (perfect resolution)
#     if len(drop_idx) == 0:
#         return 1.0 / freqs[-1] if freqs[-1] > 0 else np.inf

#     idx = drop_idx[0]
#     if idx == 0:
#         return np.inf

#     # Linear interpolation in frequency space
#     f1, f2 = frc[idx - 1], frc[idx]
#     q1, q2 = freqs[idx - 1], freqs[idx]

#     # Solve for frequency crossing the threshold
#     if f2 == f1:
#         freq_thresh = q1
#     else:
#         freq_thresh = q1 + (threshold - f1) * (q2 - q1) / (f2 - f1)

#     # Convert frequency back to spatial resolution
#     return np.inf if freq_thresh <= 0 else 1.0 / freq_thresh


def get_half_set_indices(
    num_images: int,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a reproducible random split of indices for half-sets.

    Returns:
        Tuple of torch.LongTensor indices.
    """
    g = torch.Generator(device=device)

    # Important: CUDA generators require manual_seed on the generator
    g.manual_seed(seed)

    indices = torch.randperm(
        num_images,
        generator=g,
        device=device,
        dtype=torch.long,
    )

    half_idx = num_images // 2
    return indices[:half_idx], indices[half_idx:]


### All reconstruction metrics


def compute_reconstruction_metrics(
    ground_truth_img: np.ndarray | None,
    estimated_img: np.ndarray,
    frc_thresholds: list[FRCThreshold],
    images_dict: dict[Space, torch.Tensor],
    estimator: Estimator,
    weights: torch.Tensor,
    split_indices: tuple[torch.Tensor, torch.Tensor],
    pixel_size: float = 1.0,
) -> tuple[ReconstructionMetrics, FRCData, FRCData]:

    ## Half-set reconstruction resolution (always available)
    # Separate images and weights into two half-sets
    idx_A, idx_B = split_indices
    images_A = {space: images_dict[space][idx_A] for space in Space}
    images_B = {space: images_dict[space][idx_B] for space in Space}
    weights_A = {
        space: weights[space][idx_A] if weights[space] is not None else None
        for space in Space
    }
    weights_B = {
        space: weights[space][idx_B] if weights[space] is not None else None
        for space in Space
    }

    # Reconstruct image estimation for both half sets
    if estimator is not None:
        reconstruction_A = estimator.reconstruct_from_weights(images_A, weights_A)
        reconstruction_B = estimator.reconstruct_from_weights(images_B, weights_B)
    else:
        reconstruction_A = images_A[Space.REAL].mean(dim=0)
        reconstruction_B = images_B[Space.REAL].mean(dim=0)
    reconstruction_A = reconstruction_A.detach().cpu().numpy()
    reconstruction_B = reconstruction_B.detach().cpu().numpy()

    # Calculate FRC and resolution by comparing both reconstructions
    half_set_frc_data = compute_frc(
        reconstruction_A, reconstruction_B, pixel_size=pixel_size
    )
    for threshold in frc_thresholds:
        half_set_frc_data.resolutions[threshold] = get_resolution(
            half_set_frc_data, threshold
        )
    half_set_aufrc = area_under_frc(half_set_frc_data)

    # If ground truth is not available, nothing else to calculate
    if ground_truth_img is None:
        metrics = ReconstructionMetrics(
            rmse=None,
            pearson_corr=None,
            gt_frc_resolutions=None,
            hs_frc_resolutions=half_set_frc_data.resolutions,
            gt_aufrc=None,
            hs_aufrc=half_set_aufrc,
        )
        return metrics, None, half_set_frc_data

    # Ground truth available: calculate error metrics
    rmse = root_mean_squared_error(ground_truth_img, estimated_img)
    corr, _ = pearsonr(ground_truth_img.flatten(), estimated_img.flatten())

    # and ground truth FRC
    ground_truth_frc_data = compute_frc(estimated_img, ground_truth_img)
    for threshold in frc_thresholds:
        ground_truth_frc_data.resolutions[threshold] = get_resolution(
            ground_truth_frc_data, threshold=threshold
        )
    ground_truth_aufrc = area_under_frc(ground_truth_frc_data)

    # Build and return ReconstructionMetrics object
    metrics = ReconstructionMetrics(
        rmse=rmse,
        pearson_corr=corr,
        gt_frc_resolutions=ground_truth_frc_data.resolutions,
        hs_frc_resolutions=half_set_frc_data.resolutions,
        gt_aufrc=ground_truth_aufrc,
        hs_aufrc=half_set_aufrc,
    )

    return metrics, ground_truth_frc_data, half_set_frc_data
