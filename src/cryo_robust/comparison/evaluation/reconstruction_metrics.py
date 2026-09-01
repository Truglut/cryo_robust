import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

from cryo_robust.estimators.irls import IRLSSolver
from cryo_robust.estimators.data import ImageBatch

from cryo_robust.domain import ImageSpace
from cryo_robust.comparison.domain.metrics import ReconstructionMetrics
from cryo_robust.comparison.domain.runs import AVERAGE_NAME, MethodRun

from ..domain.frc import FRCData, FRCThreshold
from .frc import (
    compute_frc,
    get_resolution,
    area_under_frc,
)
from cryo_robust.comparison.domain.runs import MEDIAN_NAME


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


def reconstruct_baseline_half_sets(
    method_name: str, batch_a: ImageBatch, batch_b: ImageBatch
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if method_name == AVERAGE_NAME:
        return (
            batch_a.ensure_real().mean(dim=0),
            batch_b.ensure_real().mean(dim=0),
        )

    if method_name == MEDIAN_NAME:
        return (
            batch_a.ensure_real().median(dim=0).values,
            batch_b.ensure_real().median(dim=0).values,
        )

    return None


### All reconstruction metrics


def compute_reconstruction_metrics(
    ground_truth_img: np.ndarray | None,
    estimated_img: np.ndarray,
    frc_thresholds: list[FRCThreshold],
    image_batch: ImageBatch,
    method_name: str,
    method_run: MethodRun,
    split_indices: tuple[torch.Tensor, torch.Tensor],
    pixel_size: float = 1.0,
    reapply_mask: bool = True,
    mask: np.ndarray = np.ndarray([1]),
    independent_half_sets: bool = True,
) -> tuple[ReconstructionMetrics, FRCData, FRCData]:

    ## Half-set reconstruction resolution (always available)
    # Separate images and weights into two half-sets
    idx_A, idx_B = split_indices
    batch_A = image_batch.subset(idx_A)
    batch_B = image_batch.subset(idx_B)

    estimator = method_run.estimator
    weights = method_run.result.weights

    # Reconstruct image estimation for both half sets
    if (
        baseline := reconstruct_baseline_half_sets(method_name, batch_A, batch_B)
    ) is not None:
        reconstruction_A, reconstruction_B = baseline

    # Independent reconstruction: fit estimators again on the half-sets
    elif independent_half_sets:
        if estimator is None:
            raise RuntimeError(f"Method {method_name!r} has no estimator.")

        reconstruction_A = estimator.fit(batch_A).average
        reconstruction_B = estimator.fit(batch_B).average

    # Not independent reconstruction: reconstruct from already available weights
    else:
        weights_A = weights.subset(idx_A)
        weights_B = weights.subset(idx_B)

        reconstruction_A = estimator.reconstruct_from_weights(batch_A, weights_A)
        reconstruction_B = estimator.reconstruct_from_weights(batch_B, weights_B)

    reconstruction_A = reconstruction_A.detach().cpu().numpy()
    reconstruction_B = reconstruction_B.detach().cpu().numpy()
    if reapply_mask:
        reconstruction_A *= mask
        reconstruction_B *= mask

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
    ground_truth_frc_data = compute_frc(
        estimated_img, ground_truth_img, pixel_size=pixel_size
    )
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
