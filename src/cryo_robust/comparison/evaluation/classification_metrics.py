import warnings
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from cryo_robust.comparison.domain.enums import ImageSpace, AggregationStrategy
from cryo_robust.comparison.domain.metrics import ClassificationMetrics

# List of all implemented recall methods
ALL_RECALL_METHODS = ["huang_tagare", "inlier_avg", "global_avg"]


def get_precision(weights: np.ndarray, idx_good: np.ndarray) -> float:
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float(weights[idx_good].sum() / total_weight)


def get_recall(
    weights: np.ndarray, idx_good: np.ndarray, average_type: str = "huang_tagare"
) -> float:
    n_in = idx_good.sum()
    if n_in == 0:
        return 0.0

    if average_type == "inlier_avg":
        omega_bar = weights[idx_good].mean()
    elif average_type == "global_avg":
        omega_bar = weights.mean()
    elif average_type == "huang_tagare":
        omega_bar = weights.sum() / n_in
    else:
        warnings.warn("Unrecognised average type: using 'huang_tagare'")
        omega_bar = weights.sum() / n_in

    if omega_bar == 0:
        return 0.0
    return float(np.clip(weights[idx_good] / omega_bar, a_min=None, a_max=1.0).mean())


def compute_soft_metrics(
    scores: np.ndarray, idx_good: np.ndarray, recall_methods: Iterable[str]
) -> ClassificationMetrics:

    # Normalize scores between 0 and 1 for AP calculation
    max_score = scores.max()
    norm_scores = scores / max_score if max_score > 0 else scores

    ap = average_precision_score(idx_good, norm_scores)
    roc_auc = roc_auc_score(idx_good, norm_scores)
    soft_precision = get_precision(scores, idx_good)
    soft_recall = {
        method: get_recall(scores, idx_good, method) for method in recall_methods
    }

    return ClassificationMetrics(
        ap=ap, roc_auc=roc_auc, soft_precision=soft_precision, soft_recall=soft_recall
    )


def compute_classification_metrics(
    agg_weights: dict[ImageSpace, dict[AggregationStrategy, np.ndarray]],
    labels: np.ndarray,
    recall_methods: Iterable[str],
) -> dict[ImageSpace, dict[AggregationStrategy, dict]]:
    classification_metrics: dict[
        ImageSpace, dict[AggregationStrategy, ClassificationMetrics]
    ] = {}

    for space, data in agg_weights.items():
        if data is None:
            continue

        classification_metrics[space] = {}

        for strategy in data:
            w = data[strategy]
            classification_metrics[space][strategy] = compute_soft_metrics(
                scores=w, idx_good=labels == 0, recall_methods=recall_methods
            )

    return classification_metrics


# Fourier ring metrics calculation


def _get_radial_map(shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """
    Generates a radial frequency ring map for a given 2D shape.
    Automatically detects if the input is full-shifted FFT or rfft2 format.
    """
    h, w = shape
    # If w matches the half-spectrum convention (H // 2 + 1), handle as rfft2
    if w == h // 2 + 1:
        y_freq = np.fft.fftfreq(h) * h
        x_freq = np.arange(w)
        yy, xx = np.meshgrid(y_freq, x_freq, indexing="ij")
        r = np.sqrt(yy**2 + xx**2)
        r_int = np.round(r).astype(np.int32)
        max_r = int(h // 2)
    else:
        # Assume full shifted FFT (centered DC component)
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.indices((h, w))
        r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        r_int = np.round(r).astype(np.int32)
        max_r = int(min(cy, cx))

    return r_int, max_r


def compute_fourier_ring_classification_metrics(
    fourier_weights: torch.Tensor,
    labels: np.ndarray,
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
) -> dict[int, ClassificationMetrics]:
    """
    Calculates soft classification metrics as a function of spatial frequency
    (per Fourier ring) for a single Fourier space weight tensor.

    Parameters
    ----------
    fourier_weights : torch.Tensor
        Tensor of shape (N_images, H, W) or (N_images, H, W // 2 + 1)
    labels : np.ndarray
        1D array of ground truth classification labels.
    """
    n, h, w = fourier_weights.shape
    r_int, max_r = _get_radial_map((h, w))

    idx_good = labels == 0
    ring_metrics: dict[int, ClassificationMetrics] = {}

    # Send radial labels map to the same device as weights for fast masking
    r_int_torch = torch.from_numpy(r_int).to(fourier_weights.device)

    for k in range(max_r + 1):
        mask = r_int_torch == k
        if not mask.any():
            continue

        # Extract and average weights within the specific range per image. Shape (N, )
        ring_scores = fourier_weights[:, mask].mean(dim=1).detach().cpu().numpy()

        # Compute metrics for this specific frequency ring
        ring_metrics[k] = compute_soft_metrics(
            scores=ring_scores, idx_good=idx_good, recall_methods=recall_methods
        )

    return ring_metrics
