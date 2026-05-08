import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, average_precision_score
from estimators.base import Space
from typing import Iterable, Tuple, Dict

# Helper for consistent coloring
LABEL_MAP = {
    0: {"name": "Inliers", "color": "blue"},
    1: {"name": "Rotated Outliers", "color": "orange"},
    2: {"name": "Misclassified", "color": "red"},
}

# List of all implemented recall methods
ALL_RECALL_METHODS = ["huang_tagare", "inlier_avg", "global_avg"]


### Weight aggregation ###


def aggregate_weights(
    weights: torch.Tensor, strategy: str = "mean", reference: torch.Tensor | None = None
) -> np.ndarray:
    """
    Aggregates per-pixel weights into a single score per image.
    Strategies:
      - "mean": Standard global average across spatial dimensions.
      - "energy": Uses the reference image to weight the pixels by signal energy.
    """
    w = weights.detach()

    # If weights are already per-image (N,) or (N, 1, 1)
    if w.ndim == 1 or (w.ndim == 3 and w.shape[1:] == (1, 1)):
        return w.cpu().numpy().flatten()

    if strategy == "mean":
        return w.mean(dim=(1, 2)).cpu().numpy()

    elif strategy == "energy":
        if reference is None:
            raise ValueError("Energy aggregation requires a reference image.")

        # Calculate signal energy from reference (normalized)
        energy = torch.abs(reference).detach() ** 2
        energy = energy / (energy.sum() + 1e-12)

        # Energy-weighted average
        scores = torch.sum(w * energy, dim=(1, 2))
        return scores.cpu().numpy()

    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


### Metric computation ###


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
) -> Dict[str, float]:

    # Normalize scores between 0 and 1 for AP calculation
    max_score = scores.max()
    norm_scores = scores / max_score if max_score > 0 else scores

    metrics = {
        "ap": average_precision_score(idx_good, norm_scores),
        "soft_precision": get_precision(scores, idx_good),
    }
    for method in recall_methods:
        metrics[f"soft_recall_{method}"] = get_recall(scores, idx_good, method)

    return metrics


### Fourier ring correlation ###


def compute_fsc(
    image1: np.ndarray, image2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 2D Fourier Ring/Shell Correlation between two images.
    Returns the normalized frequencies and the FSC curve.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape to compute FSC.")

    # Compute 2D FFTs and shift zero frequency to center
    F1 = np.fft.fftshift(np.fft.fft2(image1))
    F2 = np.fft.fftshift(np.fft.fft2(image2))

    # Create radial distance map
    shape = image1.shape
    center = (shape[0] // 2, shape[1] // 2)
    y, x = np.indices(shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = np.round(r).astype(int)

    # Calculate Nyquist frequency (max radius)
    max_r = int(np.min([center[0], center[1]]))

    fsc = np.zeros(max_r)
    freqs = np.arange(max_r) / max_r  # Normalized frequency [0, 1] (1 = Nyquist)

    for i in range(max_r):
        mask = r == i
        if np.sum(mask) == 0:
            continue

        f1_shell = F1[mask]
        f2_shell = F2[mask]

        # Cross-correlation numerator
        num = np.real(np.sum(f1_shell * np.conj(f2_shell)))

        # Normalization denominator
        den = np.sqrt(np.sum(np.abs(f1_shell) ** 2) * np.sum(np.abs(f2_shell) ** 2))

        fsc[i] = num / den if den > 0 else 0.0

    return freqs, fsc


def get_resolution_from_fsc(
    freqs: np.ndarray, fsc: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Finds the spatial frequency where the FSC curve first drops below the threshold.
    Uses linear interpolation for sub-bin precision.
    """
    drop_idx = np.where(fsc < threshold)[0]

    if len(drop_idx) == 0:
        return freqs[-1]  # Never drops below threshold (perfect resolution)

    idx = drop_idx[0]
    if idx == 0:
        return freqs[0]

    # Linear interpolation
    f1, f2 = fsc[idx - 1], fsc[idx]
    q1, q2 = freqs[idx - 1], freqs[idx]

    # Solve for frequency crossing the threshold
    freq_thresh = q1 + (threshold - f1) * (q2 - q1) / (f2 - f1)
    return freq_thresh
