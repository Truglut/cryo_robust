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
    weights: torch.Tensor, 
    strategy: str = "mean", 
    reference: torch.Tensor | None = None
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
    if total_weight == 0: return 0.0
    return float(weights[idx_good].sum() / total_weight)

def get_recall(weights: np.ndarray, idx_good: np.ndarray, average_type: str = "huang_tagare") -> float:
    n_in = idx_good.sum()
    if n_in == 0: return 0.0
    
    if average_type == "inlier_avg":
        omega_bar = weights[idx_good].mean()
    elif average_type == "global_avg":
        omega_bar = weights.mean()
    elif average_type == "huang_tagare":
        omega_bar = weights.sum() / n_in
    else:
        warnings.warn("Unrecognised average type: using 'huang_tagare'")
        omega_bar = weights.sum() / n_in

    if omega_bar == 0: return 0.0
    return float(np.clip(weights[idx_good] / omega_bar, a_min=None, a_max=1.0).mean())

def compute_soft_metrics(
    scores: np.ndarray, 
    idx_good: np.ndarray, 
    recall_methods: Iterable[str]
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