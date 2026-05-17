import warnings
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.domain.metrics import SpaceMetrics
from method_comparison.evaluation.aggregation import compute_aggregated_weights, _get_space_reference

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
) -> SpaceMetrics:

    # Normalize scores between 0 and 1 for AP calculation
    max_score = scores.max()
    norm_scores = scores / max_score if max_score > 0 else scores

    ap = average_precision_score(idx_good, norm_scores)
    soft_precision = get_precision(scores, idx_good)
    soft_recall = {
        method: get_recall(scores, idx_good, method)
        for method in recall_methods
    }

    return SpaceMetrics(
        ap=ap,
        soft_precision=soft_precision,
        soft_recall=soft_recall
    )


def compute_space_metrics(
    agg_weights: dict[Space, dict[AggregationStrategy, np.ndarray]],
    labels: np.ndarray,
    recall_methods = Iterable[str]
) -> dict[Space, dict[AggregationStrategy, dict]]:
    space_metrics: dict[Space, dict[AggregationStrategy, SpaceMetrics]] = {}

    for space, data in agg_weights.items():
        if data is None:
            continue

        space_metrics[space] = {}

        for strategy in data:
            w = data[strategy]
            space_metrics[space][strategy] = compute_soft_metrics(
                scores = w, idx_good = labels == 0, recall_methods=recall_methods
            )

    return space_metrics