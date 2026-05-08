from typing import Any, Iterable

import numpy as np
import torch

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.domain.metrics import MethodMetrics
from method_comparison.domain.reports import MethodResults, EvaluationReport
from method_comparison.evaluation.classification_metrics import (
    ALL_RECALL_METHODS,
    compute_space_metrics,
)
from method_comparison.evaluation.reconstruction_metrics import (
    compute_reconstruction_metrics,
)
from method_comparison.evaluation.aggregation import (
    compute_aggregated_weights,
    setup_energy_reference,
)


def compute_report_labeled(
    results: dict[str, Any],
    images_dict: dict[Space, torch.Tensor],
    ground_truth_img: np.ndarray,
    labels: np.ndarray,
    reapply_mask: bool = False,
    mask: np.ndarray = np.array([1]),
    fsc_threshold: float = 0.143,
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
    real_agg_strategies: Iterable[AggregationStrategy] = ("mean",),
    fourier_agg_strategies: Iterable[AggregationStrategy] = ("mean", "energy"),
    energy_reference: str = "ground_truth",
) -> EvaluationReport:
    """
    Compute all quantitative metrics for a set of estimation results.
    """
    ref_real, ref_fourier = setup_energy_reference(
        ground_truth_img, images_dict, energy_reference
    )

    all_results = []

    for method_name, data in results.items():
        # Get the estimated image for this method
        estimated_img = data["avg"].detach().cpu().numpy()
        if reapply_mask:
            estimated_img *= mask

        # Reconstruction quality metrics
        reconstruction_metrics, fsc_data = compute_reconstruction_metrics(
            ground_truth_img, estimated_img, fsc_threshold=fsc_threshold
        )

        aggregated_weights = compute_aggregated_weights(
            weights_dict=data["weights"],
            real_agg_strategies=real_agg_strategies,
            fourier_agg_strategies=fourier_agg_strategies,
            ref_real=ref_real,
            ref_fourier=ref_fourier,
        )

        # Image classification metrics by space
        space_metrics = compute_space_metrics(
            agg_weights=aggregated_weights, labels=labels, recall_methods=recall_methods
        )

        method_metrics = MethodMetrics(
            reconstruction_metrics=reconstruction_metrics, space_metrics=space_metrics
        )
        all_results.append(
            MethodResults(
                name=method_name,
                metrics=method_metrics,
                scores=aggregated_weights,
                fsc_data=fsc_data,
                estimated_img=estimated_img,
            )
        )

    return EvaluationReport(
        method_results=all_results, labels=labels, fsc_threshold=fsc_threshold
    )


def compute_report_unlabeled(
    results: dict[str, Any],
    images_dict: dict[Space, torch.Tensor],
    real_agg_strategies: Iterable[AggregationStrategy] = (AggregationStrategy.MEAN,),
    fourier_agg_strategies: Iterable[AggregationStrategy] = (
        AggregationStrategy.ENERGY,
    ),
):
    ref_real, ref_fourier = setup_energy_reference(
        ground_truth_img=None, images_dict=images_dict, energy_reference="global_avg"
    )
    all_results = []

    for method_name, data in results.items():
        aggregated_weights = compute_aggregated_weights(
            weights_dict=data["weights"],
            real_agg_strategies=real_agg_strategies,
            fourier_agg_strategies=fourier_agg_strategies,
            ref_real=ref_real,
            ref_fourier=ref_fourier,
        )

        all_results.append(
            MethodResults(
                name=method_name,
                metrics=None,
                scores=aggregated_weights,
                fsc_data=None,
                estimated_img=data["avg"].detach().cpu().numpy(),
            )
        )

    return EvaluationReport(
        method_results=all_results,
        labels=None,  # data is unlabeled
        fsc_threshold=None,  # no ground truth, unused
    )
