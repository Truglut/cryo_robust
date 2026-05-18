from typing import Any, Iterable

import numpy as np
import torch

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.domain.metrics import MethodMetrics
from method_comparison.domain.reports import MethodResults, EvaluationReport
from method_comparison.evaluation.aggregation import (
    compute_aggregated_weights,
    setup_energy_reference,
)
from method_comparison.evaluation.classification_metrics import (
    ALL_RECALL_METHODS,
    compute_space_metrics,
)
from method_comparison.evaluation.frc import FRCThreshold
from method_comparison.evaluation.reconstruction_metrics import (
    compute_reconstruction_metrics,
    get_half_set_indices,
)



def compute_report_labeled(
    results: dict[str, Any],
    images_dict: dict[Space, torch.Tensor],
    ground_truth_img: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    reapply_mask: bool = False,
    mask: np.ndarray = np.array([1]),
    frc_thresholds: list[FRCThreshold] = [FRCThreshold.ONE_OVER_SEVEN],
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
    real_agg_strategies: Iterable[AggregationStrategy] = ("mean",),
    fourier_agg_strategies: Iterable[AggregationStrategy] = ("mean", "energy"),
    energy_reference: str = "ground_truth",
    pixel_size: float = 1.0,
) -> EvaluationReport:
    """
    Compute all quantitative metrics for a set of estimation results.
    """
    ref_real, ref_fourier = setup_energy_reference(
        ground_truth_img, images_dict, energy_reference
    )

    all_results = []

    # Generate split indices for half-set resolution
    imgs = images_dict[Space.REAL]
    split_indices = get_half_set_indices(num_images=imgs.shape[0], device=imgs.device)

    for method_name, data in results.items():
        estimator = data["estimator"]
        weights = data["weights"]

        # Get the estimated image for this method
        estimated_img = data["avg"].detach().cpu().numpy()
        if reapply_mask:
            estimated_img *= mask

        # Reconstruction quality metrics
        reconstruction_metrics, gt_frc_data, hs_frc_data = (
            compute_reconstruction_metrics(
                ground_truth_img,
                estimated_img,
                frc_thresholds=frc_thresholds,
                images_dict=images_dict,
                estimator=estimator,
                weights=weights,
                split_indices=split_indices,
                pixel_size=pixel_size,
            )
        )

        aggregated_weights = compute_aggregated_weights(
            weights_dict=data["weights"],
            real_agg_strategies=real_agg_strategies,
            fourier_agg_strategies=fourier_agg_strategies,
            ref_real=ref_real,
            ref_fourier=ref_fourier,
        )

        if labels is not None:
            # Image classification metrics by space
            space_metrics = compute_space_metrics(
                agg_weights=aggregated_weights,
                labels=labels,
                recall_methods=recall_methods,
            )
        else:
            space_metrics = None

        method_metrics = MethodMetrics(
            reconstruction_metrics=reconstruction_metrics, space_metrics=space_metrics
        )
        all_results.append(
            MethodResults(
                name=method_name,
                metrics=method_metrics,
                scores=aggregated_weights,
                ground_truth_frc_data=gt_frc_data,
                half_set_frc_data=hs_frc_data,
                estimated_img=estimated_img,
            )
        )

    return EvaluationReport(
        method_results=all_results, labels=labels, frc_thresholds=frc_thresholds
    )
