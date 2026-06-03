from typing import Any, Iterable

import numpy as np
import torch

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.domain.metrics import MethodMetrics, ClassificationMetrics
from method_comparison.domain.reports import MethodResults, EvaluationReport
from method_comparison.evaluation.aggregation import (
    compute_aggregated_weights,
    setup_energy_reference,
)
from method_comparison.evaluation.classification_metrics import (
    ALL_RECALL_METHODS,
    compute_classification_metrics,
    compute_fourier_ring_classification_metrics,
)
from method_comparison.evaluation.frc import FRCThreshold
from method_comparison.evaluation.reconstruction_metrics import (
    compute_reconstruction_metrics,
    get_half_set_indices,
)


def compute_report(
    results: dict[str, Any],
    images_dict: dict[Space, torch.Tensor],
    ground_truth_img: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    reapply_mask: bool = True,
    mask: np.ndarray = np.array([1]),
    frc_thresholds: list[FRCThreshold] | None = None,
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
    real_agg_strategies: Iterable[AggregationStrategy] = ("mean",),
    fourier_agg_strategies: Iterable[AggregationStrategy] = ("mean", "energy"),
    energy_reference: str = "ground_truth",
    pixel_size: float = 1.0,
    independent_half_sets: bool = False,
    masks_dict: dict[Space, np.ndarray | torch.Tensor | None] | None = None,
) -> EvaluationReport:
    """
    Compute all quantitative metrics for a set of estimation results.
    """
    if frc_thresholds is None:
        frc_thresholds = [FRCThreshold.ONE_OVER_SEVEN]

    ref_real, ref_fourier = setup_energy_reference(
        ground_truth_img, images_dict, energy_reference
    )

    # Generate split indices for half-set resolution
    imgs = images_dict[Space.REAL]
    split_indices = get_half_set_indices(num_images=imgs.shape[0], device=imgs.device)

    # Parse and prepare target torch masks for weight aggregation in real and fourier space
    torch_masks = {}
    # If a mask has been provided for real space images and no weight mask is provided, use that
    if masks_dict is None:
        if mask is not None and mask.ndim == 2:
            torch_masks[Space.REAL] = torch.from_numpy(mask).to(imgs.device)
    else:
        for space, m in masks_dict.items():
            if m is not None:
                torch_masks[space] = (
                    torch.from_numpy(m).to(imgs.device)
                    if isinstance(m, np.ndarray)
                    else m.to(imgs.device)
                )
        # Use real-space image mask as weight mask as fallback
        if masks_dict.get(Space.REAL, None) is None:
            if mask is not None and mask.ndim == 2:
                torch_masks[Space.REAL] = torch.from_numpy(mask).to(imgs.device)

    all_results = []
    for method_name, data in results.items():
        estimator = data["estimator"]
        weights = data["weights"]

        # Get the estimated image for this method
        estimated_img = data["avg"].detach().cpu().numpy()
        if reapply_mask:
            estimated_img *= mask
            comparison_ground_truth = (
                ground_truth_img * mask if ground_truth_img is not None else None
            )
        else:
            comparison_ground_truth = ground_truth_img

        # Reconstruction quality metrics
        reconstruction_metrics, gt_frc_data, hs_frc_data = (
            compute_reconstruction_metrics(
                comparison_ground_truth,
                estimated_img,
                frc_thresholds=frc_thresholds,
                images_dict=images_dict,
                estimator=estimator,
                weights=weights,
                split_indices=split_indices,
                pixel_size=pixel_size,
                reapply_mask=reapply_mask,
                mask=mask,
                independent_half_sets=independent_half_sets,
            )
        )

        aggregated_weights = compute_aggregated_weights(
            weights_dict=data["weights"],
            real_agg_strategies=real_agg_strategies,
            fourier_agg_strategies=fourier_agg_strategies,
            ref_real=ref_real,
            ref_fourier=ref_fourier,
            masks_dict=torch_masks,
        )

        fourier_ring_metrics: dict[Space, dict[int, ClassificationMetrics]] = {}
        if labels is not None:
            # Image classification metrics by space
            space_metrics = compute_classification_metrics(
                agg_weights=aggregated_weights,
                labels=labels,
                recall_methods=recall_methods,
            )

            # Classification metrics per ring for Fourier spaces
            for space in [Space.FOURIER_REAL, Space.FOURIER_IMAG]:
                w = weights.get(space)
                if w is not None and w.shape[-1] > 1:
                    fourier_ring_metrics[space] = (
                        compute_fourier_ring_classification_metrics(
                            fourier_weights=weights[space],
                            labels=labels,
                            recall_methods=recall_methods,
                        )
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
                fourier_ring_metrics=fourier_ring_metrics,
            )
        )

    return EvaluationReport(
        method_results=all_results, labels=labels, frc_thresholds=frc_thresholds
    )
