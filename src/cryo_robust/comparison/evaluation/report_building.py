from typing import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from cryo_robust.domain import ImageSpace
from cryo_robust.estimators.data import ImageBatch

from cryo_robust.comparison.domain.enums import AggregationStrategy
from cryo_robust.comparison.domain.metrics import MethodMetrics, ClassificationMetrics
from cryo_robust.comparison.domain.reports import MethodEvaluation, EvaluationReport
from .aggregation import (
    compute_aggregated_weights,
    setup_energy_reference,
)
from cryo_robust.comparison.domain.runs import MethodRun

from .classification_metrics import (
    ALL_RECALL_METHODS,
    compute_classification_metrics,
    compute_fourier_ring_classification_metrics,
)
from ..domain.frc import FRCThreshold
from .reconstruction_metrics import (
    compute_reconstruction_metrics,
    get_half_set_indices,
)


@dataclass(frozen=True)
class ReportComputationOptions:
    reconstruction: bool = True
    scores: bool = True
    classification: bool = True
    fourier_ring_metrics: bool = True
    store_estimated_images: bool = True


def compute_report(
    results: Mapping[str, MethodRun],
    image_batch: ImageBatch,
    ground_truth_img: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    reapply_mask: bool = True,
    mask: np.ndarray = np.array([1]),
    frc_thresholds: list[FRCThreshold] | None = None,
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
    real_agg_strategies: Iterable[AggregationStrategy] = (AggregationStrategy.MEAN,),
    fourier_agg_strategies: Iterable[AggregationStrategy] = (
        AggregationStrategy.MEAN,
        AggregationStrategy.ENERGY,
    ),
    energy_reference: str = "ground_truth",
    pixel_size: float = 1.0,
    independent_half_sets: bool = False,
    masks_dict: dict[ImageSpace, np.ndarray | torch.Tensor | None] | None = None,
    options: ReportComputationOptions | None = None,
) -> EvaluationReport:
    """
    Compute all quantitative metrics for a set of estimation results.
    """
    # Calculate all available sections by default
    if options is None:
        options = ReportComputationOptions()

    if frc_thresholds is None:
        frc_thresholds = [FRCThreshold.ONE_OVER_SEVEN]

    ref_real, ref_fourier = setup_energy_reference(
        ground_truth_img, image_batch, energy_reference
    )

    # Generate split indices for half-set resolution
    real_images = image_batch.ensure_real()
    split_indices = (
        get_half_set_indices(num_images=image_batch.n_images, device=image_batch.device)
        if options.reconstruction
        else None
    )

    # Parse and prepare target torch masks for weight aggregation in real and fourier space
    torch_masks = {}
    # If a mask has been provided for real space images and no weight mask is provided, use that
    if masks_dict is None:
        if mask is not None and mask.ndim == 2:
            torch_masks[ImageSpace.REAL] = torch.from_numpy(mask).to(real_images.device)
    else:
        for space, m in masks_dict.items():
            if m is not None:
                torch_masks[space] = (
                    torch.from_numpy(m).to(real_images.device)
                    if isinstance(m, np.ndarray)
                    else m.to(real_images.device)
                )
        # Use real-space image mask as weight mask as fallback
        if masks_dict.get(ImageSpace.REAL, None) is None:
            if mask is not None and mask.ndim == 2:
                torch_masks[ImageSpace.REAL] = torch.from_numpy(mask).to(
                    real_images.device
                )

    all_results = []
    for method_name, run in results.items():
        result = run.result

        # Initialize results to None, since some may not be computed
        reconstruction_metrics = None
        gt_frc_data = None
        hs_frc_data = None
        estimated_img = None

        if options.reconstruction or options.store_estimated_images:
            estimated_img = result.average.detach().cpu().numpy()

            if reapply_mask:
                estimated_img *= mask

        if options.reconstruction:
            comparison_ground_truth = (
                ground_truth_img * mask
                if reapply_mask and ground_truth_img is not None
                else ground_truth_img
            )

            reconstruction_metrics, gt_frc_data, hs_frc_data = (
                compute_reconstruction_metrics(
                    comparison_ground_truth,
                    estimated_img,
                    frc_thresholds=frc_thresholds,
                    image_batch=image_batch,
                    method_name=method_name,
                    method_run=run,
                    split_indices=split_indices,
                    pixel_size=pixel_size,
                    reapply_mask=reapply_mask,
                    mask=mask,
                    independent_half_sets=independent_half_sets,
                )
            )

        aggregated_weights = {}

        if options.scores or options.classification:
            aggregated_weights = compute_aggregated_weights(
                weights=result.weights,
                real_agg_strategies=real_agg_strategies,
                fourier_agg_strategies=fourier_agg_strategies,
                ref_real=ref_real,
                ref_fourier=ref_fourier,
                masks_dict=torch_masks,
            )

        space_metrics = None

        if options.classification and labels is not None:
            space_metrics = compute_classification_metrics(
                agg_weights=aggregated_weights,
                labels=labels,
                recall_methods=recall_methods,
            )

        fourier_ring_metrics: dict[ImageSpace, dict[int, ClassificationMetrics]] = {}

        if options.fourier_ring_metrics and labels is not None:
            # Classification metrics per ring for Fourier spaces
            for space in [ImageSpace.FOURIER_REAL, ImageSpace.FOURIER_IMAG]:
                w = result.weights.select_space(space)

                if w is not None and w.shape[-1] > 1:
                    fourier_ring_metrics[space] = (
                        compute_fourier_ring_classification_metrics(
                            fourier_weights=w,
                            labels=labels,
                            recall_methods=recall_methods,
                        )
                    )

        method_metrics = MethodMetrics(
            reconstruction_metrics=reconstruction_metrics, space_metrics=space_metrics
        )
        all_results.append(
            MethodEvaluation(
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
