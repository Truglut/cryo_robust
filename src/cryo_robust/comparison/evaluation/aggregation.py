from typing import Callable, Iterable
import warnings

import numpy as np
import torch

from cryo_robust.comparison.domain.enums import AggregationStrategy, ImageSpace


def mean_aggregate(
    weights: torch.Tensor,
    reference: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    if mask is not None:
        try:
            mask = mask.to(dtype=weights.dtype, device=weights.device)
            # Apply mask and normalize over the total valid mask area
            numerator = torch.sum(weights * mask, dim=(-2, -1))
            denominator = mask.sum(dim=(-2, -1)) + 1.0e-12
            return (numerator / denominator).detach().cpu().numpy()
        except:
            warnings.warn(
                "Weight masking failed, proceeding with unmasked mean aggregation."
            )
            pass

    return weights.mean(dim=(-2, -1)).detach().cpu().numpy()


def energy_aggregate(
    weights: torch.Tensor,
    reference: torch.Tensor | None,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    if reference is None:
        raise ValueError("Energy aggregation requires a reference image.")

    # Calculate signal energy from reference (normalized)
    energy = torch.abs(reference) ** 2

    if mask is not None:
        mask = mask.to(dtype=energy.dtype, device=energy.device)
        energy = energy * mask

    energy = energy / (energy.sum() + 1.0e-12)

    # Energy-weighted average
    scores = torch.sum(weights * energy, dim=(1, 2))
    return scores.detach().cpu().numpy()


AGGREGATORS: dict[
    AggregationStrategy, Callable[[torch.Tensor, torch.Tensor | None], np.ndarray]
] = {
    AggregationStrategy.MEAN: mean_aggregate,
    AggregationStrategy.ENERGY: energy_aggregate,
}


def aggregate_weights(
    weights: torch.Tensor,
    strategy: AggregationStrategy = AggregationStrategy.MEAN,
    reference: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Aggregates per-pixel weights into a single score per image.
    Strategies:
      - "mean": Standard global average across spatial dimensions.
      - "energy": Uses the reference image to weight the pixels by signal energy.
    """
    # If weights are already per-image (N,) or (N, 1, 1)
    if weights.ndim == 1 or (weights.ndim == 3 and weights.shape[1:] == (1, 1)):
        return weights.cpu().numpy().flatten()

    return AGGREGATORS[strategy](weights, reference, mask)


def setup_energy_reference(
    ground_truth_img: np.ndarray | None,
    images_dict: dict[ImageSpace, torch.Tensor],
    energy_reference: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct the real- and Fourier-space reference tensors for energy aggregation.

    Parameters
    ----------
    ground_truth_img : np.ndarray or None
        Ground truth image array, shape `(H, W)`.
        `None` if ground truth is not available.
    images_dict : dict of {Space: torch.Tensor}
        Image tensors keyed by space; used to infer the target device.
    energy_reference : {"ground_truth", "global_avg"}
        Strategy for building the real-space reference.  `"ground_truth"`
        uses the known clean image; `"global_avg"` uses the mean of all
        input images.

    Returns
    -------
    ref_real : torch.Tensor
        Real-space reference image, shape `(H, W)`.
    ref_fourier : torch.Tensor
        Fourier transform of `ref_real` via `torch.fft.rfft2`, shape
        `(H, W//2 + 1)`.

    Raises
    ------
    ValueError
        If `energy_reference` is `'ground_truth'` but `ground_truth_img` is `None`.
        If `energy_reference` is not one of the accepted values.
    """
    if ground_truth_img is not None:
        gt_tensor = torch.from_numpy(ground_truth_img).to(
            dtype=torch.float32, device=images_dict[ImageSpace.REAL].device
        )
    elif energy_reference == "ground_truth":
        raise ValueError(
            "ground_truth energy reference requested, but ground_truth_img is None"
        )

    if energy_reference == "ground_truth":
        ref_real = gt_tensor
    elif energy_reference == "global_avg":
        ref_real = images_dict[ImageSpace.REAL].mean(dim=0)
    else:
        raise ValueError(
            "energy_reference must be one of 'ground_truth' or 'global_avg'"
        )
    ref_fourier = torch.fft.rfft2(ref_real, norm="ortho")

    return ref_real, ref_fourier


def _get_space_reference(
    space: ImageSpace,
    ref_real: torch.Tensor,
    ref_fourier: torch.Tensor,
) -> torch.Tensor | None:
    """
    Return the reference tensor and aggregation strategies for a given space.

    Parameters
    ----------
    space : Space
        The weight space to look up.
    ref_real : torch.Tensor
        Real-space reference tensor.
    ref_fourier : torch.Tensor
        Complex Fourier-space reference tensor.

    Returns
    -------
    torch.Tensor or None
        The appropriate reference slice, or `None` if the space is not
        handled or no reference is available.
    """
    if space == ImageSpace.REAL:
        return ref_real  # may be None for unlabeled
    elif space == ImageSpace.FOURIER_REAL:
        return ref_fourier.real if ref_fourier is not None else None
    elif space == ImageSpace.FOURIER_IMAG:
        return ref_fourier.imag if ref_fourier is not None else None
    return None


def compute_aggregated_weights(
    weights_dict: dict[ImageSpace, torch.Tensor | None],
    real_agg_strategies: Iterable[AggregationStrategy],
    fourier_agg_strategies: Iterable[AggregationStrategy],
    ref_real: torch.Tensor | None = None,
    ref_fourier: torch.Tensor | None = None,
    masks_dict: dict[ImageSpace, torch.Tensor | None] | None = None,
) -> dict[ImageSpace, dict[AggregationStrategy, np.ndarray]]:
    """Aggregate per-image weights into scalar scores for all spaces and strategies.

    Parameters
    ----------
    weights_dict : dict of {Space: torch.Tensor or None}
        Raw weight tensors keyed by space, as returned by the estimator.
    real_agg_strategies : iterable of AggregationStrategy
        Aggregation strategies to apply to real-space weights.
    fourier_agg_strategies : iterable of AggregationStrategy
        Aggregation strategies to apply to Fourier-space weights.
    ref_real : torch.Tensor or None, optional
        Real-space reference tensor, required if `"energy"` is among
        `real_agg_strategies`. Default is `None`.
    ref_fourier : torch.Tensor or None, optional
        Complex Fourier reference tensor, required if `"energy"` is among
        `fourier_agg_strategies`. Default is `None`.
    masks_dict : dict[Space, torch.Tensor | None] or None, optional
        Dict mapping each Space to the mask that should be used for weight
        aggregation.

    Returns
    -------
    dict of {Space: dict of {AggregationStrategy: np.ndarray}}
        Aggregated scores keyed by space then strategy. Spaces with
        `None` weights or no matching reference are omitted.
    """
    scores: dict[ImageSpace, dict[str, np.ndarray]] = {}

    for space, weights in weights_dict.items():
        if weights is None:
            continue
        ref = _get_space_reference(space, ref_real, ref_fourier)
        strategies = (
            real_agg_strategies if space == ImageSpace.REAL else fourier_agg_strategies
        )

        # Extract space-specific mask if it exists
        mask = masks_dict.get(space) if masks_dict is not None else None

        scores[space] = {
            strategy: aggregate_weights(
                weights, strategy=strategy, reference=ref, mask=mask
            )
            for strategy in strategies
            if not (
                ref is None and strategy == "energy"
            )  # energy strategy requested but no reference available; skip
        }

    return scores
