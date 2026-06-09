## Robust weighting functions for M-estimators
from __future__ import annotations

from functools import partial
from typing import Callable

import torch

TAGARE_CONSTANT: float = 1.0e-5
WeightFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | float], torch.Tensor
]


@torch.no_grad()
def weighted_average(
    images: torch.Tensor, weights: torch.Tensor, dim: int = 0, eps: float = 1.0e-6
) -> torch.Tensor:
    """
    Computes the weighted average of an input image tensor along a specified dimension.

    Parameters
    ----------
    images : torch.Tensor
        Input data tensor containing image components or batches.
    weights : torch.Tensor
        Weight coefficients matching or broadcastable to the shape of `images`.
    dim : int, optional
        The dimension along which the average is computed, by default 0.
    eps : float, optional
        Small constant for numerical stability to avoid zero-division, by default 1.0e-6.

    Returns
    -------
    torch.Tensor
        The resulting weighted average tensor.

    Raises
    ------
    ValueError
        If the maximum weight sum is effectively zero, indicating degenerate metrics.
    """
    weight_sum = weights.sum(dim=0)
    if weight_sum.max() < 1e-8:
        raise ValueError(
            "All weights are effectively zero — distances likely degenerate"
        )
    return (weights * images).sum(dim=dim) / (weight_sum + eps)


@torch.no_grad()
def huber_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float,
    delta: float,
    sigma_f: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Calculates robust M-estimator weight updates based on the Huber loss criterion.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float
        Standard deviation scale factor for normalizing residuals.
    delta : float
        The clipping threshold separating L2-like and L1-like optimization regions.
    sigma_f : float, optional
        Global scaling factor applied to residuals, by default 1.0.
    eps : float, optional
        Small stabilization constant, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, h, w) containing Huber sample weights.
    """
    abs_residuals = torch.abs(sigma_f * (images - reference) / (std + eps))
    return torch.where(abs_residuals > delta, delta / abs_residuals, 1.0)


@torch.no_grad()
def smooth_redescending_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: float | torch.Tensor,
    delta: float,
    sigma_f: float = 1.0,
    normalize: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Computes smooth redescending M-estimator weights using a Gaussian-like influence metric.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float
        Standard deviation scale factor.
    delta : float
        Tuning parameter governing the rejection scale threshold of outlier features.
    sigma_f : float, optional
        Residual modifier scale, by default 1.0.
    normalize : bool, optional
        If True, bounds the maximum weight value to 1.0, by default True.
    eps : float, optional
        Stabilizing denominator offset, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, h, w) containing the smooth redescending weights.
    """
    abs_residuals = torch.abs(sigma_f * (images - reference) / (std + eps))
    variance_scale = delta**2

    sq_residuals = abs_residuals.square_()
    scaled_exponent = sq_residuals.neg_().div_(variance_scale).exp_()

    if not normalize:
        return (2.0 / (variance_scale)) * scaled_exponent
    return scaled_exponent


@torch.no_grad()
def tagare_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Calculates non-local structural similarity weights matching the Huang-Tagare model.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation tracking parameter, by default 1.0 (retained for registry consistency).
    beta : float, optional
        Regularization penalty coefficient for orthogonal variance components, by default 1.0e-6.
    eps : float, optional
        Cosine similarity processing tolerance floor, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor reshaped to (n, 1, 1) to enable broadcasting back to spatial axes.
    """
    images_flat = images.flatten(1)
    reference_flat = reference.flatten()

    # First term: absolute cosine
    cos_abs = torch.abs(
        torch.cosine_similarity(images_flat, reference_flat, dim=1, eps=eps)
    )

    # Second term: norm of the orthogonal component
    image_norm_sq = torch.linalg.vector_norm(images_flat, dim=1).square_()
    orth_norm_sq = image_norm_sq * (1.0 - cos_abs.square_()).clamp_min_(0.0)

    weights = orth_norm_sq.mul_(-beta).exp_().mul_(cos_abs)
    return weights.view(-1, *([1] * reference.ndim))


def cosine_similarity(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Calculates vector-wise spatial cosine similarity arrays reshaped for automated
    dimension broadcasting.

    Parameters
    ----------
    images : torch.Tensor
        Input images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation tracking parameter, by default 1.0.
        Unused; kept for interface consistency.
    eps : float, optional
        Numerical stability factor avoiding zero-magnitude vectors during calculation steps,
        by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, 1, 1) capturing calculated vector alignment similarity distributions.
    """
    return torch.cosine_similarity(
        images.flatten(1), reference.flatten(), dim=1, eps=eps
    ).view(-1, *([1] * reference.ndim))


def cross_correlation(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Evaluates zero-mean structural cross-correlation metric matrices broadcastable over
    matching framework axes.

    Parameters
    ----------
    images : torch.Tensor
        Input images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation parameter tracking scale boundaries, by default 1.0.
        Unused; kept for interface consistency.
    eps : float, optional
        Numerical stability threshold constant passed down to internal calculations, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, 1, 1) tracking the computed cross-correlation properties.
    """
    image_dims = tuple(range(1, images.ndim))
    centered_images = images - images.mean(dim=image_dims, keepdim=True)
    centered_reference = reference - reference.mean()

    return cosine_similarity(
        centered_images,
        centered_reference,
        std,
        eps,
    ).view(-1, *([1] * reference.ndim))


@torch.no_grad()
def cc_tagare_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Computes non-local structural similarity weights substituting standard cosine
    components with zero-mean cross-correlations.

    Parameters
    ----------
    images : torch.Tensor
        Input images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation tracking metadata parameter values, by default 1.0.
    beta : float, optional
        Scale coefficient governing exponential penalty sensitivity over orthogonal variance
        components, by default 1.0e-6.
    eps : float, optional
        Small calculation tolerance constant preventing division errors, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, 1, 1) holding computed the modified
        cross-correlation Tagare weights.
    """
    images_flat = images.flatten(1)

    # First term: replace cosine similarity with cross correlation
    corr_abs = torch.abs(
        cross_correlation(images, reference=reference, std=std, eps=eps)
    ).view(-1)

    # Second term: norm of the orthogonal component
    image_norm_sq = torch.linalg.vector_norm(images_flat, dim=1).square_()
    orth_norm_sq = image_norm_sq.mul_(1.0 - corr_abs.square_()).clamp_min_(0.0)

    weights = orth_norm_sq.mul_(-beta).exp_().mul_(corr_abs)
    return weights.view(-1, *([1] * reference.ndim))


@torch.no_grad()
def cauchy_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float,
    c: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Calculates heavy-tailed robust weights derived from a Cauchy distribution profile.
    Parameters
    ----------
    images : torch.Tensor
        Input images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float
        Standard deviation scale metrics utilized to normalize raw spatial residuals.
    c : float
        Scale tuning parameter adapting the distribution tail width and tuning outlier
        damping thresholds.
    eps : float, optional
        Small stabilizer tracking denominator operations, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, h, w) containing the computed Cauchy weights.
    """
    abs_residuals = torch.abs((images - reference) / (c * std + eps))
    return 1 / (1 + abs_residuals.square())


@torch.no_grad()
def student_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float,
    df: float,
    sigma_f: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Calculates robust weights based on Student's t-distribution influence curves.
    Parameters
    ----------
    images : torch.Tensor
        Input images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float
        Standard deviation metrics optimizing scaling constraints over raw spatial residuals.
    df : float
        Degrees of freedom parameter specifying the heavy-tailed shape properties
        of the target distribution.
    sigma_f : float, optional
        Residual modifier scale multiplier parameter, by default 1.0.
    eps : float, optional
        Numerical denominator stabilization constant, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, h, w) mapping Student's t distribution calculation outputs.
    """
    abs_residuals = torch.abs(sigma_f * (images - reference) / (std + eps))
    return (df + 1) / (df + abs_residuals.square())


@torch.no_grad()
def q_norm_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float,
    q: float,
    sigma_f: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Computes robust weight matrices matching minimization metrics for custom
    sub-Gaussian L_q norm losses.

    Parameters
    ----------
    images : torch.Tensor
        Input images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float
        Standard deviation matrix parameters optimizing scaling constraints.
    q : float
        The exponent degree metric defining the target sub-Gaussian L_q optimization objective.
    sigma_f : float, optional
        Residual tracking scale multiplier parameter, by default 1.0.
    eps : float, optional
        Small structural stabilization parameter constant, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, h, w) holding the corresponding L_q weight updates.
    """
    abs_residuals = torch.abs(sigma_f * (images - reference) / (std + eps))
    return abs_residuals.clamp_min_(min=1).pow_(q - 2)


# Global weight configuration mappings
FUNCTION_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "huber": huber_weights,
    "smooth": smooth_redescending_weights,
    "global": tagare_weights,
    "cosine": cosine_similarity,
    "correlation": cross_correlation,
    "cc_tagare": cc_tagare_weights,
    "cauchy": cauchy_weights,
    "student": student_weights,
    "q_norm": q_norm_weights,
}

# Set of functions that need the beta parameter, including distance functions
NEED_BETA_PARAMETER = (
    "global",
    "cc_tagare",
    "tagare_weights",
    "cross_correlation_tagare",
    "negexp_orthogonal_residual_norm",
)


def configured_function(
    registry: dict[str, Callable[..., torch.Tensor]],
    name: str,
    params: dict | tuple | None,
    images: torch.Tensor | None = None,
):
    """
    Retrieves function from registry and configures it with the given parameters
    through partial evaluation

    Parameters
    ----------
    registry : dict[str, Callable[..., torch.Tensor]]
        Function registry mapping names to functions
    name : str
        The name of the requested function. Should match a key in `registry`
    params : dict | tuple | None
        Specified parameters for the function.
    images : torch.Tensor | None, optional
        Images tensor, used to calculate the "auto" beta parameter, by default None

    Returns
    -------
    Callable
        The configured function

    Raises
    ------
    ValueError
        If `name` does not match any of the keys in `registry`
    ValueError
        If `beta="auto"` is requested but the images are not provided
    """
    try:
        fn = registry[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown function name: {name!r}. Available: {sorted(registry)}"
        ) from exc
    params_dict = dict(params or {})
    if name in NEED_BETA_PARAMETER and params_dict.get("beta", "auto") == "auto":
        if images is None:
            raise ValueError(
                f"Cannot calculate automatic beta for {name!r} without images."
            )
        multiplier = params_dict.pop("auto_multiplier", 1.0)
        params_dict["beta"] = calculate_beta_auto(images, multiplier)
    return partial(fn, **params_dict)


def get_weight_function(
    name: str, params: dict | tuple | None = None, imgs: torch.Tensor | None = None
) -> WeightFunction:
    """
    Factory configuration utility that builds specialized partial instances of weighting
    functions loaded with target parameter configurations.

    Parameters
    ----------
    name : str
        The registration key mapping string token corresponding to the robust weight function.
    params : dict | tuple
        Hyperparameters passed to define runtime properties. If a tuple of pairs is supplied,
        it is transformed into a standard dictionary.
    images : torch.Tensor | None, optional
        Contextual batch images dataset context tensor required to compute automated beta scale
        parameters if config indicates "beta": "auto", by default None.

    Returns
    -------
    Callable[..., torch.Tensor]
        A pre-configured robust weight calculation partial function instance.

    Raises
    ------
    ValueError
        If the name is missing from the registry, or if automated beta calculations are
        requested without passing an accompanying data dataset context tensor.
    """
    return configured_function(FUNCTION_REGISTRY, name, params, imgs)


def calculate_beta_auto(imgs: torch.Tensor, mult: float = 1.0) -> float:
    """
    Automatically scales the Tagare exponential scaling factor based on the average
    variance across the provided input images dataset.

    Parameters
    ----------
    images : torch.Tensor
        Input batch dataset images tensor of shape (n, h, w).
    mult : float, optional
        Scalar multiplier adjustment value modifying the baseline parameter scaling, by default 1.0.

    Returns
    -------
    float
        The calculated floating-point automatic beta scaling parameter.
    """
    return mult * TAGARE_CONSTANT / imgs.var(dim=(1, 2)).mean().item()
