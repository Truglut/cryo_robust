"""Distance/dissimilarity functions used by GMM-based estimators."""
from __future__ import annotations

from typing import Callable

import torch

from estimators.weights import (
    tagare_weights,
    cosine_similarity,
    cross_correlation,
    cc_tagare_weights,
    configured_function,
)

DistanceFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def l2_norm(
    images: torch.Tensor, reference: torch.Tensor, std: torch.Tensor | float = 1
) -> torch.Tensor:
    """
    Calculates the L2 norm of the difference between each image and the reference,
    normalized by the number of pixels (Root Mean Squared Error / RMSE).

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w)
    reference : torch.Tensor
        Reference tensor of shape (h, w)
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0.
        Can be a float for global image normalization or a (h, w) tensor for
        per-pixel normalization.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) that contains the normalized L2 distance (MSE)
        from each image to the reference.
    """
    return ((reference - images) / std).square_().mean(dim=(1, 2)).sqrt_()


@torch.no_grad()
def l1_norm(
    images: torch.Tensor, reference: torch.Tensor, std: torch.Tensor | float = 1
) -> torch.Tensor:
    """
    Calculates L1 norm of the difference between each image and the reference,
    normalized by the number of pixels (so essentially MAE).

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w)
    reference : torch.Tensor
        Reference tensor of shape (h, w)
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0.
        Can be a float for global image normalization or a (h, w) tensor for
        per-pixel normalization.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) containing the normalized L1 distance (MAE)
        from each image to the reference.
    """
    return ((reference - images) / std).abs().mean(dim=(1, 2))


@torch.no_grad()
def lp_norm(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1,
    p: float = 2,
) -> torch.Tensor:
    """
    Calculates the Lp norm of the difference between each image and the reference,
    normalized by the number of pixels.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w)
    reference : torch.Tensor
        Reference tensor of shape (h, w)
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0.
        Can be a float for global image normalization or a (h, w) tensor for
        per-pixel normalization.
    p: float, optional
        Exponent for the Lp norm, by default 2.
        Should be a strictly positive value.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) that contains the normalized Lp distance
        from each image to the reference.
    """
    return (
        ((reference - images) / std)
        .abs()
        .float_power_(p)
        .mean(dim=(1, 2))
        .float_power_(1.0 / p)
    )


@torch.no_grad()
def tagare_distance(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Calculates the distance based on Tagare weights (similarity) between the
    reference and each of the input images.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0.
    beta : float, optional
        Scale parameter for the Tagare weight calculation, by default 1.0e-6.
    eps : float, optional
        Small constante for numerical stability to prevent division by zero, by default 1.0e-6
    inv_type : str, optional
        The inversion method used to transform the similarity metric into a distance
        metric, by default "neg". Options include "neg", "reciprocal", "neg_exp", or None.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) containing the inverted Tagare similarity distances.
    """

    # Get Tagare weights with shape (n, 1, 1)
    weights = tagare_weights(images, reference, std, beta, eps).view(-1)

    # Invert similarity
    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def cosine_similarity_dist(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Computes the cosine similarity between the input images and the reference image,
    and converts it into a distance metric based on the inversion type.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0.
    eps : float, optional
        Small constant for numerical stability in cosine calculation, by default 1.0e-8.
    inv_type : str | None, optional
        Method used to invert the similarity into a distance metric, by default "neg".

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) containing the distance derived from cosine similarity.
    """
    cos_sim = cosine_similarity(images, reference, std, eps).view(-1)
    return invert_similarity(cos_sim, inv_type=inv_type, eps=eps)


def cross_correlation_dist(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Computes the cross-correlation between the input images and the reference image,
    and converts it into a distance metric.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0.
    eps : float, optional
        Small constant for numerical stability, by default 1.0e-8.
    inv_type : str | None, optional
        Method used to invert the similarity into a distance metric, by default "neg".

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) containing the distance derived from cross-correlation.
    """
    cc = cross_correlation(images, reference, std, eps).view(-1)
    return invert_similarity(cc, inv_type=inv_type, eps=eps)


@torch.no_grad()
def cross_correlation_tagare(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Computes Tagare weights combined with cross-correlation metrics between the
    input images and the reference, returning an inverted distance metric.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0.
    beta : float, optional
        Scale parameter for the calculation, by default 1.0e-6.
    eps : float, optional
        Small constant for numerical stability, by default 1.0e-6.
    inv_type : str | None, optional
        Method used to invert the similarity into a distance metric, by default "neg".

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) containing the cross-correlation Tagare distance.
    """
    weights = cc_tagare_weights(images, reference, std, beta, eps).view(-1)

    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def orthogonal_residual_norm(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Calculates the squared norm of the orthogonal residual component of the input
    images when projected onto the space spanned by the reference image.

    Parameters
    ----------
    y : torch.Tensor
        Images tensor of shape (n, h, w).
    ref_image : torch.Tensor
        Reference tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0. 
        Unused here, kept for interface consistency.
    eps : float, optional
        Small constant for numerical stability in cosine similarity, by default 1.0e-6.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) containing the squared norm of the orthogonal residuals.
    """
    images_flat = images.flatten(1)
    reference_flat = reference.flatten()

    cos_sim = torch.cosine_similarity(images_flat, reference_flat, dim=1, eps=eps)
    squared_l2_norms = torch.linalg.vector_norm(images_flat, dim=1).square_()

    return squared_l2_norms * (1.0 - cos_sim.square_())


def negexp_orthogonal_residual_norm(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 5.0e-6,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Calculates a negative exponential transformation of the orthogonal residual norm,
    subsequently inverted to match the required distance formulation.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference tensor of shape (h, w).
    std : torch.Tensor | float, optional
        Standard deviation of the images, by default 1.0.
    beta : float, optional
        Exponential scaling factor, by default 5.0e-6.
    eps : float, optional
        Small constant for numerical stability, by default 1.0e-6.
    inv_type : str | None, optional
        Method used to invert the similarity into a distance metric, by default "neg".

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) containing the transformed orthogonal residual norm distance.
    """
    residual_similiarity = torch.exp_(
        orthogonal_residual_norm(images, reference, std, eps).mul_(-beta)
    )
    return invert_similarity(residual_similiarity, inv_type=inv_type, eps=eps)


def invert_similarity(
    similarity: torch.Tensor,
    inv_type: str | None = "neg",
    eps: float = 1e-6,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Inverts a similarity metric into a distance/dissimilarity metric using the
    specified strategy.

    Parameters
    ----------
    similarity : torch.Tensor
        Similarity scores tensor.
    inv_type : str | None, optional
        Inversion type strategy:
        - "neg" / "negative": returns -similarity
        - "reciprocal": returns 1 / similarity
        - "negative_exponential" / "neg_exp" / "negexp": returns exp(-similarity)
        - "none" / None: returns similarity unmodified
        By default "neg".
    eps : float, optional
        Lower bounding clamp value for 'reciprocal' to avoid zero division, by default 1.0e-6.
    inplace : bool, optional
        If True, performs operations in-place on the tensor to save memory, by default True.

    Returns
    -------
    torch.Tensor
        The inverted similarity values acting as a distance/dissimilarity metric.

    Raises
    -------
    ValueError
        If an unrecognized `inv_type` string is provided.
    """
    if inv_type is None or inv_type.lower() == "none":
        return similarity

    inv_type = inv_type.lower()
    if inv_type in ["neg", "negative"]:
        return similarity.neg_() if inplace else similarity.neg()
    
    if inv_type == "reciprocal":
        return (
            similarity.clamp_(min = eps).reciprocal_()
            if inplace
            else torch.reciprocal(torch.clamp(similarity, min=eps))
        )
    
    if inv_type in ["negative_exponential", "neg_exp", "negexp"]:
        return similarity.neg_().exp_() if inplace else torch.exp(-similarity)
    
    raise ValueError(f"Unknown similarity inversion strategy: {inv_type}")

# Global configuration key mappings
DISTANCE_FUNCTION_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "l2": l2_norm,
    "l1": l1_norm,
    "lp": lp_norm,
    "tagare_weights": tagare_distance,
    "cosine_similarity": cosine_similarity_dist,
    "cross_correlation": cross_correlation_dist,
    "cross_correlation_tagare": cross_correlation_tagare,
    "orthogonal_residual_norm": orthogonal_residual_norm,
    "negexp_orthogonal_residual_norm": negexp_orthogonal_residual_norm,
}


def get_distance_function(
    name: str, params: dict | tuple, imgs: torch.Tensor | None = None
) -> DistanceFunction:
    """
    Factory function to retrieve and pre-configure a specific distance function via partial evaluation.

    Parameters
    ----------
    name : str
        The registration key name of the distance metric (e.g., 'l2', 'tagare_weights').
    params : dict | tuple
        Hyperparameters passed down to the base function. If a tuple of pairs is given,
        it is transformed into a dictionary seamlessly.
    imgs : torch.Tensor | None, optional
        A collection of images tensor used exclusively for calculating automated beta scaling
        parameter if params configures it to "auto", by default None.

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        A standard configured partial function expecting `(images, reference)` arguments.

    Raises
    -------
    ValueError
        If the name is not found within `FUNCTION_REGISTRY` or if 'auto' beta scaling
        is requested without passing the backing `imgs` tensor.
    """
    return configured_function(DISTANCE_FUNCTION_REGISTRY, name, params, imgs)
