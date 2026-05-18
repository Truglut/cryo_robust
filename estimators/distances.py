### Distance functions for GMM
from functools import partial
from typing import Callable

import torch

from estimators.weights import (
    calculate_beta_auto,
    tagare_weights,
    cosine_similarity,
    cross_correlation,
    cc_tagare_weights,
)


@torch.no_grad()
def l2_norm(
    y: torch.Tensor, ref_image: torch.Tensor, std: torch.Tensor | float = 1
) -> torch.Tensor:
    return ((ref_image - y) / std).square().mean(dim=(1, 2))


@torch.no_grad()
def l1_norm(
    y: torch.Tensor, ref_image: torch.Tensor, std: torch.Tensor | float = 1
) -> torch.Tensor:
    return ((ref_image - y) / std).abs().mean(dim=(1, 2))


@torch.no_grad()
def lp_norm(
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1,
    p: float = 2,
) -> torch.Tensor:
    return ((ref_image - y) / std).abs().float_power(p).mean(dim=(1, 2))


@torch.no_grad()
def l1_and_l2_norm(
    y: torch.Tensor, ref_image: torch.Tensor, std: torch.Tensor | float = 1.0
) -> torch.Tensor:
    return torch.stack([l1_norm(y, ref_image, std), l2_norm(y, ref_image, std)], dim=1)


@torch.no_grad()
def tagare_distance(
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
) -> torch.Tensor:

    weights = tagare_weights(y, ref_image, std, beta, eps).view(-1)

    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def cosine_similarity_dist(
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
    inv_type: str = "neg",
):
    cos_sim = cosine_similarity(y, ref_image, std, eps).view(-1)
    return invert_similarity(cos_sim, inv_type=inv_type, eps=eps)


def cross_correlation_dist(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
    inv_type: str = "neg",
):
    cc = cross_correlation(images, reference, std, eps).view(-1)
    return invert_similarity(cc, inv_type=inv_type, eps=eps)


@torch.no_grad()
def cross_correlation_tagare(
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
) -> torch.Tensor:
    weights = cc_tagare_weights(y, ref_image, std, beta, eps).view(-1)

    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def orthogonal_residual_norm(
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-6,
):
    y_flat = y.flatten(1)
    ref_flat = ref_image.flatten()
    cos_sim = torch.cosine_similarity(y_flat, ref_flat, dim=1, eps=eps)

    return y_flat.square().sum(dim=1) * (1 - cos_sim.square())


def negexp_orthogonal_residual_norm(
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 5.0e-6,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
):
    return invert_similarity(
        torch.exp(-beta * orthogonal_residual_norm(y, ref_image, std, eps)),
        inv_type=inv_type,
        eps=eps,
    )


def invert_similarity(
    sim: torch.Tensor, inv_type: str | None = "neg", eps: float = 1e-6
):
    if inv_type is None:
        return sim

    inv_type = inv_type.lower()
    if inv_type in ["neg", "negative"]:
        return -sim
    if inv_type == "reciprocal":
        return torch.reciprocal(torch.clamp(sim, min=eps))
    if inv_type in ["negative_exponential", "neg_exp", "negexp"]:
        return torch.exp(-sim)
    if inv_type == "none":
        return sim
    raise ValueError(f"Unrecognized inversion type in tagare_distance: {inv_type}")


FUNCTION_REGISTRY = {
    "l2": l2_norm,
    "l1": l1_norm,
    "lp": lp_norm,
    "l1_and_l2": l1_and_l2_norm,
    "tagare_weights": tagare_distance,
    "cosine_similarity": cosine_similarity_dist,
    "cross_correlation": cross_correlation_dist,
    "cross_correlation_tagare": cross_correlation_tagare,
    "orthogonal_residual_norm": orthogonal_residual_norm,
    "negexp_orthogonal_residual_norm": negexp_orthogonal_residual_norm,
}

NEED_BETA_PARAMETER = [
    "global",  # just in case
    "tagare_weights",
    "negexp_orthogonal_residual_norm",
    "cross_correlation_tagare",
]


def get_distance_function(
    name: str, params: dict | tuple, imgs: torch.Tensor | None = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    try:
        base_function = FUNCTION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown function name: {name}")
    params = params.copy()

    # Calculate automatic beta parameter for tagare distance
    if name in NEED_BETA_PARAMETER and (params.get("beta", "auto") == "auto"):
        if imgs is None:
            raise ValueError("Cannot calculate auto beta without images")
        mult = params.pop("auto_multiplier", 1.0)
        beta = calculate_beta_auto(imgs, mult)

        # Update params
        params["beta"] = beta

        # Print calculated parameter
        print(f"Auto-calculated beta parameter: {beta = }")

    return partial(base_function, **dict(params))
