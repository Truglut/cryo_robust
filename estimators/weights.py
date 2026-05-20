from functools import partial

import torch


@torch.no_grad()
def weighted_average(
    y: torch.Tensor, weights: torch.Tensor, dim: int = 0, eps: float = 1.0e-8
) -> torch.Tensor:
    return (weights * y).sum(dim=dim) / (weights.sum(dim=0) + eps)


@torch.no_grad()
def huber_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float,
    delta: float,
    sigma_f: float = 1.0,
    eps: float = 1.0e-8,
):
    abs_residuals = torch.abs(sigma_f * (images - reference) / (std + eps))

    weights = torch.ones_like(abs_residuals, dtype=torch.float32)
    mask = abs_residuals > delta

    weights[mask] = delta / abs_residuals[mask]

    return weights


@torch.no_grad()
def smooth_redescending_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: float | torch.Tensor,
    delta: float,
    sigma_f: float = 1.0,
    normalise: bool = True,
    eps: float = 1.0e-8,
):
    abs_residuals = torch.abs(sigma_f * (images - reference) / (std + eps))
    if not normalise:
        return (2 / (delta**2)) * torch.exp(-torch.square(abs_residuals) / (delta**2))
    return torch.exp(-torch.square(abs_residuals) / (delta**2))


@torch.no_grad()
def tagare_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    y_flat = images.flatten(1)
    ref_image_flat = reference.flatten()

    # First term: absolute cosine
    cos_abs = torch.abs(torch.cosine_similarity(y_flat, ref_image_flat, dim=1, eps=eps))

    # Second term: norm of the orthogonal component
    orth_norm_sq = y_flat.square().sum(dim=1) * (1.0 - cos_abs.square())

    # Avoid negative values caused by floating point errors
    orth_norm_sq = torch.clamp(orth_norm_sq, min=0.0)

    return (cos_abs * torch.exp(-beta * orth_norm_sq)).view(-1, *([1] * reference.ndim))


def cosine_similarity(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
):
    return torch.cosine_similarity(
        images.flatten(1), reference.flatten(), dim=1, eps=eps
    ).view(-1, *([1] * reference.ndim))


def cross_correlation(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
):
    image_dims = tuple(range(1, images.ndim))
    return cosine_similarity(
        images - images.mean(dim=image_dims, keepdim=True),
        reference - reference.mean(),
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
    y_flat = images.flatten(1)

    # First term: replace cosine similarity with cross correlation
    corr_abs = torch.abs(
        cross_correlation(images, reference=reference, std=std, eps=eps)
    ).view(-1)

    # Second term: norm of the orthogonal component
    orth_norm_sq = y_flat.square().sum(dim=1) * (1.0 - corr_abs.square())

    # Avoid negative values caused by floating point errors
    orth_norm_sq = torch.clamp(orth_norm_sq, min=0.0)

    return (corr_abs * torch.exp(-beta * orth_norm_sq)).view(
        -1, *([1] * reference.ndim)
    )


@torch.no_grad()
def cauchy_weights(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: torch.Tensor | float,
    c: float,
    eps: float = 1.0e-8,
):
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
):
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
):
    abs_residuals = torch.abs(sigma_f * (images - reference) / (std + eps))
    return abs_residuals.clamp(min=1).pow(q - 2)


FUNCTION_REGISTRY = {
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


def get_weight_function(name: str, params: dict, imgs: torch.Tensor | None = None):
    try:
        base_function = FUNCTION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown function name: {name}")
    params = params.copy()

    # Calculate automatic beta parameter for tagare weights
    if name in ["global", "cc_tagare"] and (params.get("beta", "auto") == "auto"):
        if imgs is None:
            raise ValueError("Cannot calculate auto beta without images")
        mult = params.pop("auto_multiplier", 1.0)
        beta = calculate_beta_auto(imgs, mult)

        # Update params
        params["beta"] = beta

        # Print calculated parameter
        print(f"Auto-calculated beta parameter: {beta = }")

    return partial(base_function, **params)


TAGARE_CONSTANT = 1.0e-5


def calculate_beta_auto(imgs: torch.Tensor, mult: float = 1.0):
    return mult * TAGARE_CONSTANT / imgs.var(dim=(1, 2)).mean().item()
