from functools import partial

import torch


@torch.no_grad()
def huber_weights(y, x, std, delta, sigma_f=1):
    residuals = sigma_f * (y - x) / std
    abs_res = residuals.abs()

    weights = torch.ones_like(residuals, dtype=torch.float32)
    mask = abs_res > delta

    weights[mask] = delta / abs_res[mask]

    return weights


@torch.no_grad()
def smooth_redescending_weights(y, x, std, delta, sigma_f=1, normalise=True):
    residuals = sigma_f * (y - x) / (std + 1e-8)
    if not normalise:
        return (2 / (delta**2)) * torch.exp(-torch.square(residuals) / (delta**2))
    return torch.exp(-torch.square(residuals) / (delta**2))


@torch.no_grad()
def tagare_weights(
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    y_flat = y.flatten(1)
    ref_image_flat = ref_image.flatten()

    # First term: absolute cosine
    cos_abs = torch.abs(torch.cosine_similarity(y_flat, ref_image_flat, dim=1, eps=eps))

    # Second term: norm of the orthogonal component
    orth_norm_sq = y_flat.square().sum(dim=1) * (1.0 - cos_abs.square())

    # Avoid negative values caused by floating point errors
    orth_norm_sq = torch.clamp(orth_norm_sq, min=0.0)

    return (cos_abs * torch.exp(-beta * orth_norm_sq)).view(-1, *([1] * ref_image.ndim))


# @torch.no_grad()
# def global_weights(y, x, std, beta, eps=1e-8):
#     n = y.shape[0]

#     # Dot product
#     dot = (y * x).sum(dim=(1, 2))  # (n,)

#     # Squared norms
#     y_norm_sq = (y * y).sum(dim=(1, 2))  # (n,)
#     x_norm_sq = (x * x).sum()  # scalar

#     # First term: absolute cosine
#     # Protect the denominator of the cosine by combining both norms with clamp
#     denom = torch.clamp(y_norm_sq * x_norm_sq, min=eps).sqrt()
#     cos_abs = dot.abs() / denom

#     # Second term: norm of the orthogonal component
#     # Protect the division by adding eps to x_norm_sq
#     x_norm_sq_safe = torch.clamp(x_norm_sq, min=eps)
#     orth_norm_sq = y_norm_sq - (dot**2) / x_norm_sq_safe

#     # Avoid negative values caused by floating point errors
#     orth_norm_sq = torch.clamp(orth_norm_sq, min=0.0)

#     term2 = torch.exp(-beta * orth_norm_sq)

#     return (cos_abs * term2).reshape(n, 1, 1)


def cosine_similarity(
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    eps: float = 1.0e-8,
):
    return torch.cosine_similarity(
        y.flatten(1), ref_image.flatten(), dim=1, eps=eps
    ).view(-1, *([1] * ref_image.ndim))


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
    y: torch.Tensor,
    ref_image: torch.Tensor,
    std: torch.Tensor | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    y_flat = y.flatten(1)

    # First term: replace cosine similarity with cross correlation
    corr_abs = torch.abs(
        cross_correlation(y, reference=ref_image, std=std, eps=eps)
    ).view(-1)

    # Second term: norm of the orthogonal component
    orth_norm_sq = y_flat.square().sum(dim=1) * (1.0 - corr_abs.square())

    # Avoid negative values caused by floating point errors
    orth_norm_sq = torch.clamp(orth_norm_sq, min=0.0)

    return (corr_abs * torch.exp(-beta * orth_norm_sq)).view(
        -1, *([1] * ref_image.ndim)
    )


@torch.no_grad()
def cauchy_weights(y, x, std, c):
    return 1 / (1 + ((y - x) / (c * std)).square())


@torch.no_grad()
def student_weights(y, x, std, df, sigma_f=1):
    return (df + 1) / (df + (sigma_f * (y - x) / std).square())


@torch.no_grad()
def q_norm_weights(y, x, std, q, sigma_f=1):
    return (sigma_f * (y - x) / std).abs().clamp(min=1).pow(q - 2)


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
