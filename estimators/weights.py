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
def global_weights(y, x, std, beta, eps=1e-8):
    n = y.shape[0]

    # Dot product
    dot = (y * x).sum(dim=(1, 2))  # (n,)

    # Squared norms
    y_norm_sq = (y * y).sum(dim=(1, 2))  # (n,)
    x_norm_sq = (x * x).sum()  # scalar

    # First term: absolute cosine
    # Protect the denominator of the cosine by combining both norms with clamp
    denom = torch.clamp(y_norm_sq * x_norm_sq, min=eps).sqrt()
    cos_abs = dot.abs() / denom

    # Second term: norm of the orthogonal component
    # Protect the division by adding eps to x_norm_sq
    x_norm_sq_safe = torch.clamp(x_norm_sq, min=eps)
    orth_norm_sq = y_norm_sq - (dot**2) / x_norm_sq_safe

    # Avoid negative values caused by floating point errors
    orth_norm_sq = torch.clamp(orth_norm_sq, min=0.0)

    term2 = torch.exp(-beta * orth_norm_sq)

    return (cos_abs * term2).reshape(n, 1, 1)


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
    "global": global_weights,
    "cauchy": cauchy_weights,
    "student": student_weights,
    "q_norm": q_norm_weights,
}


def get_weight_function(name: str, params: dict, imgs: torch.Tensor | None = None):
    try:
        base_function = FUNCTION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown function name: {name}")

    # Calculate automatic beta parameter for tagare weights
    if name == "global" and (params.get("beta", "auto") == "auto"):
        if imgs is None:
            raise ValueError("Cannot calculate auto beta without images")
        mult = params.get("auto_multiplier", 1)
        beta = calculate_beta_auto(imgs, mult)

        # Update params
        params["beta"] = beta

        # Print calculated parameter
        print(f"Auto-calculated beta parameter: {beta = }")

    return partial(base_function, **params)


TAGARE_CONSTANT = 1.0e-5


def calculate_beta_auto(imgs: torch.Tensor, mult: float = 1.0):
    return mult * TAGARE_CONSTANT / imgs.var(dim=(1, 2)).mean().item()
