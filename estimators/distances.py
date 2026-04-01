### Distance functions for GMM

import torch
from functools import partial
from typing import Tuple, Callable


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


FUNCTION_REGISTRY = {
    "l2": l2_norm,
    "l1": l1_norm,
    "lp": lp_norm,
    "l1_and_l2": l1_and_l2_norm,
}


def get_distance_function(name: str, params: dict | Tuple) -> Callable:
    try:
        base_function = FUNCTION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown function name: {name}")

    return partial(base_function, **dict(params))
