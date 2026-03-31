import torch
from typing import Callable, Tuple
from dataclasses import dataclass


@dataclass
class IRLSConfig:
    images: torch.Tensor
    weight_function: Callable
    image_variance: torch.Tensor
    prior_mean: torch.Tensor
    prior_variance: torch.Tensor
    ctf: torch.Tensor


@torch.no_grad()
def irls_iteration(
    images: torch.Tensor,
    reference: torch.Tensor,
    weight_function: Callable,
    image_variance: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_variance: torch.Tensor | float,
    ctf: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a single iteration of the Reweighted Least Squares update.
    """

    weights = weight_function(images, reference, torch.sqrt(image_variance))

    s_1 = torch.sum(weights * ctf * images, dim=0)
    s_2 = torch.sum(weights * torch.square(ctf), dim=0)

    safe_variance = image_variance + 1e-8
    return (s_1 / safe_variance + prior_mean / prior_variance) / (
        s_2 / safe_variance + 1 / prior_variance
    ), weights


@torch.no_grad()
def irls_scheme(
    images: torch.Tensor,
    initial_reference: torch.Tensor,
    weight_function: Callable,
    image_variance: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_variance: torch.Tensor | float,
    ctf: torch.Tensor,
    max_iter: int,
    tol: float = 1e-6,
):
    """
    Executes the Iteratively Reweighted Least Squares (IRLS) optimization.
    """
    reference = initial_reference
    weights = None

    for _ in range(max_iter):
        next_reference, weights = irls_iteration(
            images,
            reference,
            weight_function,
            image_variance,
            prior_mean,
            prior_variance,
            ctf,
        )

        # Check stopping criterion
        if torch.linalg.norm(next_reference - reference) < tol:
            reference = next_reference
            print("IRLS converged")
            break

        reference = next_reference
    
    return reference, weights


class IRLSSolver:
    def __init__(
        self,
        images: torch.Tensor,
        weight_function: Callable,
        image_variance: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_variance: torch.Tensor,
        ctf: torch.Tensor,
    ):
        self.images = images
        self.weight_function = weight_function
        self.image_variance = image_variance
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.ctf = ctf

    @torch.no_grad()
    def step(self, reference: torch.Tensor) -> torch.Tensor:
        """Performs a single IRLS iteration."""
        weights = self.weight_function(
            self.images, reference, torch.sqrt(self.image_variance)
        )

        s_1 = torch.sum(weights * self.ctf * self.images, dim=0)
        s_2 = torch.sum(weights * torch.square(self.ctf), dim=0)

        return (s_1 / self.image_variance + self.prior_mean / self.prior_variance) / (
            s_2 / self.image_variance + 1 / self.prior_variance
        )

    @torch.no_grad()
    def solve(self, initial_reference: torch.Tensor, max_it: int, tol: float = 1e-6):
        """Runs the full scheme."""
        reference = initial_reference
        for _ in range(max_it):
            next_reference = self.step(reference)
            if torch.linalg.norm(next_reference - reference) < tol:
                return next_reference
            reference = next_reference
        return reference
