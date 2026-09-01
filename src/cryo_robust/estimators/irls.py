"""
IRLS-based estimators.

This module contains the basic single-space IRLS solver and several Fourier-space
wrappers. The wrappers differ in how they represent complex Fourier coefficients:
separate real/imaginary solvers, joint complex weights, or flattened real-valued
representations.
"""

from __future__ import annotations

import torch

from .base import Estimator
from .results import WeightSet, EstimatorResult
from .weights import weighted_average, WeightFunction
from .data import ImageBatch, to_tensor

from cryo_robust.domain import ImageSpace


class IRLSSolver(Estimator):
    """Iteratively reweighted least-squares solver for a single tensor space."""

    def __init__(
        self,
        weight_function: WeightFunction,
        max_iter: int,
        tol: float = 1.0e-5,
        damping_coef: float = 0.0,
        min_weight: float | None = None,
        max_weight: float | None = None,
        eps: float = 1e-8,
        space: ImageSpace | str = ImageSpace.REAL,
    ):
        # Estimator configuration
        self.weight_function = weight_function
        self.max_iter = max_iter
        self.tol = tol  # tolerance for stopping iterations
        self.eps = eps  # small numerical value to avoid division by zero
        self.space = space

        # Regularization strategies
        self.damping_coef = damping_coef
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.converged = False

    def _validate_prior(self, prior_mean, prior_variance) -> None:
        if (prior_mean is None) != (prior_variance is None):
            raise ValueError("prior_mean and prior_variance must be provided together.")

    def step(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor,
        image_std: torch.Tensor,
        reference: torch.Tensor,
        ctf: torch.Tensor | float | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single iteration of the Reweighted Least Squares update.
        """

        weights = self.weight_function(images, reference, image_std)

        # Weight capping
        if self.min_weight is not None or self.max_weight is not None:
            weights = torch.clamp_(weights, min=self.min_weight, max=self.max_weight)

        ctf_images = images if ctf is None else ctf * images
        s_1 = torch.sum(weights * ctf_images, dim=0)
        s_2 = (
            torch.sum(weights, dim=0)
            if ctf is None
            else torch.sum(weights * torch.as_tensor(ctf).square(), dim=0)
        )

        # Calculate new point (update)
        if prior_mean is None:
            update = s_1 / (s_2 + self.eps)
        else:
            safe_image_variance = image_variance + self.eps
            safe_prior_variance = prior_variance + self.eps
            numer = s_1 / safe_image_variance + prior_mean / (safe_prior_variance)
            denom = s_2 / safe_image_variance + 1 / (safe_prior_variance)
            update = numer / (denom + self.eps)

        # Return new reference with update damping and weights
        return (
            self.damping_coef * reference + (1.0 - self.damping_coef) * update,
            weights,
        )

    @torch.inference_mode()
    def fit_tensor(
        self,
        images: torch.Tensor,
        *,
        image_variance: torch.Tensor | None = None,
        image_std: torch.Tensor | None = None,
        ctf: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
        max_iter_override: int | None = None,
    ) -> EstimatorResult:
        """
        Executes the Iteratively Reweighted Least Squares (IRLS) optimization.
        """
        self._validate_prior(prior_mean, prior_variance)

        if image_variance is None:
            image_variance = images.var(dim=0)
        if image_std is None:
            image_std = image_variance.sqrt()

        # Initialize reference
        if reference is None:
            reference = images.mean(dim=0)
        else:
            reference = to_tensor(reference, device=images.device, dtype=images.dtype)

        # Algorithm initialization
        weights = None
        self.converged = False
        max_iter = max_iter_override or self.max_iter

        for iteration in range(max_iter):
            next_reference, weights = self.step(
                images,
                image_variance=image_variance,
                image_std=image_std,
                reference=reference,
                ctf=ctf,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )

            rel_diff = torch.linalg.norm(
                next_reference - reference
            ) / torch.linalg.norm(reference)
            reference = next_reference

            # Convergence check
            if rel_diff < self.tol:
                reference = next_reference
                self.converged = True
                break

        weight_set = WeightSet.for_irls_space(self.space, weights)

        if self.space == ImageSpace.REAL:
            average = reference
        else:
            average = None

        return EstimatorResult(
            average=average,
            estimate=reference,
            weights=weight_set,
            converged=self.converged,
            n_iter=iteration + 1,
        )

    @torch.inference_mode()
    def fit(
        self,
        batch: ImageBatch,
        *,
        space: ImageSpace | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
        max_iter_override: int | None = None,
    ):
        """
        Fit IRLS to an ImageBatch in the selected space.

        Parameters
        ----------
        batch:
            Canonical image batch.
        space:
            Space to operate on. Defaults to ``self.space``.
        reference:
            Optional initial reference. For Fourier spaces, this may be either a
            complex Fourier tensor or an already-selected real/imaginary component.
        prior_mean:
            Optional prior mean. Same selection rules as ``reference``.
        prior_variance:
            Optional prior variance.
        max_iter_override:
            Optional temporary iteration limit.
        """
        space = space or self.space
        self.space = space

        images, image_variance, image_std, ctf = batch.select_space_data(space)

        return self.fit_tensor(
            images,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )

    def reconstruct_from_weights(
        self,
        images: ImageBatch | torch.Tensor,
        weights: WeightSet | torch.Tensor,
        space: ImageSpace | None = None,
    ) -> torch.Tensor:
        if space is None:
            space = self.space

        if isinstance(images, ImageBatch):
            images = images.select_space(space)

        if isinstance(weights, WeightSet):
            weights = weights.select_space(space)

        if weights is None:
            raise ValueError(
                f"No weights available for reconstruction in space {space}."
            )

        return weighted_average(images, weights, eps=self.eps)
