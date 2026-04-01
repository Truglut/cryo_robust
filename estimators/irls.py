import torch
from typing import Callable, Tuple, Optional, Union
from .base import Estimator, Space


class IRLSSolver(Estimator):
    def __init__(
        self,
        weight_function: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        max_iter: int,
        tol: float = 1e-5,
        damping_coef: float = 0.0,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.weight_function = weight_function
        self.max_iter = max_iter
        self.tol = tol
        self.damping_coef = damping_coef
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.converged = False

    def step(
        self,
        images: torch.Tensor,
        image_variance: Optional[torch.Tensor],
        ctf: Optional[Union[torch.Tensor, float]],
        reference: Optional[torch.Tensor],
        prior_mean: Optional[torch.Tensor],
        prior_variance: Optional[Union[torch.Tensor, float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single iteration of the Reweighted Least Squares update.
        """

        weights = self.weight_function(images, reference, torch.sqrt(image_variance))

        # Weight capping
        if self.min_weight is not None or self.max_weight is not None:
            weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)

        s_1 = torch.sum(weights * ctf * images, dim=0)
        s_2 = torch.sum(weights * ctf**2, dim=0)

        # Calculate new point (update)
        if prior_mean is not None and prior_variance is not None:
            safe_variance = image_variance + 1e-8
            update = (s_1 / safe_variance + prior_mean / prior_variance) / (
                s_2 / safe_variance + 1 / prior_variance
            )
        else:
            update = s_1 / s_2

        # Handle update damping
        coef = self.damping_coef
        new_ref = coef * reference + (1.0 - coef) * update
        return new_ref, weights

    def fit(
        self,
        images: torch.Tensor,
        image_variance: Optional[torch.Tensor] = None,
        ctf: Optional[Union[torch.Tensor, float]] = None,
        initial_reference: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_variance: Optional[Union[torch.Tensor, float]] = None,
    ):
        """
        Executes the Iteratively Reweighted Least Squares (IRLS) optimization.
        """
        images = self._prepare_data(images)

        # Initialize missing arguments to default values
        if image_variance is None:
            image_variance = torch.var(images, dim=0)
        if ctf is None:
            ctf = 1.0
        if initial_reference is None:
            initial_reference = torch.mean(images, dim=0)

        # Algorithm initialization
        reference = initial_reference
        weights = None
        self.converged = False

        for _ in range(self.max_iter):
            next_reference, weights = self.step(
                images,
                image_variance=image_variance,
                ctf=ctf,
                reference=reference,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )

            # Convergence check
            if torch.linalg.norm(next_reference - reference).item() < self.tol:
                reference = next_reference
                self.converged = True
                break

            reference = next_reference
        
        self.avg = reference
        self.final_weights = {
            Space.REAL: weights,
            Space.FOURIER_REAL: None,
            Space.FOURIER_IMAG: None
        }

        return reference, weights
