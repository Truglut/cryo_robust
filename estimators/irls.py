import torch
from typing import Callable, Tuple, Dict
from .base import Estimator
from utils.space import Space


class IRLSSolver(Estimator):
    def __init__(
        self,
        weight_function: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        max_iter: int,
        tol: float = 1e-5,
        damping_coef: float = 0.0,
        min_weight: float | None = None,
        max_weight: float | None = None,
        space: Space = Space.REAL,
        device: str | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(device=device)
        self.weight_function = weight_function
        self.space = space
        self.max_iter = max_iter
        self.tol = tol
        self.damping_coef = damping_coef
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.converged = False
        self.eps = eps

    def step(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor,
        ctf: torch.Tensor | float,
        reference: torch.Tensor,
        prior_mean: torch.Tensor | None,
        prior_variance: torch.Tensor | float | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a single iteration of the Reweighted Least Squares update."""

        weights = self.weight_function(images, reference, torch.sqrt(image_variance))

        # Weight capping
        if self.min_weight is not None or self.max_weight is not None:
            weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)

        s_1 = torch.sum(weights * ctf * images, dim=0)
        s_2 = torch.sum(weights * ctf**2, dim=0)

        # Calculate new point (update)
        if prior_mean is not None and prior_variance is not None:
            safe_variance = image_variance + self.eps
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
        images: Dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: Dict[Space, torch.Tensor] | torch.Tensor | None = None,
        ctf: torch.Tensor | float | None = None,
        initial_reference: torch.Tensor | None = None,
        prior_mean: Dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: Dict[Space, torch.Tensor] | torch.Tensor | float | None = None,
    ):
        """
        Executes the Iteratively Reweighted Least Squares (IRLS) optimization.
        """
        images = self._prepare_data(images)

        # Initialize missing arguments to default values
        if isinstance(images, dict):
            images = images[self.space]
        if image_variance is None:
            image_variance = torch.var(images, dim=0)
        elif isinstance(image_variance, dict):
            image_variance = image_variance[self.space]
        if ctf is None:
            ctf = 1.0
        if initial_reference is None:
            initial_reference = torch.mean(images, dim=0)
        if isinstance(prior_mean, dict):
            prior_mean = prior_mean[self.space]
        if isinstance(prior_variance, dict):
            prior_variance = prior_variance[self.space]

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
            if torch.linalg.norm(next_reference - reference) < self.tol:
                reference = next_reference
                self.converged = True
                break

            reference = next_reference

        self.avg = reference
        self.final_weights = {
            Space.REAL: weights,
            Space.FOURIER_REAL: None,
            Space.FOURIER_IMAG: None,
        }

        return reference, weights


class IRLSFourier(Estimator):
    def __init__(self, irls_real: IRLSSolver, irls_imag: IRLSSolver, device=None):
        super().__init__(device)

        if irls_real.space != Space.FOURIER_REAL:
            raise ValueError("irls_real must be IRLSSolver with space FOURIER_REAL")
        self.irls_real = irls_real
        if irls_imag.space != Space.FOURIER_IMAG:
            raise ValueError("irls_imag must be IRLSSolver with space FOURIER_IMAG")
        self.irls_imag = irls_imag

    def fit(
        self,
        images: Dict[Space, torch.Tensor],
        image_variance: Dict[Space, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        initial_reference: torch.Tensor | None = None,
        prior_mean: Dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: Dict[Space, torch.Tensor] | float | None = None,
    ):
        if isinstance(prior_variance, float):
            prior_variance = {
                Space.FOURIER_IMAG: prior_variance,
                Space.FOURIER_REAL: prior_variance,
            }
        if isinstance(prior_mean, torch.Tensor):
            if torch.is_complex(prior_mean):
                prior_mean = {
                    Space.FOURIER_REAL: prior_mean.real,
                    Space.FOURIER_IMAG: prior_mean.imag,
                }
            else:
                prior_mean = {
                    Space.FOURIER_REAL: prior_mean,
                    Space.FOURIER_IMAG: prior_mean,
                }
        if initial_reference is None:
            initial_reference = {
                space: torch.mean(images[space], dim=0) for space in Space
            }
        if isinstance(initial_reference, torch.Tensor):
            if torch.is_complex(initial_reference):
                initial_reference = {
                    Space.FOURIER_REAL: initial_reference.real,
                    Space.FOURIER_IMAG: initial_reference.imag,
                }
            else:
                initial_reference = {
                    Space.FOURIER_REAL: initial_reference,
                    Space.FOURIER_IMAG: initial_reference,
                }

        ref_real, weights_real = self.irls_real.fit(
            images,
            image_variance=image_variance,
            ctf=ctf,
            initial_reference=initial_reference[Space.FOURIER_REAL],
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

        ref_imag, weights_imag = self.irls_imag.fit(
            images,
            image_variance=image_variance,
            ctf=ctf,
            initial_reference=initial_reference[Space.FOURIER_IMAG],
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

        self.avg = torch.fft.irfft2(torch.complex(ref_real, ref_imag))
        self.final_weights = {
            Space.REAL: None,
            Space.FOURIER_REAL: weights_real,
            Space.FOURIER_IMAG: weights_imag,
        }

        return self.avg, self.final_weights
