import numpy as np
import torch
from typing import Tuple, Optional, Dict, Union
from .base import Estimator, Space
from .irls import IRLSSolver


class ADMMSolver(Estimator):
    def __init__(
        self,
        irls_real: IRLSSolver,
        irls_fourier: IRLSSolver,
        max_iter: int,
        initial_mu: float,
        fourier_multiplier: float,
        atol: float,
        rtol: float,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.irls_real = irls_real
        self.irls_fourier = irls_fourier
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.initial_mu = initial_mu
        self.fourier_multiplier = fourier_multiplier
        self.converged = False

    def step(
        self,
        images: torch.Tensor,
        fourier_images: torch.Tensor,
        image_variance: torch.Tensor,
        fourier_image_variance: torch.Tensor,
        ctf: torch.Tensor,
        ref_real: torch.Tensor,
        ref_fourier: torch.Tensor,
        dual_vars: torch.Tensor,
        mu: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs a single iteration of the Reweighted Least Squares update.
        """
        # Update real space estimate with IRLS
        prior_mean = torch.fft.irfft2(ref_fourier + dual_vars / mu, norm="ortho")
        prior_variance = 1 / mu
        next_real, final_weights_real = self.irls_real.fit(
            images,
            image_variance=image_variance,
            ctf=1.0,
            initial_reference=ref_real,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

        # Update fourier space estimate
        next_real_transformed = torch.fft.rfft2(next_real, norm="ortho")
        prior_mean_fourier = next_real_transformed - dual_vars / mu

        # Real part
        next_fourier_real, final_weights_fourier_real = self.irls_fourier.fit(
            images=fourier_images.real,
            image_variance=fourier_image_variance.real / self.fourier_multiplier,
            ctf=ctf,
            initial_reference=ref_fourier.real,
            prior_mean=prior_mean_fourier.real,
            prior_variance=prior_variance,
        )
        # Imaginary part
        next_fourier_imag, final_weights_fourier_imag = self.irls_fourier.fit(
            images=fourier_images.imag,
            image_variance=fourier_image_variance.imag / self.fourier_multiplier,
            ctf=ctf,
            initial_reference=ref_fourier.imag,
            prior_mean=prior_mean_fourier.imag,
            prior_variance=prior_variance,
        )
        next_fourier = torch.complex(next_fourier_real, next_fourier_imag)

        return (
            next_real,
            next_real_transformed,
            next_fourier,
            {
                Space.REAL: final_weights_real,
                Space.FOURIER_REAL: final_weights_fourier_real,
                Space.FOURIER_IMAG: final_weights_fourier_imag,
            },
        )

    def fit(
        self,
        images: torch.Tensor,
        fourier_images: Optional[torch.Tensor] = None,
        image_variance: Optional[torch.Tensor] = None,
        fourier_image_variance: Optional[torch.Tensor] = None,
        ctf: Optional[Union[torch.Tensor, float]] = None,
        initial_ref_real: Optional[torch.Tensor] = None,
        initial_ref_fourier: Optional[torch.Tensor] = None,
    ):
        """
        Executes the Iteratively Reweighted Least Squares (IRLS) optimization.
        """
        images = self._prepare_data(images)

        # Initialize missing arguments to default values
        if fourier_images is None:
            fourier_images = torch.fft.rfft2(images, norm="ortho")
        if image_variance is None:
            image_variance = torch.var(images, dim=0)
        if fourier_image_variance is None:
            fourier_image_variance = torch.complex(
                torch.var(fourier_images.real, dim=0),
                torch.var(fourier_images.imag, dim=0),
            )
        if ctf is None:
            ctf = 1.0
        if initial_ref_real is None:
            initial_ref_real = torch.mean(images, dim=0)
            # initial_ref_real = torch.zeros_like(images[0])
        if initial_ref_fourier is None:
            initial_ref_fourier = torch.mean(fourier_images, dim=0)
            # initial_ref_fourier = torch.zeros_like(fourier_images[0])

        # Algorithm initialisation
        ref_real = initial_ref_real
        ref_fourier = initial_ref_fourier
        dual_vars = torch.zeros_like(ref_fourier)
        mu = self.initial_mu
        self.converged = False
        p = ref_fourier.shape[0] * ref_fourier.shape[1]  # number of restrictions
        n = ref_real.shape[0] * ref_real.shape[1]  # number of primal variables
        weights = None

        for i in range(self.max_iter):
            next_real, next_real_transformed, next_fourier, weights = self.step(
                images=images,
                fourier_images=fourier_images,
                image_variance=image_variance,
                fourier_image_variance=fourier_image_variance,
                ctf=ctf,
                ref_real=ref_real,
                ref_fourier=ref_fourier,
                dual_vars=dual_vars,
                mu=mu,
            )

            # Compute residuals for updates and convergence check
            primal_residual = next_fourier - next_real_transformed
            dual_residual = mu * torch.fft.irfft2(
                next_fourier - ref_fourier, norm="ortho"
            )

            # Update dual variable
            dual_vars += mu * primal_residual

            # Convergence check
            eps_pri = np.sqrt(p) * self.atol + self.rtol * max(
                torch.linalg.norm(next_real_transformed).item(),
                torch.linalg.norm(next_fourier).item(),
            )
            eps_dual = (
                np.sqrt(n) * self.atol + self.rtol * torch.linalg.norm(dual_vars).item()
            )
            primal_norm = torch.linalg.norm(primal_residual).item()
            dual_norm = torch.linalg.norm(dual_residual).item()

            print(f"ADMM iteration {i + 1}.")
            print(f"Penalty parameter: {mu = }")
            print(
                f"Residuals - Primal: {primal_norm:.4f} (Tol: {eps_pri:.4f}) | "
                f"Dual: {dual_norm:.4f} (Tol: {eps_dual:.4f})"
            )

            if primal_norm < eps_pri and dual_norm < eps_dual:
                self.converged = True
                ref_real = next_real
                ref_fourier = next_fourier
                break

            # Penalty parameter update
            if primal_norm > 10 * dual_norm:
                mu *= 2
            elif dual_norm > 10 * primal_norm:
                mu /= 2

            # Update references
            ref_real = next_real
            ref_fourier = next_fourier

        # Store final estimate and weights
        self.avg = (ref_real + torch.fft.irfft2(ref_fourier, norm="ortho")) / 2
        self.final_weights = weights
        return ref_real, ref_fourier, weights
