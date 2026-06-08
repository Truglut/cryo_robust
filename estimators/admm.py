import numpy as np
import torch

from estimators.base import Estimator
from estimators.irls import (
    IRLSSolver,
    JointIRLSFourier,
    FlatteningIRLSFourier,
    IRLSFourier,
)
from estimators.weights import weighted_average

from method_comparison.domain.enums import Space

ACCEPTED_FOURIER_SOLVERS = [
    IRLSSolver,
    JointIRLSFourier,
    FlatteningIRLSFourier,
    IRLSFourier,
]


class ADMMSolver(Estimator):
    def __init__(
        self,
        irls_real: IRLSSolver,
        irls_fourier: Estimator,
        max_iter: int,
        initial_mu: float,
        fourier_multiplier: float,
        atol: float,
        rtol: float,
        initialization: str = "mean",
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.irls_real = irls_real
        self.irls_fourier = irls_fourier
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.initial_mu = initial_mu
        self.fourier_multiplier = fourier_multiplier
        self.initialization = initialization
        self.converged = False

        if initialization not in ["mean", "zeros"]:
            raise ValueError(
                f"Unrecognized ADMM initialization strategy: {initialization}"
            )

        if not isinstance(self.irls_fourier, tuple(ACCEPTED_FOURIER_SOLVERS)):
            raise ValueError(
                f"Fourier IRLS solver must be one of {ACCEPTED_FOURIER_SOLVERS},"
                f" got {type(self.irls_fourier) = }"
            )

        self.initialization = initialization

    def _prepare_data(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        initial_ref_real: torch.Tensor | None = None,
        initial_ref_fourier: torch.Tensor | None = None,
    ) -> tuple[
        dict[Space, torch.Tensor],  # images
        dict[Space, torch.Tensor],  # image_variance
        dict[Space, torch.Tensor],  # image_std
        torch.Tensor | None,  # ctf
        torch.Tensor,  # initial_ref_real
        torch.Tensor,  # inital_ref_fourier
    ]:
        # Make sure images come in dict format
        if isinstance(images, torch.Tensor):
            images = {Space.REAL: images}
        # Compute fourier transform if necessary
        if (
            images.get(Space.FOURIER_REAL) is None
            or images.get(Space.FOURIER_IMAG) is None
        ):
            fourier_images = torch.fft.rfft2(images[Space.REAL], norm="ortho")
            images[Space.FOURIER_REAL] = fourier_images.real
            images[Space.FOURIER_IMAG] = fourier_images.imag
            del fourier_images

        # Calculate image variance (per-pixel) if not provided
        if image_variance is None:
            image_variance = {space: torch.var(images[space], dim=0) for space in Space}
        image_std = {space: torch.sqrt(image_variance[space]) for space in Space}

        # Initial references
        if initial_ref_real is None:
            if self.initialization == "mean":
                initial_ref_real = torch.mean(images[Space.REAL], dim=0)
            elif self.initialization == "zeros":
                initial_ref_real = torch.zeros_like(images[Space.REAL][0])
            else:
                raise ValueError(
                    f"Unrecognized ADMM initialization strategy: {self.initialization}"
                )

        if initial_ref_fourier is None:
            if self.initialization == "mean":
                initial_ref_fourier = torch.complex(
                    torch.mean(images[Space.FOURIER_REAL], dim=0),
                    torch.mean(images[Space.FOURIER_IMAG], dim=0),
                )
            elif self.initialization == "zeros":
                initial_ref_fourier = torch.complex(
                    torch.zeros_like(images[Space.FOURIER_REAL][0]),
                    torch.zeros_like(images[Space.FOURIER_IMAG][0]),
                )
            else:
                raise ValueError(
                    f"Unrecognized ADMM initialization strategy: {self.initialization}"
                )

        return (
            images,
            image_variance,
            image_std,
            ctf,
            initial_ref_real,
            initial_ref_fourier,
        )

    def step(
        self,
        images: dict[Space, torch.Tensor],
        image_variance: dict[Space, torch.Tensor],
        image_std: dict[Space, torch.Tensor],
        ctf: torch.Tensor,
        ref_real: torch.Tensor,
        ref_fourier: torch.Tensor,
        dual_vars: torch.Tensor,
        mu: float,
        real_irls_max_iter: int | None = None,
        fourier_irls_max_iter: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[Space, torch.Tensor]]:
        """
        Performs a single iteration of the ADMM method, updating real and fourier space
        through IRLS with their respective prior means and variances.
        """
        # Update real space estimate with IRLS
        prior_mean = torch.fft.irfft2(ref_fourier + dual_vars / mu, norm="ortho")
        prior_variance = 1 / mu
        next_real, final_weights_real = self.irls_real.fit(
            images[Space.REAL],
            image_variance=image_variance[Space.REAL],
            image_std=image_std[Space.REAL],
            reference=ref_real,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=real_irls_max_iter,
        )

        # Update fourier space estimate
        next_real_transformed = torch.fft.rfft2(next_real, norm="ortho")
        prior_mean_fourier = next_real_transformed - dual_vars / mu

        if isinstance(self.irls_fourier, IRLSSolver):
            # Real part
            next_fourier_real, final_weights_fourier_real = self.irls_fourier.fit(
                images=images[Space.FOURIER_REAL],
                image_variance=image_variance[Space.FOURIER_REAL],
                image_std=image_std[Space.FOURIER_REAL],
                ctf=ctf,
                reference=ref_fourier.real,
                prior_mean=prior_mean_fourier.real,
                prior_variance=prior_variance * self.fourier_multiplier,
                max_iter_override=fourier_irls_max_iter,
            )
            # Imaginary part
            next_fourier_imag, final_weights_fourier_imag = self.irls_fourier.fit(
                images=images[Space.FOURIER_IMAG],
                image_variance=image_variance[Space.FOURIER_IMAG],
                image_std=image_std[Space.FOURIER_IMAG],
                ctf=ctf,
                reference=ref_fourier.imag,
                prior_mean=prior_mean_fourier.imag,
                prior_variance=prior_variance * self.fourier_multiplier,
                max_iter_override=fourier_irls_max_iter,
            )
            next_fourier = torch.complex(next_fourier_real, next_fourier_imag)
        elif isinstance(self.irls_fourier, IRLSFourier):
            next_fourier, final_weights_fourier_real, final_weights_fourier_imag = (
                self.irls_fourier.fit(
                    images=images,
                    image_variance=image_variance,
                    image_std=image_std,
                    ctf=ctf,
                    reference=ref_fourier,
                    prior_mean=prior_mean_fourier,
                    prior_variance=prior_variance * self.fourier_multiplier,
                    max_iter_override=fourier_irls_max_iter,
                )
            )
        else:
            next_fourier, final_weights_fourier = self.irls_fourier.fit(
                images=images,
                image_variance=image_variance,
                image_std=image_std,
                ctf=ctf,
                reference=ref_fourier,
                prior_mean=prior_mean_fourier,
                prior_variance=prior_variance * self.fourier_multiplier,
                max_iter_override=fourier_irls_max_iter,
            )
            final_weights_fourier_real = final_weights_fourier
            final_weights_fourier_imag = final_weights_fourier

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

    @torch.inference_mode()
    def fit(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        initial_ref_real: torch.Tensor | None = None,
        initial_ref_fourier: torch.Tensor | None = None,
        verbose: bool = True,
    ):
        """
        Executes the Alternating Direction Method of Multipliers (ADMM) optimization.
        """
        (
            images,
            image_variance,
            image_std,
            ctf,
            initial_ref_real,
            initial_ref_fourier,
        ) = self._prepare_data(
            images, image_variance, ctf, initial_ref_real, initial_ref_fourier
        )

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
            real_irls_max_iter = min(3 + i, self.irls_real.max_iter)
            fourier_irls_max_iter = min(3 + i, self.irls_fourier.max_iter)
            next_real, next_real_transformed, next_fourier, weights = self.step(
                images=images,
                image_variance=image_variance,
                image_std=image_std,
                ctf=ctf,
                ref_real=ref_real,
                ref_fourier=ref_fourier,
                dual_vars=dual_vars,
                mu=mu,
                real_irls_max_iter=real_irls_max_iter,
                fourier_irls_max_iter=fourier_irls_max_iter,
            )

            # Update dual variable
            primal_residual = next_fourier - next_real_transformed
            dual_vars += mu * primal_residual

            # Convergence check every five iterations
            if i % 5 == 4:
                primal_norm = torch.linalg.norm(primal_residual).item()
                dual_norm = mu * torch.linalg.norm(next_fourier - ref_fourier).item()

                eps_pri = np.sqrt(p) * self.atol + self.rtol * max(
                    torch.linalg.norm(next_real_transformed).item(),
                    torch.linalg.norm(next_fourier).item(),
                )
                eps_dual = (
                    np.sqrt(n) * self.atol
                    + self.rtol * torch.linalg.norm(dual_vars).item()
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

                if verbose:
                    print(f"ADMM iteration {i + 1}.")
                    print(f"Penalty parameter: {mu = }")
                    print(
                        f"Residuals - Primal: {primal_norm:.4f} (Tol: {eps_pri:.4f}) | "
                        f"Dual: {dual_norm:.4f} (Tol: {eps_dual:.4f})"
                    )

            # Update references
            ref_real = next_real
            ref_fourier = next_fourier

        # Store final estimate and weights
        self.avg = (ref_real + torch.fft.irfft2(ref_fourier, norm="ortho")) / 2
        self.final_weights = weights
        return ref_real, ref_fourier, weights

    def reconstruct_from_weights(self, images, weights):
        # Compute real space estimate using images and weights
        ref_real = weighted_average(images[Space.REAL], weights[Space.REAL], eps=1.0e-8)

        # Compute estimates for real and imaginary part of Fourier space
        ref_fourier_real = weighted_average(
            images[Space.FOURIER_REAL], weights[Space.FOURIER_REAL], eps=1.0e-8
        )
        ref_fourier_imag = weighted_average(
            images[Space.FOURIER_IMAG], weights[Space.FOURIER_IMAG], eps=1.0e-8
        )

        # Transform Fourier space estimates back to real space through irfft2
        ref_inverse_fourier = torch.fft.irfft2(
            torch.complex(ref_fourier_real, ref_fourier_imag), norm="ortho"
        )

        # Return average of real-space and fourier-space estimates
        return 0.5 * (ref_real + ref_inverse_fourier)
