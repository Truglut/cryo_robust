import numpy as np
import torch

from estimators.base import Estimator
from estimators.data import ImageBatch
from estimators.results import EstimatorResult, WeightSet
from estimators.irls import (
    IRLSSolver,
    JointIRLSFourier,
    FlatteningIRLSFourier,
    IRLSFourier,
)

from method_comparison.domain.enums import ImageSpace

ACCEPTED_FOURIER_SOLVERS = (
    IRLSSolver,
    JointIRLSFourier,
    FlatteningIRLSFourier,
    IRLSFourier,
)


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
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.irls_real = irls_real
        assert self.irls_real.space == ImageSpace.REAL
        self.irls_fourier = irls_fourier
        assert self.irls_fourier.space == ImageSpace.FOURIER_COMPLEX
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.initial_mu = initial_mu
        self.fourier_multiplier = fourier_multiplier
        self.converged = False

        if not isinstance(self.irls_fourier, ACCEPTED_FOURIER_SOLVERS):
            raise ValueError(
                f"Fourier IRLS solver must be one of {ACCEPTED_FOURIER_SOLVERS},"
                f" got {type(self.irls_fourier) = }"
            )

    def _real_update(
        self,
        batch: ImageBatch,
        *,
        reference_real: torch.Tensor,
        reference_fourier: torch.Tensor,
        dual_vars: torch.Tensor,
        mu: float,
        real_irls_max_iter: int | None = None,
    ) -> EstimatorResult:
        """
        Solves the real-space subproblem with IRLS
        """
        prior_mean = torch.fft.irfft2(
            reference_fourier + dual_vars / mu, norm=batch.norm
        )
        return self.irls_real.fit(
            batch,
            space=ImageSpace.REAL,
            reference=reference_real,
            prior_mean=prior_mean,
            prior_variance=1.0 / mu,
            max_iter_override=real_irls_max_iter,
        )

    def _fourier_update(
        self,
        batch: ImageBatch,
        *,
        next_real_transformed: torch.Tensor,
        reference_fourier: torch.Tensor,
        dual_vars: torch.Tensor,
        mu: float,
        fourier_irls_max_iter: int | None = None,
    ) -> EstimatorResult:
        """
        Solves the Fourier-space subproblem with one IRLS strategy
        """
        prior_mean = next_real_transformed - dual_vars / mu
        prior_variance = self.fourier_multiplier * (1.0 / mu)

        if isinstance(self.irls_fourier, IRLSSolver):
            # Real part
            real_results = self.irls_fourier.fit(
                batch,
                space=ImageSpace.FOURIER_REAL,
                reference=reference_fourier.real,
                prior_mean=prior_mean.real,
                prior_variance=prior_variance,
                max_iter_override=fourier_irls_max_iter,
            )
            real_weights = real_results.weights.fourier_real
            # Imaginary part
            imag_results = self.irls_fourier.fit(
                batch,
                space=ImageSpace.FOURIER_IMAG,
                reference=reference_fourier.imag,
                prior_mean=prior_mean.imag,
                prior_variance=prior_variance,
                max_iter_override=fourier_irls_max_iter,
            )
            imag_weights = imag_results.weights.fourier_imag
            next_fourier = torch.complex(real_results.estimate, imag_results.estimate)
            return EstimatorResult(
                average=None,
                estimate=next_fourier,
                weights=WeightSet(
                    real=None,
                    fourier_real=real_weights,
                    fourier_imag=imag_weights,
                ),
                converged=real_results.converged and imag_results.converged,
                n_iter=max(real_results.n_iter, imag_results.n_iter),
            )

        return self.irls_fourier.fit(
            batch=batch,
            space=ImageSpace.FOURIER_COMPLEX,
            reference=reference_fourier,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=fourier_irls_max_iter,
        )

    def step(
        self,
        batch: ImageBatch,
        *,
        reference_real: torch.Tensor,
        reference_fourier: torch.Tensor,
        dual_vars: torch.Tensor,
        mu: float,
        real_irls_max_iter: int | None = None,
        fourier_irls_max_iter: int | None = None,
    ) -> tuple[EstimatorResult, EstimatorResult, torch.Tensor]:
        """
        Performs a single iteration of the ADMM method, updating real and fourier space
        through IRLS with their respective prior means and variances.
        """
        # Real update
        real_results = self._real_update(
            batch,
            reference_real=reference_real,
            reference_fourier=reference_fourier,
            dual_vars=dual_vars,
            mu=mu,
            real_irls_max_iter=real_irls_max_iter,
        )
        next_real = real_results.estimate

        # Update fourier space estimate
        next_real_transformed = torch.fft.rfft2(next_real, norm="ortho")
        fourier_results = self._fourier_update(
            batch=batch,
            next_real_transformed=next_real_transformed,
            reference_fourier=reference_fourier,
            dual_vars=dual_vars,
            mu=mu,
            fourier_irls_max_iter=fourier_irls_max_iter,
        )

        return real_results, fourier_results, next_real_transformed

    @torch.inference_mode()
    def fit(
        self,
        batch: ImageBatch,
        *,
        initial_reference_real: torch.Tensor | None = None,
        initial_reference_fourier: torch.Tensor | None = None,
        verbose: bool = True,
    ):
        """
        Executes the Alternating Direction Method of Multipliers (ADMM) optimization.
        """
        # Reference initialization
        if initial_reference_real is None:
            real_images = batch.ensure_real()
            initial_reference_real = real_images.mean(dim=0)
        if initial_reference_fourier is None:
            fourier_images = batch.ensure_fourier()
            initial_reference_fourier = fourier_images.mean(dim=0)

        # Algorithm initialization
        reference_real = initial_reference_real
        reference_fourier = initial_reference_fourier
        dual_vars = torch.zeros_like(reference_fourier)
        mu = self.initial_mu
        self.converged = False

        # Compute residuals and check convergence every 5 iterations to save time
        for i in range(self.max_iter):
            # Perform one iteration: update real and fourier estimates
            real_results, fourier_results, next_real_transformed = self.step(
                batch,
                reference_real=reference_real,
                reference_fourier=reference_fourier,
                dual_vars=dual_vars,
                mu=mu,
                real_irls_max_iter=min(3 + i, self.irls_real.max_iter),
                fourier_irls_max_iter=min(3 + i, self.irls_fourier.max_iter),
            )
            next_fourier = fourier_results.estimate
            next_real = real_results.estimate

            # Update dual variable
            primal_residual = next_fourier - next_real_transformed
            dual_vars += mu * primal_residual

            # Convergence check every five iterations
            if i % 5 != 4:
                reference_real = next_real
                reference_fourier = next_fourier
            else:
                # Calculate residual norms for convergence check
                # (has to be done before updating references)
                primal_norm, dual_norm, eps_primal, eps_dual = self._residuals(
                    next_real=next_real,
                    next_real_transformed=next_real_transformed,
                    reference_fourier=reference_fourier,
                    next_fourier=next_fourier,
                    dual_vars=dual_vars,
                    mu=mu,
                    primal_residual=primal_residual,
                )

                # Update references
                reference_real = next_real
                reference_fourier = reference_fourier

                # Convergence check
                if primal_norm < eps_primal and dual_norm < eps_dual:
                    self.converged = True
                    break

                # Penalty parameter update
                mu = self._mu_update(mu, primal_norm, dual_norm)

                if verbose:
                    print(f"ADMM iteration {i + 1}.")
                    print(f"Penalty parameter: {mu = }")
                    print(
                        f"Residuals - Primal: {primal_norm:.4f} (Tol: {eps_primal:.4f}) | "
                        f"Dual: {dual_norm:.4f} (Tol: {eps_dual:.4f})"
                    )

        weights = WeightSet(
            real=real_results.weights.real,
            fourier_real=fourier_results.weights.fourier_real,
            fourier_imag=fourier_results.weights.fourier_imag
        )
        # Store final estimate and weights
        self.avg = (reference_real + torch.fft.irfft2(reference_fourier, norm=batch.norm)) / 2
        self.final_weights = weights.as_space_dict()

        return EstimatorResult(
            average=self.avg,
            estimate=None,
            weights=weights,
            converged=real_results.converged and fourier_results.converged,
            n_iter=i + 1,
        )

    def _mu_update(self, mu: float, primal_norm: float, dual_norm: float) -> float:
        if primal_norm > 10 * dual_norm:
            return 2.0 * mu
        if dual_norm > 10 * primal_norm:
            return 0.5 * mu
        return mu

    def _residuals(
        self,
        next_real: torch.Tensor,
        next_real_transformed: torch.Tensor,
        reference_fourier: torch.Tensor,
        next_fourier: torch.Tensor,
        dual_vars: torch.Tensor,
        mu: float,
        primal_residual: torch.Tensor,
    ):
        p = next_fourier.numel()  # number of restrictions
        n = next_real.numel()  # number of primal variables
        primal_norm = torch.linalg.norm(primal_residual).item()
        dual_norm = mu * torch.linalg.norm(next_fourier - reference_fourier).item()

        eps_primal = np.sqrt(p) * self.atol + self.rtol * max(
            torch.linalg.norm(next_real_transformed).item(),
            torch.linalg.norm(next_fourier).item(),
        )
        eps_dual = (
            np.sqrt(n) * self.atol + self.rtol * torch.linalg.norm(dual_vars).item()
        )

        return primal_norm, dual_norm, eps_primal, eps_dual

    def reconstruct_from_weights(
        self,
        images: ImageBatch | dict[ImageSpace, torch.Tensor],
        weights: WeightSet | dict[ImageSpace, torch.Tensor | None],
    ):
        norm = images.norm if isinstance(images, ImageBatch) else "ortho"

        ref_real = self.irls_real.reconstruct_from_weights(
            images, weights, space=ImageSpace.REAL
        )

        if isinstance(self.irls_fourier, IRLSSolver):
            ref_fourier_real = self.irls_fourier.reconstruct_from_weights(
                images, weights, space=ImageSpace.FOURIER_REAL
            )
            ref_fourier_imag = self.irls_fourier.reconstruct_from_weights(
                images, weights, space=ImageSpace.FOURIER_IMAG
            )
            ref_fourier = torch.complex(ref_fourier_real, ref_fourier_imag)
        else:
            ref_fourier = self.irls_fourier.reconstruct_from_weights(
                images, weights, space=ImageSpace.FOURIER_COMPLEX
            )

        # Transform Fourier space estimate back to real space through irfft2
        ref_inverse_fourier = torch.fft.irfft2(ref_fourier, norm=norm)

        # Return average of real-space and fourier-space estimates
        return 0.5 * (ref_real + ref_inverse_fourier)
