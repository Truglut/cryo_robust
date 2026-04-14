import numpy as np
import torch
from typing import Tuple, Dict
from .base import Estimator
from .irls import IRLSSolver
from utils.space import Space


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
            raise ValueError(f"Unrecognized ADMM initialization strategy: {initialization}")
        
        self.initialization = initialization

    def step(
        self,
        images: Dict[Space, torch.Tensor],
        image_variance: Dict[Space, torch.Tensor],
        ctf: torch.Tensor,
        precomp_ctf_images: Dict[Space, torch.Tensor],
        precomp_ctf_squared: Dict[Space, torch.Tensor | float],
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
            images[Space.REAL],
            image_variance=image_variance[Space.REAL],
            precomp_ctf_images=precomp_ctf_images[Space.REAL],
            precomp_ctf_squared=precomp_ctf_squared[Space.REAL],
            initial_reference=ref_real,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

        # Update fourier space estimate
        next_real_transformed = torch.fft.rfft2(next_real, norm="ortho")
        prior_mean_fourier = next_real_transformed - dual_vars / mu

        # Real part
        next_fourier_real, final_weights_fourier_real = self.irls_fourier.fit(
            images=images[Space.FOURIER_REAL],
            image_variance=image_variance[Space.FOURIER_REAL],
            ctf=ctf,
            precomp_ctf_images=precomp_ctf_images[Space.FOURIER_REAL],
            precomp_ctf_squared=precomp_ctf_squared[Space.FOURIER_REAL],
            initial_reference=ref_fourier.real,
            prior_mean=prior_mean_fourier.real,
            prior_variance=prior_variance * self.fourier_multiplier,
        )
        # Imaginary part
        next_fourier_imag, final_weights_fourier_imag = self.irls_fourier.fit(
            images=images[Space.FOURIER_IMAG],
            image_variance=image_variance[Space.FOURIER_IMAG],
            ctf=ctf,
            precomp_ctf_images=precomp_ctf_images[Space.FOURIER_IMAG],
            precomp_ctf_squared=precomp_ctf_squared[Space.FOURIER_IMAG],
            initial_reference=ref_fourier.imag,
            prior_mean=prior_mean_fourier.imag,
            prior_variance=prior_variance * self.fourier_multiplier,
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

    @torch.inference_mode()
    def fit(
        self,
        images: Dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: Dict[Space, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        initial_ref_real: torch.Tensor | None = None,
        initial_ref_fourier: torch.Tensor | None = None,
    ):
        """
        Executes the Iteratively Reweighted Least Squares (IRLS) optimization.
        """
        images = self._prepare_data(images)

        # Initialize missing arguments to default values
        
        # Images: compute fourier transform
        if (
            isinstance(images, torch.Tensor)
            or images[Space.FOURIER_REAL] is None
            or images[Space.FOURIER_IMAG] is None
        ):
            fourier_images = torch.fft.rfft2(images, norm="ortho")
            images[Space.FOURIER_REAL] = fourier_images.real
            images[Space.FOURIER_IMAG] = fourier_images.imag
            del fourier_images
        
        # Calculate image variance (per-pixel) if not provided
        if image_variance is None:
            image_variance = {space: torch.var(images[space], dim=0) for space in Space}
        
        # CTF
        if ctf is None:
            ctf = 1.0
            
        # Precompute ctf * images and ctf**2 for efficiency
        precomp_ctf_images = {
            Space.REAL: images[Space.REAL], # CTF is implicitly 1.0 in real space
            Space.FOURIER_REAL: ctf * images[Space.FOURIER_REAL],
            Space.FOURIER_IMAG: ctf * images[Space.FOURIER_IMAG]
        }
        if isinstance(ctf, torch.Tensor):
            precomp_ctf_squared = {
                Space.REAL: 1.0,
                Space.FOURIER_REAL: torch.square(ctf),
                Space.FOURIER_IMAG: torch.square(ctf)
            }
        else:
            precomp_ctf_squared = {
                Space.REAL: 1.0,
                Space.FOURIER_REAL: ctf**2,
                Space.FOURIER_IMAG: ctf**2
            }

        # Initial references
        if initial_ref_real is None:
            if self.initialization == "mean":
                initial_ref_real = torch.mean(images[Space.REAL], dim=0)
            elif self.initialization == "zeros":
                initial_ref_real = torch.zeros_like(images[Space.REAL][0])
            else:
                raise ValueError(f"Unrecognized ADMM initialization strategy: {self.initialization}")

        if initial_ref_fourier is None:
            if self.initialization == "mean":
                initial_ref_fourier = torch.complex(
                    torch.mean(images[Space.FOURIER_REAL], dim=0),
                    torch.mean(images[Space.FOURIER_IMAG], dim=0),
                )
            elif self.initialization == "zeros":
                initial_ref_fourier = torch.complex(
                    torch.zeros_like(images[Space.FOURIER_REAL][0]),
                    torch.zeros_like(images[Space.FOURIER_IMAG][0])
                )
            else:
                raise ValueError(f"Unrecognized ADMM initialization strategy: {self.initialization}")

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
                image_variance=image_variance,
                ctf=ctf,
                precomp_ctf_images = precomp_ctf_images,
                precomp_ctf_squared = precomp_ctf_squared,
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

            if i == 0 or i % 5 == 4:
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
