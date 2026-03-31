import torch
import numpy as np
from typing import Callable, Tuple, Dict
from dataclasses import dataclass
from irls import irls_scheme


@dataclass
class ADMMResults:
    """
    Stores the results of the ADMM optimization scheme.

    Attributes:
        converged (bool): True if the primal and dual residuals fell below the thresholds.
        iterations (int): Total number of ADMM iterations executed.
        estimation_real (torch.Tensor): Final robust estimation in the real space domain.
        estimation_fourier (torch.Tensor): Final robust estimation in the Fourier domain.
        last_weights (Dict[str, torch.Tensor]): The final IRLS weights used for real,
                                                fourier-real, and fourier-imaginary steps.
    """

    converged: bool
    iterations: int
    estimation_real: torch.Tensor
    estimation_fourier: torch.Tensor
    last_weights: Dict[str, torch.Tensor]


@torch.no_grad()
def admm_scheme(
    images: torch.Tensor,
    fourier_images: torch.Tensor,
    ctf: torch.Tensor,
    initial_ref_real: torch.Tensor,
    initial_ref_fourier: torch.Tensor,
    mu: float,
    C: float,
    weight_function_real: Callable,
    weight_function_fourier: Callable,
    max_iter_admm: int = 50,
    max_iter_irls: int = 20,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Initialize variables
    ref_real = initial_ref_real
    ref_fourier = initial_ref_fourier
    dual_vars = torch.zeros_like(ref_fourier)
    p = ref_fourier.shape[0] * ref_fourier.shape[1]  # number of restrictions
    n = ref_real.shape[0] * ref_real.shape[1]  # number of primal variables
    converged = False

    # Estimate variances
    image_variance = torch.var(images, dim=0)
    fourier_image_variance = torch.complex(
        torch.var(fourier_images.real, dim=0), torch.var(fourier_images.imag, dim=0)
    )

    for i in range(max_iter_admm):
        print(f"ADMM iteration {i + 1}.")
        # Update real space estimate with IRLS
        prior_mean = torch.fft.irfft2(ref_fourier + dual_vars / mu, norm="ortho")

        print("Executing real space IRLS...")
        next_real, last_weights_real = irls_scheme(
            images,
            ref_real,
            weight_function_real,
            image_variance,
            prior_mean,
            prior_variance=1 / mu,
            ctf=torch.tensor(1, device=images.device, dtype=images.dtype),
            max_iter=max_iter_irls,
        )

        # Update fourier estimate with IRLS
        next_real_transformed = torch.fft.rfft2(next_real, norm="ortho")

        print("Executing Fourier space IRLS:")
        # Update real part
        print("Real part...")
        next_fourier_real, last_weights_fourier_real = irls_scheme(
            fourier_images.real,
            ref_fourier.real,
            weight_function_fourier,
            image_variance=fourier_image_variance.real / C,
            prior_mean=next_real_transformed.real - dual_vars.real / mu,
            prior_variance=1 / mu,
            ctf=ctf,
            max_iter=max_iter_irls,
        )
        # Update imaginary part
        print("Imaginary part...")
        next_fourier_imag, last_weights_fourier_imag = irls_scheme(
            fourier_images.imag,
            ref_fourier.imag,
            weight_function_fourier,
            image_variance=fourier_image_variance.imag / C,
            prior_mean=next_real_transformed.imag - dual_vars.imag / mu,
            prior_variance=1 / mu,
            ctf=ctf,
            max_iter=max_iter_irls,
        )
        next_fourier = torch.complex(next_fourier_real, next_fourier_imag)

        primal_residual = next_fourier - next_real_transformed
        dual_residual = mu * torch.fft.irfft2(next_fourier - ref_fourier, norm="ortho")

        # Update dual variable
        dual_vars += mu * (primal_residual)

        # Convergence check
        eps_pri = np.sqrt(p) * atol + rtol * max(
            torch.linalg.norm(next_real_transformed).item(),
            torch.linalg.norm(ref_fourier).item(),
        )
        eps_dual = np.sqrt(n) * atol + rtol * torch.linalg.norm(dual_vars).item()

        primal_norm = torch.linalg.norm(primal_residual).item()
        dual_norm = torch.linalg.norm(dual_residual).item()
        print(
            f"Residuals - Primal: {primal_norm:.4f} (Tol: {eps_pri:.4f}) | "
            f"Dual: {dual_norm:.4f} (Tol: {eps_dual:.4f})"
        )
        if primal_norm < eps_pri and dual_norm < eps_dual:
            converged = True
            ref_real = next_real
            ref_fourier = next_fourier
            break

        # Penalty parameter update:
        if primal_norm > 10 * dual_norm:
            mu *= 2
        elif dual_norm > 10 * primal_norm:
            mu /= 2
        print(f"Penalty parameter: {mu = }")
        ref_real = next_real
        ref_fourier = next_fourier

    return ADMMResults(
        converged=converged,
        iterations=i + 1,
        estimation_real=ref_real,
        estimation_fourier=ref_fourier,
        last_weights={
            "real": last_weights_real,
            "fourier_real": last_weights_fourier_real,
            "fourier_imag": last_weights_fourier_imag,
        },
    )
