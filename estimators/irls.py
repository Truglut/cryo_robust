from typing import Callable

import torch

from estimators.base import Estimator
from estimators.weights import weighted_average

from method_comparison.domain.enums import Space


class IRLSSolver(Estimator):
    def __init__(
        self,
        weight_function: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        max_iter: int,
        tol: float = 1.0e-5,
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
        image_std: torch.Tensor,
        precomp_ctf_images: torch.Tensor,
        precomp_ctf_squared: torch.Tensor | float,
        reference: torch.Tensor,
        prior_mean: torch.Tensor | None,
        prior_variance: torch.Tensor | float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a single iteration of the Reweighted Least Squares update."""

        weights = self.weight_function(images, reference, image_std)

        # Weight capping
        if self.min_weight is not None or self.max_weight is not None:
            weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)

        s_1 = torch.sum(weights * precomp_ctf_images, dim=0)
        s_2 = torch.sum(weights * precomp_ctf_squared, dim=0)

        # Calculate new point (update)
        if prior_mean is not None and prior_variance is not None:
            safe_variance = image_variance + self.eps
            numer = s_1 / safe_variance + prior_mean / (prior_variance + self.eps)
            denom = s_2 / safe_variance + 1 / (prior_variance + self.eps)
            update = numer / (denom + self.eps)
        else:
            update = s_1 / (s_2 + self.eps)

        # Handle update damping
        coef = self.damping_coef
        new_ref = coef * reference + (1.0 - coef) * update
        return new_ref, weights

    @torch.inference_mode()
    def fit(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        image_std: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[Space, torch.Tensor] | torch.Tensor | float | None = None,
        precomp_ctf_images: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        precomp_ctf_squared: (
            dict[Space, torch.Tensor] | torch.Tensor | float | None
        ) = None,
        max_iter_override: int | None = None,
    ):
        """
        Executes the Iteratively Reweighted Least Squares (IRLS) optimization.
        """

        # Unpack dictionaries based on space
        if isinstance(images, dict):
            images = images[self.space]
        images = self._prepare_data(images)
        if isinstance(image_variance, dict):
            image_variance = image_variance[self.space]
        if isinstance(prior_mean, dict):
            prior_mean = prior_mean[self.space]
        if isinstance(prior_variance, dict):
            prior_variance = prior_variance[self.space]
        if isinstance(precomp_ctf_images, dict):
            precomp_ctf_images = precomp_ctf_images[self.space]
        if isinstance(precomp_ctf_squared, dict):
            precomp_ctf_squared = precomp_ctf_squared[self.space]

        # Initialize missing arguments to default values

        # Image initialisation: choose relevant space
        if isinstance(images, dict):
            images = images[self.space]

        # Image variance: compute or choose relevant space
        if image_variance is None:
            image_variance = torch.var(images, dim=0)
        elif isinstance(image_variance, dict):
            image_variance = image_variance[self.space]
        if image_std is None or isinstance(image_std, dict):
            image_std = torch.sqrt(image_variance)
        image_std = torch.clamp(image_std, min=self.eps)

        # ctf
        if ctf is None:
            ctf = 1.0
        # Precompute ctf * images and ctf**2 for efficiency
        if precomp_ctf_images is None:
            precomp_ctf_images = ctf * images
        if precomp_ctf_squared is None:
            if isinstance(ctf, torch.Tensor):
                precomp_ctf_squared = torch.square(ctf)
            else:
                precomp_ctf_squared = ctf**2

        # Initial reference
        if reference is None:
            reference = torch.mean(images, dim=0)
        else:
            reference = reference.clone()

        # Prior mean and variance: choose relevant space
        if isinstance(prior_mean, dict):
            prior_mean = prior_mean[self.space]
        if isinstance(prior_variance, dict):
            prior_variance = prior_variance[self.space]

        # Configure maximum iterations
        max_iter = max_iter_override or self.max_iter

        # Algorithm initialization
        weights = None
        self.converged = False

        for _ in range(max_iter):
            next_reference, weights = self.step(
                images,
                image_variance=image_variance,
                image_std=image_std,
                precomp_ctf_images=precomp_ctf_images,
                precomp_ctf_squared=precomp_ctf_squared,
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
        self.final_weights = {space: None for space in Space}
        self.final_weights[self.space] = weights

        return reference, weights

    def reconstruct_from_weights(
        self,
        images: dict[Space, torch.Tensor],
        weights: dict[Space, torch.Tensor | None],
    ) -> torch.Tensor:
        imgs = images[self.space]
        weights = weights[self.space]

        return weighted_average(imgs, weights, eps=self.eps)


class IRLSFourier(Estimator):
    def __init__(self, irls_real: IRLSSolver, irls_imag: IRLSSolver, device=None):
        super().__init__(device)

        if irls_real.space != Space.FOURIER_REAL:
            raise ValueError("irls_real must be IRLSSolver with space FOURIER_REAL")
        self.irls_real = irls_real
        if irls_imag.space != Space.FOURIER_IMAG:
            raise ValueError("irls_imag must be IRLSSolver with space FOURIER_IMAG")
        self.irls_imag = irls_imag

    @torch.inference_mode()
    def fit(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[Space, torch.Tensor] | float | None = None,
        precomp_ctf_images: dict[Space, torch.Tensor] | None = None,
        precomp_ctf_squared: torch.Tensor | float | None = None,
    ):
        if isinstance(images, torch.Tensor):
            images = {Space.REAL: images}
        if (
            images.get(Space.FOURIER_IMAG) is None
            or images.get(Space.FOURIER_REAL) is None
        ):
            fourier_images = torch.fft.rfft2(images[Space.REAL], norm="ortho")
            images[Space.FOURIER_REAL] = fourier_images.real
            images[Space.FOURIER_IMAG] = fourier_images.imag
            del fourier_images
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
        if reference is None:
            reference = {space: torch.mean(images[space], dim=0) for space in Space}
        if isinstance(reference, torch.Tensor):
            if torch.is_complex(reference):
                reference = {
                    Space.FOURIER_REAL: reference.real,
                    Space.FOURIER_IMAG: reference.imag,
                }
            else:
                reference = {
                    Space.FOURIER_REAL: reference,
                    Space.FOURIER_IMAG: reference,
                }

        ref_real, weights_real = self.irls_real.fit(
            images,
            image_variance=image_variance,
            ctf=ctf,
            reference=reference[Space.FOURIER_REAL],
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            precomp_ctf_images=precomp_ctf_images,
            precomp_ctf_squared=precomp_ctf_squared,
        )

        ref_imag, weights_imag = self.irls_imag.fit(
            images,
            image_variance=image_variance,
            ctf=ctf,
            reference=reference[Space.FOURIER_IMAG],
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            precomp_ctf_images=precomp_ctf_images,
            precomp_ctf_squared=precomp_ctf_squared,
        )

        fourier_estimate = torch.complex(ref_real, ref_imag)
        self.avg = torch.fft.irfft2(fourier_estimate, norm="ortho")
        self.final_weights = {
            Space.REAL: None,
            Space.FOURIER_REAL: weights_real,
            Space.FOURIER_IMAG: weights_imag,
        }

        return fourier_estimate, weights_real, weights_imag

    def reconstruct_from_weights(
        self,
        images: dict[Space, torch.Tensor],
        weights: dict[Space, torch.Tensor | None],
    ) -> torch.Tensor:
        reconstructed_fourier_real = self.irls_real.reconstruct_from_weights(
            images, weights
        )
        reconstructed_fourier_imag = self.irls_imag.reconstruct_from_weights(
            images, weights
        )

        return torch.fft.irfft2(
            torch.complex(reconstructed_fourier_real, reconstructed_fourier_imag),
            norm="ortho",
        )


class JointIRLSFourier(Estimator):
    def __init__(self, solver: IRLSSolver, device=None):
        super().__init__(device)

        if not solver.space == Space.REAL:
            raise ValueError(
                "By convention, JointIRLSFourier solver space must be Space.REAL"
            )
        self.solver = solver
        self.max_iter = self.solver.max_iter

    @torch.inference_mode()
    def step(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor,
        image_std: torch.Tensor,
        precomp_ctf_images: torch.Tensor,
        precomp_ctf_squared: torch.Tensor,
        reference: torch.Tensor,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
    ):
        return self.solver.step(
            images=images,
            image_variance=image_variance,
            image_std=image_std,
            precomp_ctf_images=precomp_ctf_images,
            precomp_ctf_squared=precomp_ctf_squared,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

    @torch.inference_mode()
    def fit(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | torch.Tensor | float | None = None,
        image_std: dict[Space, torch.Tensor] | torch.Tensor | float | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
        precomp_ctf_images: dict[Space, torch.Tensor] | None = None,
        precomp_ctf_squared: torch.Tensor | float | None = None,
        max_iter_override: int | None = None,
    ):
        # Get fourier space representation of images
        if isinstance(images, torch.Tensor):
            fourier_images = torch.fft.rfft2(images, norm="ortho")
        else:
            fourier_images = torch.complex(
                images[Space.FOURIER_REAL], images[Space.FOURIER_IMAG]
            )

        # Get per-frequency image variance if given as space dict or None
        if not isinstance(image_variance, (torch.Tensor, float)):
            image_variance = torch.var(fourier_images, dim=0)
        image_std = torch.sqrt(image_variance)

        # ctf
        if ctf is None:
            ctf = 1.0
        # Precompute ctf * images and ctf**2 for efficiency
        if precomp_ctf_images is None:
            precomp_ctf_images = ctf * fourier_images
        elif isinstance(precomp_ctf_images, dict):
            precomp_ctf_images = torch.complex(
                precomp_ctf_images[Space.FOURIER_REAL],
                precomp_ctf_images[Space.FOURIER_IMAG],
            )
        if precomp_ctf_squared is None:
            if isinstance(ctf, torch.Tensor):
                precomp_ctf_squared = torch.square(ctf)
            else:
                precomp_ctf_squared = ctf**2

        if reference is None:
            reference = torch.mean(fourier_images, dim=0)

        if isinstance(prior_mean, dict):
            prior_mean = torch.complex(
                prior_mean[Space.FOURIER_REAL], prior_mean[Space.FOURIER_IMAG]
            )

        fourier_estimate, weights = self.solver.fit(
            images=fourier_images,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            precomp_ctf_images=precomp_ctf_images,
            precomp_ctf_squared=precomp_ctf_squared,
            max_iter_override=max_iter_override,
        )

        weights = {
            Space.REAL: None,
            Space.FOURIER_REAL: weights,
            Space.FOURIER_IMAG: weights,
        }

        self.final_weights = weights
        self.avg = torch.fft.irfft2(fourier_estimate, norm="ortho")

        return fourier_estimate, weights[Space.FOURIER_REAL]

    @torch.inference_mode()
    def reconstruct_from_weights(
        self, images: dict[Space, torch.Tensor], weights: dict[Space, torch.Tensor]
    ):
        fourier_real = weighted_average(
            images[Space.FOURIER_REAL], weights[Space.FOURIER_REAL]
        )
        fourier_imag = weighted_average(
            images[Space.FOURIER_IMAG], weights[Space.FOURIER_IMAG]
        )

        return torch.fft.irfft2(torch.complex(fourier_real, fourier_imag), norm="ortho")


def flatten_complex_batch(batch: torch.Tensor) -> torch.Tensor:
    n = batch.shape[0]
    return torch.view_as_real(batch).reshape(n, -1)

def flatten_complex_tensor(v: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(v).reshape(-1)

class FlatteningIRLSFourier(Estimator):
    def __init__(self, solver: IRLSSolver, device=None):
        super().__init__(device)

        if not solver.space == Space.REAL:
            raise ValueError(
                "By convention, JointIRLSFourier solver space must be Space.REAL"
            )
        self.solver = solver
        self.space = Space.FOURIER_REAL
        self.max_iter = self.solver.max_iter

    @torch.inference_mode()
    def step(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor,
        image_std: torch.Tensor,
        precomp_ctf_images: torch.Tensor,
        precomp_ctf_squared: torch.Tensor,
        reference: torch.Tensor,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
    ):
        return self.solver.step(
            images=images,
            image_variance=image_variance,
            image_std=image_std,
            precomp_ctf_images=precomp_ctf_images,
            precomp_ctf_squared=precomp_ctf_squared,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

    @torch.inference_mode()
    def fit(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | torch.Tensor | float | None = None,
        image_std: dict[Space, torch.Tensor] | torch.Tensor | float | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
        precomp_ctf_images: dict[Space, torch.Tensor] | None = None,
        precomp_ctf_squared: torch.Tensor | float | None = None,
        max_iter_override: int | None = None
    ):
        # Get fourier space representation of images
        if isinstance(images, torch.Tensor):
            fourier_images = torch.fft.rfft2(images, norm="ortho")
        else:
            fourier_images = torch.complex(
                images[Space.FOURIER_REAL], images[Space.FOURIER_IMAG]
            )

        n = fourier_images.shape[0]
        fourier_images_realimag = torch.view_as_real(fourier_images)
        fourier_images_realimag = fourier_images_realimag.reshape(n, -1)

        # Get per-frequency image variance if given as space dict or None
        if not isinstance(image_variance, (torch.Tensor, float)):
            image_variance = torch.var(fourier_images_realimag, dim=0)
        image_std = torch.sqrt(image_variance)

        # ctf
        if ctf is None:
            ctf = 1.0
        elif isinstance(ctf, torch.Tensor):
            ctf = ctf.unsqueeze(-1)
            ctf = ctf.expand(*ctf.shape, 2)
            ctf = ctf.reshape(n, -1)
        # Precompute ctf * images and ctf**2 for efficiency
        precomp_ctf_images = ctf * fourier_images_realimag
        if precomp_ctf_squared is None:
            if isinstance(ctf, torch.Tensor):
                precomp_ctf_squared = torch.square(ctf)
            else:
                precomp_ctf_squared = ctf**2

        reference = torch.mean(fourier_images_realimag, dim=0)

        if isinstance(prior_mean, dict):
            prior_mean = torch.complex(
                prior_mean[Space.FOURIER_REAL], prior_mean[Space.FOURIER_IMAG]
            )
        if prior_mean is not None:
            prior_mean = flatten_complex_tensor(prior_mean)
        if isinstance(prior_variance, torch.Tensor) and prior_variance.ndim == 2:
            prior_variance = flatten_complex_tensor(prior_variance)

        fourier_estimate, weights = self.solver.fit(
            images=fourier_images_realimag,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            precomp_ctf_images=precomp_ctf_images,
            precomp_ctf_squared=precomp_ctf_squared,
            max_iter_override=max_iter_override
        )
        # Reshape weights to the standard (N, 1, 1) format
        weights = weights.reshape(n, 1, 1)

        fourier_estimate = torch.view_as_complex(
            fourier_estimate.reshape(*fourier_images.shape[1:], 2)
        )

        weights = {
            Space.REAL: weights,
            Space.FOURIER_REAL: weights,
            Space.FOURIER_IMAG: None,
        }

        self.final_weights = weights
        self.avg = torch.fft.irfft2(fourier_estimate, norm="ortho")

        return fourier_estimate, weights[Space.FOURIER_REAL]

    @torch.inference_mode()
    def reconstruct_from_weights(
        self, images: dict[Space, torch.Tensor], weights: dict[Space, torch.Tensor]
    ):
        fourier_images = torch.complex(
            images[Space.FOURIER_REAL], images[Space.FOURIER_IMAG]
        )

        n = fourier_images.shape[0]
        fourier_images_realimag = torch.view_as_real(fourier_images)
        fourier_images_realimag = fourier_images_realimag.reshape(n, -1)

        w = weights[self.space]
        w = w.reshape(n, 1)
        realimag_estimate = weighted_average(fourier_images_realimag, weights=w)
        fourier_estimate = torch.view_as_complex(
            realimag_estimate.reshape(*fourier_images.shape[1:], 2)
        )

        return torch.fft.irfft2(fourier_estimate, norm="ortho")
