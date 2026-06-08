from typing import Callable
import warnings

import torch

from estimators.base import Estimator
from estimators.weights import weighted_average

from method_comparison.domain.enums import Space


class IRLSSolver(Estimator):
    def __init__(
        self,
        weight_function: Callable[..., torch.Tensor],
        max_iter: int,
        tol: float = 1.0e-5,
        damping_coef: float = 0.0,
        min_weight: float | None = None,
        max_weight: float | None = None,
        device: str | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(device=device)
        self.weight_function = weight_function
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
        reference: torch.Tensor,
        ctf: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single iteration of the Reweighted Least Squares update.
        """

        if prior_mean is not None and prior_variance is None:
            raise ValueError("Prior mean provided but prior variance is None")
        if prior_mean is None and prior_variance is not None:
            raise ValueError("Prior variance provided but prior mean is None")

        weights = self.weight_function(images, reference, image_std)

        # Weight capping
        if self.min_weight is not None or self.max_weight is not None:
            weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)

        ctf_images = images if ctf is None else ctf * images
        s_1 = torch.sum(weights * ctf_images, dim=0)
        del ctf_images

        s_2 = (
            torch.sum(weights, dim=0)
            if ctf is None
            else torch.sum(weights * ctf.square(), dim=0)
        )

        # Calculate new point (update)
        if prior_mean is not None:
            safe_image_variance = image_variance + self.eps
            safe_prior_variance = prior_variance + self.eps
            numer = s_1 / safe_image_variance + prior_mean / (safe_prior_variance)
            denom = s_2 / safe_image_variance + 1 / (safe_prior_variance)
            update = numer / (denom + self.eps)
        else:
            update = s_1 / (s_2 + self.eps)

        # Handle update damping
        coef = self.damping_coef
        new_ref = coef * reference + (1.0 - coef) * update
        return new_ref, weights

    def _prepare_data(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor | None = None,
        image_std: torch.Tensor | None = None,
        ctf: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
    ) -> tuple[
        torch.Tensor,  # images
        torch.Tensor,  # image_variance
        torch.Tensor,  # image_std
        torch.Tensor,  # ctf
        torch.Tensor,  # reference
        torch.Tensor | None,  # prior mean
        torch.Tensor | None,  # prior_variance
    ]:
        if isinstance(images, dict):
            images = images[Space.REAL]

        if image_variance is None:
            image_variance = torch.var(images, dim=0)
        if image_std is None:
            image_std = image_variance.sqrt().clamp_min_(self.eps)

        if reference is None:
            reference = torch.mean(images, dim=0)
        else:
            reference = reference.clone()

        return (
            images,
            image_variance,
            image_std,
            ctf,
            reference,
            prior_mean,
            prior_variance,
        )

    @torch.inference_mode()
    def fit(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: torch.Tensor | None = None,
        image_std: torch.Tensor | None = None,
        ctf: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
        max_iter_override: int | None = None,
    ):
        """
        Executes the Iteratively Reweighted Least Squares (IRLS) optimization.
        """
        (
            images,
            image_variance,
            image_std,
            ctf,
            reference,
            prior_mean,
            prior_variance,
        ) = self._prepare_data(
            images,
            image_variance,
            image_std,
            ctf,
            reference,
            prior_mean,
            prior_variance,
        )

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
                reference=reference,
                ctf=ctf,
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
        self.final_weights[Space.REAL] = weights

        return reference, weights

    def reconstruct_from_weights(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        weights: dict[Space, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(images, dict):
            imgs = images[Space.REAL]
        else:
            imgs = images
        
        if isinstance(weights, dict):
            weights = weights[Space.REAL]
        else:
            weights = weights

        return weighted_average(imgs, weights, eps=self.eps)


class IRLSFourier(Estimator):
    def __init__(self, irls_real: IRLSSolver, irls_imag: IRLSSolver, device=None):
        super().__init__(device)

        self.irls_real = irls_real
        self.irls_imag = irls_imag

    def _prepare_data(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[Space, torch.Tensor | float] | float | None = None,
    ) -> tuple[
        dict[Space, torch.Tensor],
        dict[Space, torch.Tensor],
        torch.Tensor | float | None,
        dict[Space, torch.Tensor],
        dict[Space, torch.Tensor],
        dict[Space, torch.Tensor | float],
    ]:
        if isinstance(images, torch.Tensor):
            if torch.is_complex(images):
                fourier_images = images
            else:
                fourier_images = torch.fft.rfft2(images, norm="ortho")

            images = {
                Space.FOURIER_REAL: fourier_images.real,
                Space.FOURIER_IMAG: fourier_images.imag,
            }

        if isinstance(image_variance, torch.Tensor):
            if torch.is_complex(image_variance):
                image_variance = {
                    Space.FOURIER_REAL: image_variance.real,
                    Space.FOURIER_IMAG: image_variance.imag,
                }
            else:
                image_variance = {
                    Space.FOURIER_REAL: image_variance,
                    Space.FOURIER_IMAG: image_variance,
                }
        elif image_variance is None:
            image_variance = {
                Space.FOURIER_REAL: torch.var(images[Space.FOURIER_REAL], dim=0),
                Space.FOURIER_IMAG: torch.var(images[Space.FOURIER_IMAG], dim=0),
            }

        if prior_mean is None:
            prior_mean = {
                Space.FOURIER_REAL: None,
                Space.FOURIER_IMAG: None,
            }
        elif isinstance(prior_mean, torch.Tensor):
            if torch.is_complex(prior_mean):
                fourier_prior_mean = prior_mean
            else:
                fourier_prior_mean = torch.fft.rfft2(prior_mean, norm="ortho")

            prior_mean = {
                Space.FOURIER_REAL: fourier_prior_mean.real,
                Space.FOURIER_IMAG: fourier_prior_mean.imag,
            }

        if isinstance(prior_variance, torch.Tensor) and torch.is_complex(
            prior_variance
        ):
            prior_variance = {
                Space.FOURIER_REAL: prior_variance.real,
                Space.FOURIER_IMAG: prior_variance.imag,
            }
        elif not isinstance(prior_variance, dict):
            prior_variance = {
                Space.FOURIER_REAL: prior_variance,
                Space.FOURIER_IMAG: prior_variance,
            }

        if reference is None:
            reference = {
                Space.FOURIER_REAL: torch.mean(images[Space.FOURIER_REAL], dim=0),
                Space.FOURIER_IMAG: torch.mean(images[Space.FOURIER_IMAG], dim=0),
            }
        elif isinstance(reference, torch.Tensor):
            if not torch.is_complex(reference):
                reference = torch.fft.rfft2(reference, norm="ortho")

            reference = {
                Space.FOURIER_REAL: reference.real,
                Space.FOURIER_IMAG: reference.imag,
            }

        return images, image_variance, ctf, reference, prior_mean, prior_variance

    @torch.inference_mode()
    def fit(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[Space, torch.Tensor] | float | None = None,
    ):
        images, image_variance, ctf, reference, prior_mean, prior_variance = (
            self._prepare_data(
                images, image_variance, ctf, reference, prior_mean, prior_variance
            )
        )

        ref_real, weights_real = self.irls_real.fit(
            images[Space.FOURIER_REAL],
            image_variance=image_variance[Space.FOURIER_REAL],
            ctf=ctf,
            reference=reference[Space.FOURIER_REAL],
            prior_mean=prior_mean[Space.FOURIER_REAL],
            prior_variance=prior_variance[Space.FOURIER_REAL],
        )

        ref_imag, weights_imag = self.irls_imag.fit(
            images[Space.FOURIER_IMAG],
            image_variance=image_variance[Space.FOURIER_IMAG],
            ctf=ctf,
            reference=reference[Space.FOURIER_IMAG],
            prior_mean=prior_mean[Space.FOURIER_IMAG],
            prior_variance=prior_variance[Space.FOURIER_IMAG],
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
            images[Space.FOURIER_REAL], weights[Space.FOURIER_REAL]
        )
        reconstructed_fourier_imag = self.irls_imag.reconstruct_from_weights(
            images[Space.FOURIER_IMAG], weights[Space.FOURIER_IMAG]
        )

        return torch.fft.irfft2(
            torch.complex(reconstructed_fourier_real, reconstructed_fourier_imag),
            norm="ortho",
        )


class JointIRLSFourier(Estimator):
    def __init__(self, solver: IRLSSolver, device=None):
        super().__init__(device)

        self.solver = solver
        self.max_iter = self.solver.max_iter

    @torch.inference_mode()
    def step(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor,
        image_std: torch.Tensor,
        reference: torch.Tensor,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
    ):
        return self.solver.step(
            images=images,
            image_variance=image_variance,
            image_std=image_std,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

    def _prepare_data(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[Space, torch.Tensor | float] | float | None = None,
    ) -> tuple[
        torch.Tensor,  # images
        torch.Tensor,  # image_variance
        torch.Tensor,  # image_std
        torch.Tensor | float | None,  # ctf
        torch.Tensor,  # reference
        torch.Tensor,  # prior_mean
        torch.Tensor | float,  # prior_variance
    ]:
        # Images: return fourier transform as complex tensor
        if isinstance(images, torch.Tensor):
            if torch.is_complex(images):
                fourier_images = images
            else:
                fourier_images = torch.fft.rfft2(images, norm="ortho")
        else:
            fourier_images = torch.complex(
                images[Space.FOURIER_REAL], images[Space.FOURIER_IMAG]
            )

        # Image variance: return a single real tensor (per-frequency variance)
        if isinstance(image_variance, dict):
            if torch.allclose(
                image_variance[Space.FOURIER_REAL], image_variance[Space.FOURIER_IMAG]
            ):
                image_variance = image_variance[Space.FOURIER_REAL]
            else:
                raise ValueError(
                    "Two different Fourier-space image variances provided to JointIRLSFourier"
                )
        elif image_variance is None:
            image_variance = torch.var(fourier_images, dim=0)
        image_std = image_variance.sqrt()

        # Prior mean: return a single complex tensor
        if isinstance(prior_mean, torch.Tensor) and not torch.is_complex(prior_mean):
            prior_mean = torch.fft.rfft2(prior_mean, norm="ortho")
        elif isinstance(prior_mean, dict):
            prior_mean = torch.complex(
                prior_mean[Space.FOURIER_REAL], prior_mean[Space.FOURIER_IMAG]
            )
        

        # Prior variance: return a single real tensor or a float
        # Prior variance passed as dict: check that real part and imaginary part are equal, return any one
        if isinstance(prior_variance, dict):
            if torch.allclose(
                prior_variance[Space.FOURIER_REAL], prior_variance[Space.FOURIER_IMAG]
            ):
                raise ValueError(
                    "Two different Fourier-Space prior variances provided to JointIRLSFourier"
                )
            else:
                prior_variance = prior_variance[Space.FOURIER_REAL]
        # Prior variance passed as complex tensor: check that real and imaginary part are equal, return any one
        elif isinstance(prior_variance, torch.Tensor) and torch.is_complex(
            prior_variance
        ):
            if torch.allclose(prior_variance.real, prior_variance.imag):
                raise ValueError(
                    "Two different Fourier-Space prior variances provided to JointIRLSFourier"
                )
            else:
                prior_variance = prior_variance.real
        # Otherwise (prior variance passed as float or real tensor), return original value

        # Reference: return a complex tensor
        if reference is None:
            reference = torch.mean(fourier_images, dim=0)
        elif isinstance(reference, torch.Tensor) and not torch.is_complex(reference):
            reference = torch.fft.rfft2(reference, norm="ortho")

        return (
            fourier_images,
            image_variance,
            image_std,
            ctf,
            reference,
            prior_mean,
            prior_variance,
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
        max_iter_override: int | None = None,
    ):
        (
            fourier_images,
            image_variance,
            image_std,
            ctf,
            reference,
            fourier_prior_mean,
            prior_variance,
        ) = self._prepare_data(
            images, image_variance, ctf, reference, prior_mean, prior_variance
        )

        fourier_estimate, weights = self.solver.fit(
            images=fourier_images,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=fourier_prior_mean,
            prior_variance=prior_variance,
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

        self.solver = solver
        self.max_iter = self.solver.max_iter

    @torch.inference_mode()
    def step(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor,
        image_std: torch.Tensor,
        reference: torch.Tensor,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
    ):
        return self.solver.step(
            images=images,
            image_variance=image_variance,
            image_std=image_std,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

    def _prepare_data(
        self,
        images: dict[Space, torch.Tensor] | torch.Tensor,
        image_variance: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[Space, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[Space, torch.Tensor | float] | float | None = None,
    ) -> tuple[
        torch.Tensor,  # images
        tuple,  # fourier_shape
        torch.Tensor,  # image_variance
        torch.Tensor,  # image_std
        torch.Tensor | float | None,  # ctf
        torch.Tensor,  # reference
        torch.Tensor,  # prior_mean
        torch.Tensor | float,  # prior_variance
    ]:
        # Images: return fourier transform as a flattened real tensor
        if isinstance(images, torch.Tensor):
            if torch.is_complex(images):
                fourier_images = images
            else:
                fourier_images = torch.fft.rfft2(images, norm="ortho")
        else:
            fourier_images = torch.complex(
                images[Space.FOURIER_REAL], images[Space.FOURIER_IMAG]
            )
        fourier_shape = tuple(fourier_images.shape[1:])
        fourier_images_realimag = flatten_complex_batch(fourier_images)

        # Image variance: return a real tensor matching the flattened fourier images
        if isinstance(image_variance, dict):
            image_variance = flatten_complex_tensor(
                torch.complex(
                    image_variance[Space.FOURIER_REAL],
                    image_variance[Space.FOURIER_IMAG],
                )
            )
        elif isinstance(image_variance, torch.Tensor):
            if torch.is_complex(image_variance):
                image_variance = flatten_complex_tensor(image_variance)
            else:
                image_variance = flatten_complex_tensor(
                    torch.complex(image_variance, image_variance)
                )
        elif image_variance is None:
            image_variance = torch.var(fourier_images_realimag, dim=0)
        image_std = image_variance.sqrt()

        # Prior mean: return a single flattened complex tensor
        if prior_mean is None:
            fourier_prior_mean_realimag = None
        elif isinstance(prior_mean, torch.Tensor):
            if torch.is_complex(prior_mean):
                fourier_prior_mean = prior_mean
            else:
                fourier_prior_mean = torch.fft.rfft2(prior_mean, norm="ortho")
            fourier_prior_mean_realimag = flatten_complex_tensor(fourier_prior_mean)
        elif isinstance(prior_mean, dict):
            fourier_prior_mean = torch.complex(
                prior_mean[Space.FOURIER_REAL], prior_mean[Space.FOURIER_IMAG]
            )
            fourier_prior_mean_realimag = flatten_complex_tensor(fourier_prior_mean)

        # Prior variance: return a single real flattened tensor or a float
        if isinstance(prior_variance, dict):
            prior_variance = torch.complex(
                prior_variance[Space.FOURIER_REAL], prior_variance[Space.FOURIER_IMAG]
            )
            prior_variance = flatten_complex_tensor(prior_variance)
        # Prior variance passed as complex tensor: check that real and imaginary part are equal, return any one
        elif isinstance(prior_variance, torch.Tensor):
            if torch.is_complex(prior_variance):
                prior_variance = flatten_complex_tensor(prior_variance)
            else:
                prior_variance = flatten_complex_tensor(
                    torch.complex(prior_variance, prior_variance)
                )
        # Otherwise (prior variance passed as float), return original value

        # ctf: if tensor, cast to appropriate flattened dimensions
        if isinstance(ctf, torch.Tensor):
            n = fourier_images_realimag.shape[0]
            ctf = ctf.unsqueeze(-1)
            ctf = ctf.expand(*ctf.shape, 2)
            ctf = ctf.reshape(n, -1)

        # Reference: return a flattened complex tensor
        if reference is None:
            reference = torch.mean(fourier_images, dim=0)
        elif isinstance(reference, torch.Tensor) and not torch.is_complex(reference):
            reference = torch.fft.rfft2(reference, norm="ortho")
        reference = flatten_complex_tensor(reference)

        return (
            fourier_images_realimag,
            fourier_shape,
            image_variance,
            image_std,
            ctf,
            reference,
            fourier_prior_mean_realimag,
            prior_variance,
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
        max_iter_override: int | None = None,
    ):
        (
            fourier_images_realimag,
            fourier_shape,
            image_variance,
            image_std,
            ctf,
            reference,
            fourier_prior_mean_realimag,
            prior_variance,
        ) = self._prepare_data(
            images, image_variance, ctf, reference, prior_mean, prior_variance
        )
        n = fourier_images_realimag.shape[0]

        fourier_estimate, weights = self.solver.fit(
            images=fourier_images_realimag,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=fourier_prior_mean_realimag,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )
        # Reshape weights to the standard (N, 1, 1) format
        weights = weights.reshape(n, 1, 1)

        fourier_estimate = torch.view_as_complex(
            fourier_estimate.reshape(*fourier_shape, 2)
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
        fourier_images = torch.complex(
            images[Space.FOURIER_REAL], images[Space.FOURIER_IMAG]
        )

        n = fourier_images.shape[0]
        fourier_images_realimag = torch.view_as_real(fourier_images)
        fourier_images_realimag = fourier_images_realimag.reshape(n, -1)

        w = weights[Space.FOURIER_REAL]
        w = w.reshape(n, 1)
        realimag_estimate = weighted_average(fourier_images_realimag, weights=w)
        fourier_estimate = torch.view_as_complex(
            realimag_estimate.reshape(*fourier_images.shape[1:], 2)
        )

        return torch.fft.irfft2(fourier_estimate, norm="ortho")
