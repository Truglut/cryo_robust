from __future__ import annotations
from typing import Callable

import torch

from estimators.base import Estimator
from estimators.results import WeightSet, EstimatorResult
from estimators.weights import weighted_average
from estimators.data import ImageBatch, to_tensor

from method_comparison.domain.enums import ImageSpace


class IRLSSolver(Estimator):
    """Iteratively reweighted least-squares solver for a single tensor space."""

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
        space: ImageSpace | str = ImageSpace.REAL,
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
        self.space = space

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
    ):
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
        self.final_weights = weight_set.as_space_dict()
        self.avg = reference

        return reference, weights

        # if self.space == ImageSpace.REAL:
        #     average = reference
        # else:
        #     average = None

        # return EstimatorResult(
        #     average=average,
        #     estimate=reference,
        #     weights=weight_set,
        #     converged = self.converged,
        #     n_iter = iteration + 1
        # )

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
        images: ImageBatch | dict[ImageSpace, torch.Tensor | None],
        weights: dict[ImageSpace, torch.Tensor | None] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(images, ImageBatch):
            images = images.select_space_images(self.space)
        elif isinstance(images, dict):
            images = images[self.space]

        if isinstance(weights, dict):
            weights = weights[self.space]

        return weighted_average(images, weights, eps=self.eps)


class IRLSFourier(Estimator):
    def __init__(self, irls_real: IRLSSolver, irls_imag: IRLSSolver, device=None):
        super().__init__(device)

        self.irls_real = irls_real
        self.irls_imag = irls_imag

    def _prepare_data(
        self,
        images: dict[ImageSpace, torch.Tensor] | torch.Tensor,
        image_variance: dict[ImageSpace, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[ImageSpace, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[ImageSpace, torch.Tensor | float] | float | None = None,
    ) -> tuple[
        dict[ImageSpace, torch.Tensor],
        dict[ImageSpace, torch.Tensor],
        torch.Tensor | float | None,
        dict[ImageSpace, torch.Tensor],
        dict[ImageSpace, torch.Tensor],
        dict[ImageSpace, torch.Tensor | float],
    ]:
        if isinstance(images, torch.Tensor):
            if torch.is_complex(images):
                fourier_images = images
            else:
                fourier_images = torch.fft.rfft2(images, norm="ortho")

            images = {
                ImageSpace.FOURIER_REAL: fourier_images.real,
                ImageSpace.FOURIER_IMAG: fourier_images.imag,
            }

        if isinstance(image_variance, torch.Tensor):
            if torch.is_complex(image_variance):
                image_variance = {
                    ImageSpace.FOURIER_REAL: image_variance.real,
                    ImageSpace.FOURIER_IMAG: image_variance.imag,
                }
            else:
                image_variance = {
                    ImageSpace.FOURIER_REAL: image_variance,
                    ImageSpace.FOURIER_IMAG: image_variance,
                }
        elif image_variance is None:
            image_variance = {
                ImageSpace.FOURIER_REAL: torch.var(
                    images[ImageSpace.FOURIER_REAL], dim=0
                ),
                ImageSpace.FOURIER_IMAG: torch.var(
                    images[ImageSpace.FOURIER_IMAG], dim=0
                ),
            }

        if prior_mean is None:
            prior_mean = {
                ImageSpace.FOURIER_REAL: None,
                ImageSpace.FOURIER_IMAG: None,
            }
        elif isinstance(prior_mean, torch.Tensor):
            if torch.is_complex(prior_mean):
                fourier_prior_mean = prior_mean
            else:
                fourier_prior_mean = torch.fft.rfft2(prior_mean, norm="ortho")

            prior_mean = {
                ImageSpace.FOURIER_REAL: fourier_prior_mean.real,
                ImageSpace.FOURIER_IMAG: fourier_prior_mean.imag,
            }

        if isinstance(prior_variance, torch.Tensor) and torch.is_complex(
            prior_variance
        ):
            prior_variance = {
                ImageSpace.FOURIER_REAL: prior_variance.real,
                ImageSpace.FOURIER_IMAG: prior_variance.imag,
            }
        elif not isinstance(prior_variance, dict):
            prior_variance = {
                ImageSpace.FOURIER_REAL: prior_variance,
                ImageSpace.FOURIER_IMAG: prior_variance,
            }

        if reference is None:
            reference = {
                ImageSpace.FOURIER_REAL: torch.mean(
                    images[ImageSpace.FOURIER_REAL], dim=0
                ),
                ImageSpace.FOURIER_IMAG: torch.mean(
                    images[ImageSpace.FOURIER_IMAG], dim=0
                ),
            }
        elif isinstance(reference, torch.Tensor):
            if not torch.is_complex(reference):
                reference = torch.fft.rfft2(reference, norm="ortho")

            reference = {
                ImageSpace.FOURIER_REAL: reference.real,
                ImageSpace.FOURIER_IMAG: reference.imag,
            }

        return images, image_variance, ctf, reference, prior_mean, prior_variance

    @torch.inference_mode()
    def fit_tensor(
        self,
        images: dict[ImageSpace, torch.Tensor] | torch.Tensor,
        image_variance: dict[ImageSpace, torch.Tensor] | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[ImageSpace, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[ImageSpace, torch.Tensor] | float | None = None,
    ):
        images, image_variance, ctf, reference, prior_mean, prior_variance = (
            self._prepare_data(
                images, image_variance, ctf, reference, prior_mean, prior_variance
            )
        )

        ref_real, weights_real = self.irls_real.fit_tensor(
            images[ImageSpace.FOURIER_REAL],
            image_variance=image_variance[ImageSpace.FOURIER_REAL],
            ctf=ctf,
            reference=reference[ImageSpace.FOURIER_REAL],
            prior_mean=prior_mean[ImageSpace.FOURIER_REAL],
            prior_variance=prior_variance[ImageSpace.FOURIER_REAL],
        )

        ref_imag, weights_imag = self.irls_imag.fit_tensor(
            images[ImageSpace.FOURIER_IMAG],
            image_variance=image_variance[ImageSpace.FOURIER_IMAG],
            ctf=ctf,
            reference=reference[ImageSpace.FOURIER_IMAG],
            prior_mean=prior_mean[ImageSpace.FOURIER_IMAG],
            prior_variance=prior_variance[ImageSpace.FOURIER_IMAG],
        )

        fourier_estimate = torch.complex(ref_real, ref_imag)
        self.avg = torch.fft.irfft2(fourier_estimate, norm="ortho")
        self.final_weights = {
            ImageSpace.REAL: None,
            ImageSpace.FOURIER_REAL: weights_real,
            ImageSpace.FOURIER_IMAG: weights_imag,
        }

        return fourier_estimate, weights_real, weights_imag

    # @torch.inference_mode()
    # def fit(
    #     self,
    #     batch: ImageBatch,
    #     *,
    #     space: ImageSpace | None = None,
    #     reference: torch.Tensor | None = None,
    #     prior_mean: torch.Tensor | None = None,
    #     prior_variance: torch.Tensor | float | None = None,
    #     max_iter_override: int | None = None,
    # ):
    #     fourier_real, real_var, real_std, ctf = batch.select_space_data(
    #         ImageSpace.FOURIER_REAL
    #     )
    #     fourier_imag, imag_var, imag_std, _ = batch.select_space_data(
    #         ImageSpace.FOURIER_IMAG
    #     )

    #     if reference is None:
    #         reference = torch.complex(
    #             fourier_real.mean(dim=0), fourier_imag.mean(dim=0)
    #         )

    def reconstruct_from_weights(
        self,
        images: dict[ImageSpace, torch.Tensor],
        weights: dict[ImageSpace, torch.Tensor | None],
    ) -> torch.Tensor:
        reconstructed_fourier_real = self.irls_real.reconstruct_from_weights(
            images[ImageSpace.FOURIER_REAL], weights[ImageSpace.FOURIER_REAL]
        )
        reconstructed_fourier_imag = self.irls_imag.reconstruct_from_weights(
            images[ImageSpace.FOURIER_IMAG], weights[ImageSpace.FOURIER_IMAG]
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
        images: dict[ImageSpace, torch.Tensor] | torch.Tensor,
        image_variance: dict[ImageSpace, torch.Tensor] | torch.Tensor | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[ImageSpace, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[ImageSpace, torch.Tensor | float] | float | None = None,
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
                images[ImageSpace.FOURIER_REAL], images[ImageSpace.FOURIER_IMAG]
            )

        # Image variance: return a single real tensor (per-frequency variance)
        if isinstance(image_variance, dict):
            if torch.allclose(
                image_variance[ImageSpace.FOURIER_REAL],
                image_variance[ImageSpace.FOURIER_IMAG],
            ):
                image_variance = image_variance[ImageSpace.FOURIER_REAL]
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
                prior_mean[ImageSpace.FOURIER_REAL], prior_mean[ImageSpace.FOURIER_IMAG]
            )

        # Prior variance: return a single real tensor or a float
        # Prior variance passed as dict: check that real part and imaginary part are equal, return any one
        if isinstance(prior_variance, dict):
            if torch.allclose(
                prior_variance[ImageSpace.FOURIER_REAL],
                prior_variance[ImageSpace.FOURIER_IMAG],
            ):
                raise ValueError(
                    "Two different Fourier-Space prior variances provided to JointIRLSFourier"
                )
            else:
                prior_variance = prior_variance[ImageSpace.FOURIER_REAL]
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
    def fit_tensor(
        self,
        images: dict[ImageSpace, torch.Tensor] | torch.Tensor,
        image_variance: (
            dict[ImageSpace, torch.Tensor] | torch.Tensor | float | None
        ) = None,
        image_std: dict[ImageSpace, torch.Tensor] | torch.Tensor | float | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[ImageSpace, torch.Tensor] | torch.Tensor | None = None,
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

        fourier_estimate, weights = self.solver.fit_tensor(
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
            ImageSpace.REAL: None,
            ImageSpace.FOURIER_REAL: weights,
            ImageSpace.FOURIER_IMAG: weights,
        }

        self.final_weights = weights
        self.avg = torch.fft.irfft2(fourier_estimate, norm="ortho")

        return fourier_estimate, weights[ImageSpace.FOURIER_REAL]

    @torch.inference_mode()
    def reconstruct_from_weights(
        self,
        images: dict[ImageSpace, torch.Tensor],
        weights: dict[ImageSpace, torch.Tensor],
    ):
        fourier_real = weighted_average(
            images[ImageSpace.FOURIER_REAL], weights[ImageSpace.FOURIER_REAL]
        )
        fourier_imag = weighted_average(
            images[ImageSpace.FOURIER_IMAG], weights[ImageSpace.FOURIER_IMAG]
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
        images: dict[ImageSpace, torch.Tensor] | torch.Tensor,
        image_variance: dict[ImageSpace, torch.Tensor] | torch.Tensor | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[ImageSpace, torch.Tensor] | torch.Tensor | None = None,
        prior_variance: dict[ImageSpace, torch.Tensor | float] | float | None = None,
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
                images[ImageSpace.FOURIER_REAL], images[ImageSpace.FOURIER_IMAG]
            )
        fourier_shape = tuple(fourier_images.shape[1:])
        fourier_images_realimag = flatten_complex_batch(fourier_images)

        # Image variance: return a real tensor matching the flattened fourier images
        if isinstance(image_variance, dict):
            image_variance = flatten_complex_tensor(
                torch.complex(
                    image_variance[ImageSpace.FOURIER_REAL],
                    image_variance[ImageSpace.FOURIER_IMAG],
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
                prior_mean[ImageSpace.FOURIER_REAL], prior_mean[ImageSpace.FOURIER_IMAG]
            )
            fourier_prior_mean_realimag = flatten_complex_tensor(fourier_prior_mean)

        # Prior variance: return a single real flattened tensor or a float
        if isinstance(prior_variance, dict):
            prior_variance = torch.complex(
                prior_variance[ImageSpace.FOURIER_REAL],
                prior_variance[ImageSpace.FOURIER_IMAG],
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
    def fit_tensor(
        self,
        images: dict[ImageSpace, torch.Tensor] | torch.Tensor,
        image_variance: (
            dict[ImageSpace, torch.Tensor] | torch.Tensor | float | None
        ) = None,
        image_std: dict[ImageSpace, torch.Tensor] | torch.Tensor | float | None = None,
        ctf: torch.Tensor | float | None = None,
        reference: torch.Tensor | None = None,
        prior_mean: dict[ImageSpace, torch.Tensor] | torch.Tensor | None = None,
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

        fourier_estimate, weights = self.solver.fit_tensor(
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
            ImageSpace.REAL: None,
            ImageSpace.FOURIER_REAL: weights,
            ImageSpace.FOURIER_IMAG: weights,
        }

        self.final_weights = weights
        self.avg = torch.fft.irfft2(fourier_estimate, norm="ortho")

        return fourier_estimate, weights[ImageSpace.FOURIER_REAL]

    @torch.inference_mode()
    def reconstruct_from_weights(
        self,
        images: dict[ImageSpace, torch.Tensor],
        weights: dict[ImageSpace, torch.Tensor],
    ):
        fourier_images = torch.complex(
            images[ImageSpace.FOURIER_REAL], images[ImageSpace.FOURIER_IMAG]
        )

        n = fourier_images.shape[0]
        fourier_images_realimag = torch.view_as_real(fourier_images)
        fourier_images_realimag = fourier_images_realimag.reshape(n, -1)

        w = weights[ImageSpace.FOURIER_REAL]
        w = w.reshape(n, 1)
        realimag_estimate = weighted_average(fourier_images_realimag, weights=w)
        fourier_estimate = torch.view_as_complex(
            realimag_estimate.reshape(*fourier_images.shape[1:], 2)
        )

        return torch.fft.irfft2(fourier_estimate, norm="ortho")
