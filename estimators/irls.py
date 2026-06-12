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
    ) -> EstimatorResult:
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

        if self.space == ImageSpace.REAL:
            average = reference
        else:
            average = None

        return EstimatorResult(
            average=average,
            estimate=reference,
            weights=weight_set,
            converged=self.converged,
            n_iter=iteration + 1,
        )

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
        images: ImageBatch | dict[ImageSpace, torch.Tensor | None] | torch.Tensor,
        weights: dict[ImageSpace, torch.Tensor | None] | WeightSet | torch.Tensor,
        space: ImageSpace | None = None,
    ) -> torch.Tensor:
        if weights is None:
            print("weights son None al entrar")
        if space is None:
            space = self.space

        if isinstance(images, ImageBatch):
            images = images.select_space_images(space)
        elif isinstance(images, dict):
            if space == ImageSpace.FOURIER_COMPLEX:
                images = torch.complex(
                    images[ImageSpace.FOURIER_REAL], images[ImageSpace.FOURIER_IMAG]
                )
            else:
                images = images[space]

        if isinstance(weights, dict):
            if space == ImageSpace.FOURIER_COMPLEX:
                weights = weights[ImageSpace.FOURIER_REAL]
            else:
                weights = weights[space]
        elif isinstance(weights, WeightSet):
            weights = weights.select_space_weights(space)
        if weights is None:
            print("weights son None después de seleccionar")

        return weighted_average(images, weights, eps=self.eps)


class IRLSFourier(Estimator):
    def __init__(self, irls_real: IRLSSolver, irls_imag: IRLSSolver, device=None):
        super().__init__(device)

        self.irls_real = irls_real
        assert self.irls_real.space == ImageSpace.FOURIER_REAL
        self.irls_imag = irls_imag
        assert self.irls_imag.space == ImageSpace.FOURIER_IMAG
        self.space = ImageSpace.FOURIER_COMPLEX

    @torch.inference_mode()
    def fit(
        self,
        batch: ImageBatch,
        *,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX,
        reference: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
        max_iter_override: int | None = None,
    ):
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )
        ref_real, ref_imag = self._setup_reference(reference, batch.norm)

        real_results = self.irls_real.fit(
            batch=batch,
            space=ImageSpace.FOURIER_REAL,
            reference=ref_real,
            prior_mean=None if prior_mean is None else prior_mean.real,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )
        imag_results = self.irls_imag.fit(
            batch=batch,
            space=ImageSpace.FOURIER_IMAG,
            reference=ref_imag,
            prior_mean=None if prior_mean is None else prior_mean.imag,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )

        fourier_estimate = torch.complex(real_results.estimate, imag_results.estimate)
        self.avg = torch.fft.irfft2(fourier_estimate, norm=batch.norm)
        weight_set = WeightSet(
            real=None,
            fourier_real=real_results.weights.fourier_real,
            fourier_imag=imag_results.weights.fourier_imag,
        )
        self.final_weights = weight_set.as_space_dict()

        return EstimatorResult(
            average=self.avg,
            estimate=fourier_estimate,
            weights=weight_set,
            converged=real_results.converged and imag_results.converged,
            n_iter=max(real_results.n_iter, imag_results.n_iter),
        )

    def _setup_reference(
        self, reference: torch.Tensor | None, norm: str = "ortho"
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        reference_real = None
        reference_imag = None
        if isinstance(reference, torch.Tensor):
            if torch.is_complex(reference):
                reference_real = reference.real
                reference_imag = reference.imag
            else:
                fourier_ref = torch.fft.rfft2(reference, norm=norm)
                reference_real = fourier_ref.real
                reference_imag = fourier_ref.imag

        return reference_real, reference_imag

    def reconstruct_from_weights(
        self,
        images: dict[ImageSpace, torch.Tensor | None] | ImageBatch,
        weights: dict[ImageSpace, torch.Tensor | None] | WeightSet,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX
    ) -> torch.Tensor:
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )
        
        reconstructed_fourier_real = self.irls_real.reconstruct_from_weights(
            images, weights, space=ImageSpace.FOURIER_REAL
        )
        reconstructed_fourier_imag = self.irls_imag.reconstruct_from_weights(
            images, weights, space=ImageSpace.FOURIER_IMAG
        )

        norm = images.norm if isinstance(images, ImageBatch) else "ortho"
        return torch.fft.irfft2(
            torch.complex(reconstructed_fourier_real, reconstructed_fourier_imag),
            norm=norm,
        )


class JointIRLSFourier(Estimator):
    def __init__(self, solver: IRLSSolver, device=None):
        super().__init__(device)

        self.solver = solver
        assert self.solver.space == ImageSpace.FOURIER_COMPLEX
        self.max_iter = self.solver.max_iter
        self.space = ImageSpace.FOURIER_COMPLEX

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

    @torch.inference_mode()
    def fit(
        self,
        batch: ImageBatch,
        *,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX,
        reference: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
        max_iter_override: int | None = None,
    ):
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )
        irls_result = self.solver.fit(
            batch,
            space=ImageSpace.FOURIER_COMPLEX,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )

        self.final_weights = irls_result.weights.as_space_dict()
        self.avg = torch.fft.irfft2(irls_result.estimate, norm="ortho")

        return EstimatorResult(
            average=self.avg,
            estimate=irls_result.estimate,
            weights=irls_result.weights,
            converged=irls_result.converged,
            n_iter=irls_result.n_iter,
        )

    @torch.inference_mode()
    def reconstruct_from_weights(
        self,
        images: dict[ImageSpace, torch.Tensor] | ImageBatch,
        weights: dict[ImageSpace, torch.Tensor] | WeightSet,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX
    ):
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )
        
        fourier_reconstruction = self.solver.reconstruct_from_weights(
            images, weights, space=ImageSpace.FOURIER_COMPLEX
        )
        norm = images.norm if isinstance(images, ImageBatch) else "ortho"
        return torch.fft.irfft2(fourier_reconstruction, norm=norm)


def flatten_complex_batch(batch: torch.Tensor) -> torch.Tensor:
    n = batch.shape[0]
    return torch.view_as_real(batch).reshape(n, -1)


def flatten_complex_tensor(v: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(v).reshape(-1)


def unflatten_complex_tensor(
    v: torch.Tensor, original_shape: tuple[int, ...]
) -> torch.Tensor:
    return torch.view_as_complex(v.reshape(*original_shape, 2))


def expand_real_batch_to_flat_complex_batch(batch: torch.Tensor) -> torch.Tensor:
    n = batch.shape[0]
    batch = batch.unsqueeze(-1)
    batch = batch.expand(*batch.shape, 2)
    batch = batch.reshape(n, -1)
    return batch


def expand_real_tensor_to_flat_complex_tensor(v: torch.Tensor) -> torch.Tensor:
    v = v.unsqueeze(-1)
    v = v.expand(*v.shape, 2)
    v = v.reshape(-1)
    return v


class FlatteningIRLSFourier(Estimator):
    def __init__(self, solver: IRLSSolver, device=None):
        super().__init__(device)

        self.solver = solver
        self.max_iter = self.solver.max_iter
        self.space = ImageSpace.FOURIER_COMPLEX

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

    @torch.inference_mode()
    def fit(
        self,
        batch: ImageBatch,
        *,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX,
        reference: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
        max_iter_override: int | None = None,
    ):
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )
        (
            fourier_images_realimag,
            fourier_shape,
            variance_realimag,
            ctf,
            reference_realimag,
            prior_mean_realimag,
            prior_variance,
        ) = self._setup_flat_data(
            batch,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

        irls_results = self.solver.fit_tensor(
            images=fourier_images_realimag,
            image_variance=variance_realimag,
            ctf=ctf,
            reference=reference_realimag,
            prior_mean=prior_mean_realimag,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )

        # Recover fourier space estimate
        fourier_estimate_realimag = irls_results.estimate
        fourier_estimate = unflatten_complex_tensor(
            fourier_estimate_realimag, original_shape=fourier_shape
        )

        # Recover final weights
        weights = irls_results.weights.select_space_weights(self.solver.space)
        # Reshape weights to the standard (N, 1, 1) format
        weights = weights.reshape(batch.n_images, 1, 1)
        weight_set = WeightSet(real=None, fourier_real=weights, fourier_imag=weights)

        self.final_weights = weight_set.as_space_dict()
        self.avg = torch.fft.irfft2(fourier_estimate, norm="ortho")
        return EstimatorResult(
            average=self.avg,
            estimate=fourier_estimate,
            weights=weight_set,
            converged=irls_results.converged,
            n_iter=irls_results.n_iter,
        )

    def _setup_flat_data(
        self,
        batch: ImageBatch,
        *,
        reference: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | float | None = None,
    ) -> tuple[
        torch.Tensor,  # fourier images flattened
        tuple[int, ...],  # fourier shape
        torch.Tensor | None,  # flattened reference
        torch.Tensor | None,  # flattened prior mean
        torch.Tensor | float | None,  # flattened prior variance
    ]:
        fourier_images = batch.select_space_images(ImageSpace.FOURIER_COMPLEX)
        fourier_shape = tuple(fourier_images[0].shape)
        fourier_images_realimag = flatten_complex_batch(fourier_images)

        variance = torch.complex(*batch.fourier_component_variances())
        variance_realimag = flatten_complex_tensor(variance)

        ctf = batch.ctf
        if isinstance(ctf, torch.Tensor) and ctf.ndim == fourier_images.ndim:
            ctf = expand_real_batch_to_flat_complex_batch(ctf)

        if reference is None:
            reference_realimag = None
        elif isinstance(reference, torch.Tensor):
            if not torch.is_complex(reference):
                reference = torch.fft.rfft2(reference, norm=batch.norm)
            reference_realimag = flatten_complex_tensor(reference)

        if prior_mean is None:
            prior_mean_realimag = None
        else:
            if not torch.is_complex(prior_mean):
                prior_mean = torch.fft.rfft2(prior_mean, norm=batch.norm)
            prior_mean_realimag = flatten_complex_tensor(prior_mean)

        if (
            isinstance(prior_variance, torch.Tensor)
            and prior_variance.shape == fourier_images[0].shape
        ):
            prior_variance = expand_real_tensor_to_flat_complex_tensor(prior_variance)

        return (
            fourier_images_realimag,
            fourier_shape,
            variance_realimag,
            ctf,
            reference_realimag,
            prior_mean_realimag,
            prior_variance,
        )

    @torch.inference_mode()
    def reconstruct_from_weights(
        self,
        images: dict[ImageSpace, torch.Tensor] | ImageBatch,
        weights: dict[ImageSpace, torch.Tensor] | WeightSet,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX
    ):
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )
        
        if isinstance(images, dict):
            fourier_images = torch.complex(
                images[ImageSpace.FOURIER_REAL], images[ImageSpace.FOURIER_IMAG]
            )
        else:
            fourier_images = images.select_space_images(ImageSpace.FOURIER_COMPLEX)

        n = fourier_images.shape[0]
        fourier_shape = tuple(fourier_images.shape[1:])
        fourier_images_realimag = flatten_complex_batch(fourier_images)

        if isinstance(weights, dict):
            weights = weights[ImageSpace.FOURIER_REAL]
        elif isinstance(weights, WeightSet):
            weights = weights.fourier_real

        # Reshape weights from (n, 1, 1) convention to (n, 1) for weighted average
        weights = weights.reshape(n, 1)

        realimag_estimate = weighted_average(fourier_images_realimag, weights=weights)
        fourier_estimate = unflatten_complex_tensor(realimag_estimate, fourier_shape)

        norm = images.norm if isinstance(images, ImageBatch) else "ortho"
        return torch.fft.irfft2(fourier_estimate, norm=norm)
