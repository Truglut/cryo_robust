from cryo_robust.domain import ImageSpace
from cryo_robust.estimators.base import Estimator
from cryo_robust.estimators.data import ImageBatch
from cryo_robust.estimators.irls import IRLSSolver
from cryo_robust.estimators.results import EstimatorResult, WeightSet


import torch

from cryo_robust.estimators.weights import weighted_average


class IRLSFourier(Estimator):
    """Fourier estimator using separate IRLS solvers for real and imaginary parts."""

    def __init__(self, irls_real: IRLSSolver, irls_imag: IRLSSolver):
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
        weight_set = WeightSet(
            real=None,
            fourier_real=real_results.weights.fourier_real,
            fourier_imag=imag_results.weights.fourier_imag,
        )

        return EstimatorResult(
            average=torch.fft.irfft2(fourier_estimate, norm=batch.norm),
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
        images: ImageBatch,
        weights: WeightSet,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX,  # only present for API compatibility
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

        return torch.fft.irfft2(
            torch.complex(reconstructed_fourier_real, reconstructed_fourier_imag),
            norm=images.norm,
        )


class JointIRLSFourier(Estimator):
    """
    Fourier estimator using one IRLS solver on complex Fourier coefficients, meant to
    operate on the modulus of the complex residual.
    """

    def __init__(self, solver: IRLSSolver):
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

        return EstimatorResult(
            average=torch.fft.irfft2(irls_result.estimate, norm=batch.norm),
            estimate=irls_result.estimate,
            weights=irls_result.weights,
            converged=irls_result.converged,
            n_iter=irls_result.n_iter,
        )

    @torch.inference_mode()
    def reconstruct_from_weights(
        self,
        images: ImageBatch,
        weights: WeightSet,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX,  # only present for API compatibility
    ):
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )

        fourier_reconstruction = self.solver.reconstruct_from_weights(
            images, weights, space=ImageSpace.FOURIER_COMPLEX
        )
        return torch.fft.irfft2(fourier_reconstruction, norm=images.norm)


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
    """
    Fourier estimator that applies real-valued IRLS to flattened complex coefficients.
    Necessary for applying the global weighting schemes to complex images.
    """

    def __init__(self, solver: IRLSSolver):
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
        weights = irls_results.weights.select_space(self.solver.space)
        # Reshape weights to the standard (N, 1, 1) format
        weights = weights.reshape(batch.n_images, 1, 1)
        weight_set = WeightSet(real=None, fourier_real=weights, fourier_imag=weights)

        return EstimatorResult(
            average=torch.fft.irfft2(fourier_estimate, norm=batch.norm),
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
        fourier_images = batch.select_space(ImageSpace.FOURIER_COMPLEX)
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
        images: ImageBatch,
        weights: WeightSet,
        space: ImageSpace = ImageSpace.FOURIER_COMPLEX,  # only present for API compatibility
    ):
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )

        fourier_images = images.select_space(ImageSpace.FOURIER_COMPLEX)
        n = images.n_images

        fourier_shape = tuple(fourier_images.shape[1:])
        fourier_images_realimag = flatten_complex_batch(fourier_images)

        weights = weights.fourier_real

        # Reshape weights from (n, 1, 1) convention to (n, 1) for weighted average
        weights = weights.reshape(n, 1)

        realimag_estimate = weighted_average(fourier_images_realimag, weights=weights)
        fourier_estimate = unflatten_complex_tensor(realimag_estimate, fourier_shape)

        return torch.fft.irfft2(fourier_estimate, norm=images.norm)
