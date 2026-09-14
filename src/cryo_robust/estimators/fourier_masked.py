from typing import Iterable

from cryo_robust.domain import ImageSpace
from cryo_robust.estimators.base import Estimator
from cryo_robust.estimators.data import ImageBatch
from cryo_robust.estimators.irls import IRLSSolver
from cryo_robust.estimators.results import EstimatorResult, WeightSet
from cryo_robust.utils.masks import (
    create_lowpass_rfft_mask,
    create_bandpass_rfft_mask,
    create_highpass_rfft_mask,
)

import torch


def irls_update(
    images: torch.Tensor,
    weights: torch.Tensor,
    image_variance: torch.Tensor | None = None,
    ctf: torch.Tensor | None = None,
    prior_mean: torch.Tensor | None = None,
    prior_variance: torch.Tensor | None = None,
    eps: float = 1.0e-8,
):
    ctf_images = images if ctf is None else ctf * images
    s_1 = torch.sum(weights * ctf_images, dim=0)
    s_2 = (
        torch.sum(weights, dim=0)
        if ctf is None
        else torch.sum(weights * torch.as_tensor(ctf).square(), dim=0)
    )

    # No prior: weighted average with ctf
    if prior_mean is None or prior_variance is None:
        return s_1 / (s_2 + eps)

    # With prior: additional term
    if image_variance is None:
        raise ValueError(
            "Must provide image variance to irls_update if giving a prior term"
        )
    safe_image_variance = image_variance + eps
    safe_prior_variance = prior_variance + eps
    numer = s_1 / safe_image_variance + prior_mean / (safe_prior_variance)
    denom = s_2 / safe_image_variance + 1 / (safe_prior_variance)
    return numer / (denom + eps)


def average_weights(weights: Iterable[torch.Tensor | None]) -> torch.Tensor | None:
    n_weights = 0
    w_average = None
    for w in weights:
        if w is None:
            continue

        n_weights += 1
        if w_average is None:
            w_average = w.clone()
        else:
            w_average += w

    if n_weights:
        return w_average / n_weights

    return None


def average_weight_sets(weight_sets: Iterable[WeightSet]) -> WeightSet:
    return WeightSet(
        real=average_weights([weights.real for weights in weight_sets]),
        fourier_real=average_weights([weights.fourier_real for weights in weight_sets]),
        fourier_imag=average_weights([weights.fourier_imag for weights in weight_sets]),
    )


class MaskedIRLSFourier(Estimator):
    """
    Fourier estimator using one IRLS solver on complex Fourier coefficients, meant to
    operate on the modulus of the complex residual. Takes a ``mask`` parameter
    to operate only on a given set of frequencies.
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
        mask: torch.Tensor | None = None,
    ) -> EstimatorResult:
        if space != ImageSpace.FOURIER_COMPLEX:
            raise ValueError(
                f"Can only set {type(self)} space to {ImageSpace.FOURIER_COMPLEX.name}, "
                f"got {space.name}"
            )

        fourier_images = batch.ensure_fourier()

        if mask is not None:
            images_masked = fourier_images[:, mask]

            if prior_variance is not None:
                if isinstance(prior_variance, torch.Tensor) and prior_variance.ndim > 1:
                    prior_variance_masked = prior_variance[mask]
                else:
                    prior_variance_masked = prior_variance
            else:
                prior_variance_masked = None

            ctf_masked = None if batch.ctf is None else batch.ctf[mask]
            reference = None if reference is None else reference[mask]
            prior_mean_masked = None if prior_mean is None else prior_mean[mask]

        else:
            images_masked = fourier_images
            prior_mean_masked = prior_mean
            prior_variance_masked = prior_variance
            ctf_masked = batch.ctf

        irls_result = self.solver.fit_tensor(
            images=images_masked,
            ctf=ctf_masked,
            reference=reference,
            prior_mean=prior_mean_masked,
            prior_variance=prior_variance_masked,
            max_iter_override=max_iter_override,
        )

        weights = irls_result.weights.fourier_real

        n_images = fourier_images.shape[0]
        if weights.numel() != n_images:
            raise ValueError(
                "Number of weights in MaskedIRLSFourier does not match number of images. "
                "MaskedIRLSFourier only supports global weight functions."
            )

        image_ndims = fourier_images.ndim - 1
        weight_shape = (n_images,) + (1,) * (image_ndims)
        weights = weights.reshape(weight_shape)

        reconstructed_fourier = irls_update(
            images=fourier_images, weights=weights, ctf=batch.ctf
        )
        reconstructed_real = torch.fft.irfft2(reconstructed_fourier, norm=batch.norm)

        return EstimatorResult(
            average=reconstructed_real,
            estimate=reconstructed_fourier,
            weights=irls_result.weights,
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


class ThreeMasksFourier(Estimator):
    def __init__(
        self, low_cutoff: float, high_cutoff: float, solver: MaskedIRLSFourier
    ):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.solver = solver

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
    ) -> EstimatorResult:
        image_shape = batch.real_shape
        lowpass_mask = create_lowpass_rfft_mask(
            image_shape=image_shape, cutoff=self.low_cutoff, unit="normalized"
        )
        bandpass_mask = create_bandpass_rfft_mask(
            image_shape=image_shape,
            low_cutoff=self.low_cutoff,
            high_cutoff=self.high_cutoff,
            unit="normalized",
        )
        highpass_mask = create_highpass_rfft_mask(
            image_shape=image_shape, cutoff=self.high_cutoff, unit="normalized"
        )

        lowpass_results = self.solver.fit(
            batch=batch,
            space=space,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
            mask=lowpass_mask,
        )

        bandpass_results = self.solver.fit(
            batch=batch,
            space=space,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
            mask=bandpass_mask,
        )

        highpass_results = self.solver.fit(
            batch=batch,
            space=space,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
            mask=highpass_mask,
        )

        averaged_weights = average_weight_sets(
            [
                lowpass_results.weights,
                bandpass_results.weights,
                highpass_results.weights,
            ]
        )
        n_images = batch.n_images

        if averaged_weights.real is not None:
            averaged_weights.real = averaged_weights.real.reshape(n_images, 1, 1)

        if averaged_weights.fourier_real is not None:
            averaged_weights.fourier_real = averaged_weights.fourier_real.reshape(
                n_images, 1, 1
            )

        if averaged_weights.fourier_imag is not None:
            averaged_weights.fourier_imag = averaged_weights.fourier_imag.reshape(
                n_images, 1, 1
            )

        fourier_estimate = torch.zeros(size=batch.rfft2_shape, dtype=torch.complex64)

        fourier_estimate[lowpass_mask] = lowpass_results.estimate[lowpass_mask]
        fourier_estimate[bandpass_mask] = bandpass_results.estimate[bandpass_mask]
        fourier_estimate[highpass_mask] = highpass_results.estimate[highpass_mask]

        real_estimate = torch.fft.irfft2(fourier_estimate, norm=batch.norm)

        return EstimatorResult(
            estimate=fourier_estimate,
            average=real_estimate,
            weights=averaged_weights,
        )
