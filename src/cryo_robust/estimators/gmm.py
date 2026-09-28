"""
GMM-based robust estimator.

This module implements a recursive robust averaging method that fits a two-component
Gaussian mixture model to distances/dissimilarities to the current reference and uses
the responsibilities of the closest component as image weights.
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .base import Estimator
from .weights import weighted_average
from .distances import DistanceFunction
from .data import ImageBatch
from .results import EstimatorResult, WeightSet, GMMDiagnostics

from cryo_robust.domain import ImageSpace

MIN_ELEMENTS_FOR_GMM = 50
GMM_MIN_SEPARATION = 0.20
GMM_MIN_GOOD_COMPONENT_WEIGHT = 0.50


class RecursiveGMMEstimator(Estimator):
    """Recursive robust averaging with peak-corrected GMM responsibilities.

    Unusable iterations stop the recursion and return the ordinary image mean
    with unit weights. ``fallback_reason`` records why this happened; it is
    ``None`` after a successful fit. GMM parameters in the diagnostics are NaN
    when the final attempted iteration did not produce a valid model.

    The good component has the lower mean distance. By default, reject models
    whose mean separation divided by ``sqrt(var_1 + var_2)`` is below a certain
    threshold (``GMM_MIN_SEPARATION``), or whose good component has mixture
    weight below ``GMM_MIN_GOOD_COMPONENT_WEIGHT``.
    """

    def __init__(
        self,
        distance_function: DistanceFunction,
        max_iter: int = 1,
        tol: float = 1.0e-4,
        standardize_distances: bool = True,
        space: ImageSpace = ImageSpace.REAL,
        random_state: int | None = None,
        gmm_max_iter: int = 20,
        gmm_tol: float = 1.0e-4,
        initialize_params: bool = True,
        check_degenerate_model: bool = True,
        min_component_separation: float = GMM_MIN_SEPARATION,
        min_good_component_weight: float = GMM_MIN_GOOD_COMPONENT_WEIGHT,
    ):
        self.model = GaussianMixture(
            n_components=2,
            max_iter=gmm_max_iter,
            tol=gmm_tol,
            random_state=random_state,
            warm_start=True,
        )

        self.distance_function = distance_function
        self.max_iter = max_iter
        self.tol = tol
        self.standardize_distances = standardize_distances
        self.space = space
        self.initialize_params = initialize_params
        self.check_degenerate_model = check_degenerate_model
        self.min_component_separation = min_component_separation
        self.min_good_component_weight = min_good_component_weight

        self.gmm_max_iter = gmm_max_iter
        self.gmm_tol = gmm_tol

        self.n_its = None
        self.converged = False
        self.fallback_reason: str | None = None

    def _new_model(self) -> GaussianMixture:
        """Create a fresh GMM, resetting any state from previous ``fit`` calls."""
        return GaussianMixture(
            n_components=2,
            max_iter=self.gmm_max_iter,
            tol=self.gmm_tol,
            random_state=self.model.random_state,
            warm_start=True,
        )

    def _initialize_model_params(self, distances: torch.Tensor) -> None:
        """
        Initialize component weights and means from the observed distances.

        Uses weights (0.95, 0.05) and means at the (0.50, 0.95) quantiles. Uses
        existing common empirical variance initialization for sklearn's
        full-covariance model.
        """
        component_weights = torch.tensor(
            [0.95, 0.05],
            dtype=distances.dtype,
            device=distances.device,
        )

        component_means = torch.quantile(
            distances.reshape(-1),
            torch.tensor([0.5, 0.95], dtype=distances.dtype, device=distances.device),
        )

        # Initialize both components with the same empirical variance
        variance = (
            distances.reshape(-1).var(unbiased=False).clamp_min(self.model.reg_covar)
        )
        component_precisions = (1.0 / variance).expand(2, 1, 1)

        self.model.means_init = component_means.reshape(2, 1).detach().cpu().numpy()
        self.model.weights_init = component_weights.detach().cpu().numpy()
        self.model.precisions_init = component_precisions.detach().cpu().numpy()

    def _standardize(
        self, distances: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        """Optionally standardize distances to zero mean and unit variance."""
        if not self.standardize_distances:
            return distances, 0.0, 1.0

        std = distances.std().clamp_min(1.0e-8)
        mean = distances.mean()

        return (distances - mean) / std, mean.item(), std.item()

    def _responsibility_weights(
        self,
        model: GaussianMixture,
        distances_np: np.ndarray,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior probabilities of the lower-distance GMM component."""
        good_component = np.argmin(model.means_.mean(axis=1))
        responsibilities = model.predict_proba(distances_np)[:, good_component]

        order = np.argsort(distances_np.reshape(-1), kind="stable")
        sorted_raw = responsibilities[order]
        peak_idx = int(sorted_raw.argmax())
        sorted_peak = sorted_raw.copy()
        sorted_peak[:peak_idx] = sorted_raw[peak_idx]
        sorted_peak[peak_idx:] = np.minimum.accumulate(sorted_raw[peak_idx:])
        peak_weights = np.empty_like(responsibilities)
        peak_weights[order] = sorted_peak

        return (
            torch.as_tensor(responsibilities, dtype=dtype, device=device),
            torch.as_tensor(peak_weights, dtype=dtype, device=device),
        )

    def _degeneracy_checks(self) -> tuple[bool, bool]:
        """Check normalized mean separation and the good component's mass."""
        means = self.model.means_[:, 0]
        variances = self.model.covariances_[:, 0, 0]
        separation = abs(means[1] - means[0]) / np.sqrt(variances.sum())
        good_component = np.argmin(means)

        return (
            bool(separation < self.min_component_separation),
            bool(self.model.weights_[good_component] < self.min_good_component_weight),
        )

    def _make_diagnostics(
        self,
        reference: torch.Tensor,
        distances: torch.Tensor,
        weights: torch.Tensor,
        next_reference: torch.Tensor,
        converged: bool = False,
        model_valid: bool = True,
    ) -> GMMDiagnostics:
        """Keep the existing diagnostics schema, including on fallback paths."""
        missing = np.full(2, np.nan)
        if torch.is_complex(reference):
            initial_reference = torch.fft.irfft2(reference)
        else:
            initial_reference = reference
        return GMMDiagnostics(
            initial_reference=initial_reference,
            distances=distances,
            standardized_distances=self.standardize_distances,
            component_weights=(
                self.model.weights_.copy() if model_valid else missing.copy()
            ),
            means=self.model.means_[:, 0].copy() if model_valid else missing.copy(),
            variances=(
                self.model.covariances_[:, 0, 0].copy()
                if model_valid
                else missing.copy()
            ),
            converged=converged,
            weights=weights,
            weighted_average=next_reference,
        )

    def _fit_one_iteration(
        self,
        images: torch.Tensor,
        reference: torch.Tensor,
        initialize_params: bool = False,
    ) -> GMMDiagnostics:
        """Perform one iteration of the recursive GMM estimation procedure."""
        distances = self.distance_function(images, reference)
        if distances.is_complex() or distances.numel() != images.shape[0]:
            raise ValueError(
                "The distance function must return one real distance per image"
            )
        weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)

        def fallback(reason: str, model_valid: bool = False) -> GMMDiagnostics:
            self.fallback_reason = reason
            average = images.mean(dim=0)
            if not torch.isfinite(average).all():
                raise ValueError("Cannot fall back to a non-finite image mean")
            return self._make_diagnostics(
                reference,
                distances,
                torch.ones(weight_shape, dtype=images.real.dtype, device=images.device),
                average,
                model_valid=model_valid,
            )

        if distances.numel() < MIN_ELEMENTS_FOR_GMM:
            return fallback("too_few_distances")
        if not torch.isfinite(distances).all():
            return fallback("nonfinite_distances")
        if (
            not torch.isfinite(distances.std(unbiased=False))
            or distances.max() - distances.min() < 1.0e-8
        ):
            return fallback("constant_distances")

        std_distances, _, _ = self._standardize(distances.reshape(-1))
        if not torch.isfinite(std_distances).all():
            return fallback("nonfinite_distances")

        # Prepare distances for sklearn's GaussianMixture
        if std_distances.ndim == 1:
            std_distances = std_distances[:, None]

        if initialize_params:
            self._initialize_model_params(std_distances)

        distances_np = std_distances.detach().cpu().numpy()

        # Fit GMM to the distance distribution. With warm_start=True, later
        # recursive iterations start from the previous iteration's fitted model.
        self.model.fit(distances_np)

        parameters = (self.model.means_, self.model.covariances_, self.model.weights_)
        if not (
            all(np.isfinite(p).all() for p in parameters)
            and (self.model.covariances_ > 0).all()
            and (self.model.weights_ > 0).all()
            and np.isclose(self.model.weights_.sum(), 1.0)
        ):
            return fallback("invalid_gmm")

        # Get weights and update reference
        raw_responsibilities, weights = self._responsibility_weights(
            self.model,
            distances_np,
            dtype=torch.float32,
            device=images.device,
        )

        if not all(
            torch.isfinite(w).all() and bool(((w >= 0) & (w <= 1)).all())
            for w in (raw_responsibilities, weights)
        ):
            return fallback("invalid_weights", model_valid=True)

        if self.check_degenerate_model and any(self._degeneracy_checks()):
            return fallback("degenerate_components", model_valid=True)

        weight_sum = weights.sum()
        if (
            not torch.isfinite(weight_sum)
            or weights.mean() < 1.0e-4
            or weight_sum < 0.05 * raw_responsibilities.sum()
        ):
            return fallback("collapsed_weights", model_valid=True)

        weights = weights.reshape(weight_shape)
        next_reference = weighted_average(images, weights, eps=0)
        if not torch.isfinite(next_reference).all():
            return fallback("invalid_reference", model_valid=True)
        rel_change = torch.linalg.norm(next_reference - reference) / (
            torch.linalg.norm(reference) + 1.0e-8
        )

        return self._make_diagnostics(
            reference,
            distances,
            weights,
            next_reference,
            converged=bool(rel_change < self.tol),
        )

    @torch.inference_mode()
    def solve(
        self, images: ImageBatch | torch.Tensor, reference: torch.Tensor | None
    ) -> tuple[torch.Tensor, GMMDiagnostics]:
        # Reset the GMM to avoid carrying over state from previous solve() calls
        self.model = self._new_model()

        # Select real-space images
        if isinstance(images, ImageBatch):
            images = images.ensure_real()
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative")

        # Get initial reference
        reference = (
            images.mean(dim=0) if reference is None else reference.to(images.device)
        )

        self.converged = False
        self.n_its = 0
        self.fallback_reason = "no_iterations" if self.max_iter == 0 else None

        if self.max_iter == 0:
            weights = torch.ones(
                (images.shape[0],) + (1,) * (images.ndim - 1),
                dtype=images.real.dtype,
                device=images.device,
            )
            diagnostics = self._make_diagnostics(
                reference,
                torch.zeros_like(weights),
                weights,
                reference,
                model_valid=False,
            )

        for i in range(self.max_iter):
            self.n_its = i + 1
            diagnostics = self._fit_one_iteration(
                images,
                reference,
                initialize_params=self.initialize_params and i == 0,
            )

            if diagnostics.weighted_average is None:
                raise RuntimeError(
                    "RecursiveGMMEstimator returned None as its weighted average"
                )

            reference = diagnostics.weighted_average
            if diagnostics.converged:
                self.converged = True
                break

        return reference, diagnostics

    @torch.inference_mode()
    def fit(
        self, images: ImageBatch | torch.Tensor, reference: torch.Tensor | None = None
    ) -> EstimatorResult:
        """Fit the recursive estimator and return its result."""
        reference, diagnostics = self.solve(images, reference)

        # Save results using the existing data model
        weight_set = WeightSet(
            real=diagnostics.weights, fourier_real=None, fourier_imag=None
        )

        estimator_result = EstimatorResult(
            average=reference,
            estimate=reference,
            weights=weight_set,
            gmm_diagnostics=diagnostics,
        )

        return estimator_result

    def reconstruct_from_weights(
        self, images: ImageBatch, weights: WeightSet
    ) -> torch.Tensor:
        return weighted_average(images.ensure_real(), weights.real, eps=0.0)


class FourierGMM(Estimator):
    def __init__(self, solver: RecursiveGMMEstimator):
        self.solver = solver
        self.space = ImageSpace.FOURIER_COMPLEX

    def fit(
        self,
        images: ImageBatch,
        reference: torch.Tensor,
    ) -> EstimatorResult:
        fourier_images = images.ensure_fourier()
        if reference is None:
            reference = fourier_images.mean(dim=0)
        elif not torch.is_complex(reference):
            reference = torch.fft.rfft2(reference, norm=images.norm)

        fourier_estimate, diagnostics = self.solver.solve(fourier_images, reference)
        estimate = torch.fft.irfft2(fourier_estimate, norm=images.norm)

        # Save results using the existing data model
        weight_set = WeightSet(
            real=None,
            fourier_real=diagnostics.weights,
            fourier_imag=diagnostics.weights,
        )

        return EstimatorResult(
            average=estimate,
            estimate=estimate,
            weights=weight_set,
            gmm_diagnostics=diagnostics,
        )
