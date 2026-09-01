"""
GMM-based robust estimator.

This module implements a recursive robust averaging method that fits a two-component
Gaussian mixture model to distances/dissimilarities to the current reference and uses
the responsibilities of the closest component as image weights.
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import matplotlib.pyplot as plt

from .base import Estimator
from .weights import weighted_average
from .distances import DistanceFunction
from .data import ImageBatch
from .results import EstimatorResult, WeightSet

from cryo_robust.domain import ImageSpace


class RecursiveGMMEstimator(Estimator):
    """Recursive robust averaging estimator based on GMM responsibilities."""

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

        self.gmm_max_iter = gmm_max_iter
        self.gmm_tol = gmm_tol

        self.n_its = None
        self.converged = False

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
        """Initialize component weights and means from the observed distances.

        The lower-distance component is initialized with weight 0.8 and its mean at
        the 0.2 quantile. The higher-distance component is initialized with weight
        0.2 and its mean at the 0.8 quantile.
        """
        component_weights = torch.tensor(
            [0.8, 0.2],
            dtype=distances.dtype,
            device=distances.device,
        )

        component_means = torch.quantile(
            distances.reshape(-1),
            1.0 - component_weights,
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
    ) -> torch.Tensor:
        """Return posterior probabilities of the lower-distance GMM component."""
        good_component = np.argmin(model.means_.mean(axis=1))
        responsibilities = model.predict_proba(distances_np)[:, good_component]

        # .view(-1, 1, 1) allows the weights to broadcast over image batches.
        # NOTE: this would need to be modified to generalize to other image dimensions.
        return torch.as_tensor(
            responsibilities,
            dtype=dtype,
            device=device,
        ).view(-1, 1, 1)

    def _fit_one_iteration(
        self,
        images: torch.Tensor,
        reference: torch.Tensor,
        initialize_params: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, bool, float, float]:
        """Perform one iteration of the recursive GMM estimation procedure."""
        distances = self.distance_function(images, reference)
        std_distances, dist_mean, dist_std = self._standardize(distances)

        # Prepare distances for sklearn's GaussianMixture
        if std_distances.ndim == 1:
            std_distances = std_distances[:, None]

        if initialize_params:
            self._initialize_model_params(std_distances)

        distances_np = std_distances.detach().cpu().numpy()

        # Fit GMM to the distance distribution. With warm_start=True, later
        # recursive iterations start from the previous iteration's fitted model.
        self.model.fit(distances_np)

        # Get weights and update reference
        weights = self._responsibility_weights(
            self.model,
            distances_np,
            dtype=images.dtype,
            device=images.device,
        )
        next_reference = weighted_average(images, weights)
        rel_change = torch.linalg.norm(next_reference - reference) / (
            torch.linalg.norm(reference) + 1.0e-8
        )

        return (
            distances_np,
            weights,
            next_reference,
            bool(rel_change < self.tol),
            dist_mean,
            dist_std,
        )

    @torch.inference_mode()
    def fit(
        self,
        images: ImageBatch | torch.Tensor,
        reference: torch.Tensor | None = None,
        initialize_params: bool = False,
        plot_fits: bool = False,
        plot_title: str = "GMM Distances & Fit",
    ) -> EstimatorResult:
        """Fit the recursive estimator and return its result."""
        # Reset the GMM to avoid carrying over state from previous fit() calls
        self.model = self._new_model()

        # Select real-space images
        if isinstance(images, ImageBatch):
            images = images.real

        # Get initial reference
        reference = (
            images.mean(dim=0) if reference is None else reference.to(images.device)
        )

        self.converged = False
        for i in range(self.max_iter):
            distances_np, weights, next_reference, converged, dist_mean, dist_std = (
                self._fit_one_iteration(
                    images,
                    reference,
                    initialize_params=initialize_params and i == 0,
                )
            )

            # Update reference
            reference = next_reference

            # Plot initial fit
            if i == 0 and plot_fits:
                fig, axes = self._produce_initial_diagnostics(
                    distances_np,
                    dist_mean,
                    dist_std,
                    plot_title,
                )

            # Check convergence
            if converged:
                self.converged = True
                break

        # Save results using the existing gmm.py data model
        weight_set = WeightSet(real=weights, fourier_real=None, fourier_imag=None)

        # Plot final model fit
        if plot_fits:
            ax = axes[1]
            self._produce_final_diagnostics(
                ax,
                distances_np,
                dist_mean,
                dist_std,
            )
            fig.tight_layout()

        return EstimatorResult(
            average=reference,
            estimate=reference,
            weights=weight_set,
        )

    def _produce_initial_diagnostics(
        self,
        distances_to_ref_np: np.ndarray,
        avg_distance: float,
        std_distance: float,
        plot_title: str,
    ) -> tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.suptitle(plot_title)
        ax = axes[0]
        plot_gmm_fit(
            ax,
            distances_to_ref_np,
            self.model,
            plot_overall_model_pdf=True,
            plot_each_component=True,
            avg_distance=avg_distance,
            std_distance=std_distance,
            negate_distance=True,
        )
        ax.set_title("1st iteration")

        # Fit a one component GMM and compare AIC with two component GMM
        one_comp_model = GaussianMixture(n_init=10, init_params="k-means++")
        one_comp_model.fit(distances_to_ref_np)

        # Fit comparison
        print("GMM Fit Comparison")
        print("AIC:")
        print(f"- One comp: {one_comp_model.aic(distances_to_ref_np)}")
        print(f"- Two comp: {self.model.aic(distances_to_ref_np)}")
        print("BIC:")
        print(f"- One comp: {one_comp_model.bic(distances_to_ref_np)}")
        print(f"- Two comp: {self.model.bic(distances_to_ref_np)}")

        return fig, axes

    def _produce_final_diagnostics(
        self,
        ax: plt.Axes,
        distances_to_ref_np: np.ndarray,
        avg_distance: float,
        std_distance: float,
    ) -> None:
        plot_gmm_fit(
            ax,
            distances_to_ref_np,
            self.model,
            plot_overall_model_pdf=True,
            avg_distance=avg_distance,
            std_distance=std_distance,
            negate_distance=True,
        )
        ax.set_title("Last iteration")

    def reconstruct_from_weights(
        self, images: ImageBatch, weights: WeightSet
    ) -> torch.Tensor:
        return weighted_average(images.ensure_real(), weights.real)


def plot_gmm_fit(
    ax,
    distances: np.ndarray,
    model: GaussianMixture,
    plot_distances: bool = True,
    plot_each_component: bool = True,
    plot_overall_model_pdf: bool = False,
    avg_distance: np.float64 | float | None = None,
    std_distance: np.float64 | float | None = None,
    negate_distance: bool = False,
) -> None:
    """Helper to overlay GMM probability density function on a histogram."""
    z = np.linspace(distances.min() * 0.9, distances.max() * 1.1, 1000)

    if std_distance is None:
        std_distance = 1

    if avg_distance is None:
        avg_distance = 0

    if negate_distance:
        negate_mult = -1
    else:
        negate_mult = 1

    z = np.linspace(distances.min() * 0.9, distances.max() * 1.1, 1000)
    x = negate_mult * ((z * std_distance) + avg_distance)
    original_distances = negate_mult * ((distances * std_distance) + avg_distance)

    if plot_distances:
        ax.hist(original_distances, density=True)

    # Plot the individual Gaussian components
    if plot_each_component:
        for i in range(model.n_components):
            # Get mean and variance in original distance scale
            mean = negate_mult * ((model.means_[i, 0] * std_distance) + avg_distance)
            var = model.covariances_[i, 0, 0] * (std_distance**2)

            # Plot the component's pdf
            weight = model.weights_[i]
            pdf = weight * stats.norm.pdf(x, mean, np.sqrt(var))
            ax.plot(
                x,
                pdf,
                linestyle="--",
                linewidth=2,
                label=f"Gaussian {i+1} (w={weight:.2f})",
            )

    # Plot the overall model pdf
    if plot_overall_model_pdf:
        pdf = np.exp(model.score_samples(z.reshape(-1, 1))) / np.abs(std_distance)
        ax.plot(x, pdf, linestyle="--", linewidth=2, label="Aggregated GMM density")

    ax.set_ylabel("Density")
    ax.legend()
