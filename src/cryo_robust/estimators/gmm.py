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

from cryo_robust.comparison.domain.enums import ImageSpace


class RecursiveGMMEstimator(Estimator):
    def __init__(
        self,
        distance_function: DistanceFunction,
        max_iter: int = 1,
        tol: float = 1.0e-4,
        standardize_distances: bool = True,
        space: ImageSpace = ImageSpace.REAL,
        device: str = "cpu",
        random_state: int | None = None,
    ):
        super().__init__(device=device)
        self.model = GaussianMixture(n_components=2, random_state=random_state)
        self.distance_function = distance_function
        self.max_iter = max_iter
        self.n_its = None
        self.converged = False
        self.tol = tol
        self.space = space
        self.standardize_distances = standardize_distances

    # @torch.inference_mode()
    # def fit_tensor(
    #     self,
    #     images: dict[ImageSpace, torch.Tensor] | torch.Tensor,
    #     reference: torch.Tensor | None = None,
    #     initialize_params: bool = False,
    #     plot_fits: bool = False,
    #     plot_title: str = "GMM Distances & Fit",
    # ) -> tuple[torch.Tensor, dict[ImageSpace, torch.Tensor]]:
    #     # Reset the GMM to avoid carrying over state from previous fit() calls
    #     self.model = GaussianMixture(
    #         n_components=2,
    #         random_state=self.model.random_state,
    #         warm_start=False,
    #     )

    #     # Ensure images are a pytorch tensor on the correct device
    #     if isinstance(images, dict):
    #         images = images[self.space]
    #     images = self._prepare_data(images)

    #     if self.pixel_normalize:
    #         per_pixel_std = images.std(dim=0) + 1.0e-8

    #     # Get initial reference
    #     if reference is None:
    #         print("Using image average as reference")
    #         reference = images.mean(dim=0)

    #     # Initilization of GMM params
    #     self.model.warm_start = True
    #     if initialize_params:
    #         n_features = 1
    #         self.model.means_init = np.array(
    #             [-np.ones(n_features), np.ones(n_features)]
    #         )
    #         self.model.weights_init = np.array([0.8, 0.2])

    #     self.converged = False
    #     for i in range(self.max_iter):
    #         # Calculate distances to the reference
    #         if self.pixel_normalize:
    #             distances_to_ref = self.distance_function(
    #                 images / per_pixel_std.unsqueeze(0), reference / per_pixel_std
    #             )
    #         else:
    #             distances_to_ref = self.distance_function(images, reference)

    #         # Initialize avg_distance and std_distance to avoid errors
    #         # in case self.standardize_distances is False
    #         avg_distance = torch.tensor(0.0, device=images.device)
    #         std_distance = torch.tensor(1.0, device=images.device)
    #         if self.standardize_distances:
    #             avg_distance = torch.mean(distances_to_ref)
    #             std_distance = torch.std(distances_to_ref)
    #             if std_distance < 1e-8:
    #                 print(
    #                     "Warning: near-zero std in distances, skipping standardization"
    #                 )
    #                 std_distance = torch.tensor(1.0, device=images.device)
    #             distances_to_ref = (distances_to_ref - avg_distance) / (std_distance)

    #         # Prepare distances for GMM (numpy array of shape (n_samples, n_features))
    #         distances_to_ref_np = distances_to_ref.detach().cpu().numpy()
    #         if distances_to_ref_np.ndim == 1:
    #             distances_to_ref_np = distances_to_ref_np.reshape(-1, 1)
    #         n_features = distances_to_ref_np.shape[1]

    #         # Fit GMM to distances
    #         self.model.fit(distances_to_ref_np)

    #         # Identify "good" class and get predictions
    #         idx_good = np.argmin(self.model.means_.mean(axis=1))
    #         predicted_proba = self.model.predict_proba(distances_to_ref_np)
    #         weights = torch.tensor(predicted_proba.T[idx_good]).view(-1, 1, 1)

    #         # Weighted average according to predicted responsibilities
    #         next_avg = weighted_average(images, weights)

    #         # Plot initial fit
    #         if i == 0 and plot_fits:
    #             fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #             fig.suptitle(plot_title)
    #             ax = axes[0]
    #             plot_gmm_fit(
    #                 ax,
    #                 distances_to_ref_np,
    #                 self.model,
    #                 plot_overall_model_pdf=True,
    #                 plot_each_component=True,
    #                 avg_distance=avg_distance.item(),
    #                 std_distance=std_distance.item(),
    #                 negate_distance=True,
    #             )
    #             ax.set_title("1st iteration")

    #             # Fit a one component GMM and compare AIC with two component GMM
    #             one_comp_model = GaussianMixture(n_init=10, init_params="k-means++")
    #             one_comp_model.fit(distances_to_ref_np)

    #             # plot_gmm_fit(
    #             #     ax, distances_to_ref_np, one_comp_model, plot_distances=False
    #             # )

    #             # Fit comparison
    #             print(f"GMM Fit Comparison")
    #             print(f"AIC:")
    #             print(f"- One comp: {one_comp_model.aic(distances_to_ref_np)}")
    #             print(f"- Two comp: {self.model.aic(distances_to_ref_np)}")
    #             print(f"BIC:")
    #             print(f"- One comp: {one_comp_model.bic(distances_to_ref_np)}")
    #             print(f"- Two comp: {self.model.bic(distances_to_ref_np)}")

    #         # Convergence check
    #         diff_norm = torch.norm(next_avg - reference)
    #         ref_norm = torch.norm(reference)
    #         if (diff_norm / (ref_norm + 1.0e-8)) < self.tol:
    #             reference = next_avg
    #             self.converged = True
    #             print(f"Achieved tolerance on iteration {i + 1}")
    #             break

    #         # Update reference for next iteration
    #         reference = next_avg

    #     # Save results
    #     self.avg = reference
    #     self.n_its = i + 1
    #     self.final_weights = {
    #         ImageSpace.REAL: weights,
    #         ImageSpace.FOURIER_REAL: None,
    #         ImageSpace.FOURIER_IMAG: None,
    #     }

    #     # Plot final model fit
    #     if plot_fits:
    #         ax = axes[1]
    #         plot_gmm_fit(
    #             ax,
    #             distances_to_ref_np,
    #             self.model,
    #             plot_overall_model_pdf=True,
    #             avg_distance=avg_distance.item(),
    #             std_distance=std_distance.item(),
    #             negate_distance=True,
    #         )
    #         ax.set_title("Last iteration")
    #         fig.tight_layout()

    #     return self.avg, self.final_weights

    def _new_model(self, initialize_params: bool) -> GaussianMixture:
        model = GaussianMixture(
            n_components=2,
            random_state=self.model.random_state,
            warm_start=True,
        )

        if initialize_params:
            n_features = 1
            model.means_init = np.array([-np.ones(n_features), np.ones(n_features)])
            model.weights_init = np.array([0.8, 0.2])

        return model

    def _standardize(
        self, distances: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        if not self.standardize_distances:
            return distances, 0.0, 1.0
        std = distances.std()
        mean = distances.mean()
        return (distances - mean) / std, mean.item(), std.item()

    def _responsibility_weights(
        self,
        model: GaussianMixture,
        distances_np: np.ndarray,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        good_component = np.argmin(model.means_.mean(axis=1))
        responsibilities = model.predict_proba(distances_np)[:, good_component]
        return torch.as_tensor(responsibilities, dtype=dtype, device=device).view(
            -1, 1, 1
        )

    def _fit_one_iteration(
        self, images: torch.Tensor, reference: torch.Tensor
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, bool]:
        distances = self.distance_function(images, reference)
        distances, dist_mean, dist_std = self._standardize(distances)

        # Prepare distances for sklearn's GaussianMixture
        distances_np = distances.detach().cpu().numpy()
        if distances_np.ndim == 1:
            distances_np = distances_np.reshape(-1, 1)

        # Fit GMM to the distance distribution
        self.model.fit(distances_np)

        # Get weights and update reference
        weights = self._responsibility_weights(
            self.model, distances_np, dtype=images.dtype, device=images.device
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
    ) -> tuple[torch.Tensor, dict[ImageSpace, torch.Tensor]]:
        # Reset the GMM to avoid carrying over state from previous fit() calls
        self.model = self._new_model(initialize_params)

        # Select real space images
        if isinstance(images, ImageBatch):
            images = images.real

        # Get initial reference
        reference = (
            images.mean(dim=0) if reference is None else reference.to(images.device)
        )

        self.converged = False
        for i in range(self.max_iter):
            distances_np, weights, next_reference, converged, dist_mean, dist_std = (
                self._fit_one_iteration(images, reference)
            )

            # Update reference
            reference = next_reference

            # Plot initial fit
            if i == 0 and plot_fits:
                fig, axes = self._produce_initial_diagnostics(
                    distances_np, dist_mean, dist_std, plot_title
                )

            # Check convergence
            if converged:
                self.converged = True
                break

        # Save results
        self.avg = reference
        weight_set = WeightSet(real=weights, fourier_real=None, fourier_imag=None)
        self.final_weights = weight_set.as_space_dict()

        # Plot final model fit
        if plot_fits:
            ax = axes[1]
            self._produce_final_diagnostics(ax, distances_np, dist_mean, dist_std)
            fig.tight_layout()

        return EstimatorResult(
            average=reference,
            estimate=reference,
            weights=weight_set,
            converged=self.converged,
            n_iter=i+1
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
        print(f"GMM Fit Comparison")
        print(f"AIC:")
        print(f"- One comp: {one_comp_model.aic(distances_to_ref_np)}")
        print(f"- Two comp: {self.model.aic(distances_to_ref_np)}")
        print(f"BIC:")
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
        self,
        images: dict[ImageSpace, torch.Tensor],
        weights: dict[ImageSpace, torch.Tensor | None],
    ) -> torch.Tensor:
        return weighted_average(images[ImageSpace.REAL], weights[ImageSpace.REAL])


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
