import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Callable, Dict, Tuple
from estimators.base import Estimator
from utils.space import Space
import matplotlib.pyplot as plt
import scipy.stats as stats


@torch.no_grad()
def weighted_average(
    y: torch.Tensor, weights: torch.Tensor, dim: int = 0, eps: float = 0.0
) -> torch.Tensor:
    return (weights * y).sum(dim=dim) / (weights.sum(dim=0) + eps)


class GMMEstimator(Estimator):
    def __init__(
        self,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        random_state: int | None = None,
        standardize_distances: bool = True,
        space: Space = Space.REAL,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.model = GaussianMixture(n_components=2, random_state=random_state)
        self.distance_function = distance_function
        self.space = space
        self.standardize_distances = standardize_distances

        # Possible extensions:
        # hard binary classification with a threshold
        # classify on more than one property
        # iterative recursive estimation

    @torch.inference_mode()
    def fit(
        self,
        images: Dict[Space, torch.Tensor] | torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[Space, torch.Tensor]]:
        # Ensure images are a pytorch tensor on the correct device
        if isinstance(images, dict):
            images = images[self.space]
        images = self._prepare_data(images)

        # Get initial reference
        if reference is None:
            reference = images.mean(dim=0)

        # Calculate distances to reference
        distances_to_ref = self.distance_function(images, reference)
        if self.standardize_distances:
            distances_to_ref = (
                distances_to_ref - distances_to_ref.mean()
            ) / distances_to_ref.std()

        # Prepare distances for GMM (numpy array of shape (n_samples, n_features))
        mean_distances_np = distances_to_ref.detach().cpu().numpy().reshape(-1, 1)

        # Fit GMM model to the distances
        self.model.fit(mean_distances_np)

        # Identify "good" class and get predictions
        idx_good = np.argmin(self.model.means_)
        predicted_proba = self.model.predict_proba(mean_distances_np)
        weights = torch.tensor(predicted_proba.T[idx_good]).view(-1, 1, 1)

        # Weighted average according to predicted responsibilities
        self.avg = weighted_average(images, weights)

        # Store final weights in the standard format
        self.final_weights = {
            Space.REAL: weights,
            Space.FOURIER_REAL: None,
            Space.FOURIER_IMAG: None,
        }

        return self.avg, self.final_weights


class RecursiveGMMEstimator(Estimator):
    def __init__(
        self,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        random_state: int | None = None,
        max_iter: int = 1,
        tol: float = 1e-3,
        standardize_distances: bool = True,
        space: Space = Space.REAL,
        device: str = "cpu",
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

        # Possible extensions:
        # hard binary classification with a threshold
        # classify on more than one property
        # iterative recursive estimation

    @torch.inference_mode()
    def fit(
        self,
        images: Dict[Space, torch.Tensor] | torch.Tensor,
        reference: torch.Tensor | None = None,
        initialize_params: bool = False,
        plot_fits: bool = False,
        plot_title: str = "GMM Distances & Fit",
    ) -> Tuple[torch.Tensor, Dict[Space, torch.Tensor]]:
        # Ensure images are a pytorch tensor on the correct device
        if isinstance(images, dict):
            images = images[self.space]
        images = self._prepare_data(images)

        # Get initial reference
        if reference is None:
            print("Using image average as reference")
            reference = images.mean(dim=0)

        # Initiliazation of GMM params
        self.model.warm_start = True
        if initialize_params:
            n_features = 1
            self.model.means_init = np.array(
                [-np.ones(n_features), np.ones(n_features)]
            )
            self.model.weights_init = np.array([0.8, 0.2])

        self.converged = False
        for i in range(self.max_iter):
            # Calculate distances to the reference
            distances_to_ref = self.distance_function(images, reference)
            if self.standardize_distances:
                avg_distance = torch.mean(distances_to_ref)
                std_distance = torch.std(distances_to_ref)
                distances_to_ref = (distances_to_ref - avg_distance) / (std_distance)

            # Prepare distances for GMM (numpy array of shape (n_samples, n_features))
            distances_to_ref_np = distances_to_ref.detach().cpu().numpy()
            if distances_to_ref_np.ndim == 1:
                distances_to_ref_np = distances_to_ref_np.reshape(-1, 1)
            n_features = distances_to_ref_np.shape[1]

            # Fit GMM to distances
            self.model.fit(distances_to_ref_np)

            # Identify "good" class and get predictions
            idx_good = np.argmin(self.model.means_.mean(axis=1))
            predicted_proba = self.model.predict_proba(distances_to_ref_np)
            weights = torch.tensor(predicted_proba.T[idx_good]).view(-1, 1, 1)

            # Weighted average according to predicted responsibilities
            next_avg = weighted_average(images, weights)

            # Plot initial fit
            if i == 0 and plot_fits:
                fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
                fig.suptitle(plot_title)
                ax = axes[0]
                plot_gmm_fit(
                    ax,
                    distances_to_ref_np,
                    self.model,
                    plot_overall_model_pdf=True,
                    plot_each_component=True,
                )
                ax.set_title("1st iteration")

                # Fit a one component GMM and compare AIC with two component GMM
                one_comp_model = GaussianMixture(n_init=10, init_params="k-means++")
                one_comp_model.fit(distances_to_ref_np)

                plot_gmm_fit(
                    ax, distances_to_ref_np, one_comp_model, plot_distances=False
                )

                # Fit comparison
                print(f"GMM Fit Comparison")
                print(f"AIC:")
                print(f"- One comp: {one_comp_model.aic(distances_to_ref_np)}")
                print(f"- Two comp: {self.model.aic(distances_to_ref_np)}")
                print(f"BIC:")
                print(f"- One comp: {one_comp_model.bic(distances_to_ref_np)}")
                print(f"- Two comp: {self.model.bic(distances_to_ref_np)}")

            # Convergence check
            if torch.norm(next_avg - reference) < self.tol:
                reference = next_avg
                self.converged = True
                print(f"Achieved tolerance on iteration {i + 1}")
                break

            # Update reference for next iteration
            reference = next_avg

        # Save results
        self.avg = reference
        self.n_its = i + 1
        self.final_weights = {
            Space.REAL: weights,
            Space.FOURIER_REAL: None,
            Space.FOURIER_IMAG: None,
        }

        # Plot final model fit
        if plot_fits:
            ax = axes[1]
            plot_gmm_fit(ax, distances_to_ref_np, self.model, plot_overall_model_pdf=True)
            ax.set_title("Last iteration")
            fig.tight_layout()

        return self.avg, self.final_weights


def plot_gmm_fit(
    ax,
    distances: np.ndarray,
    model: GaussianMixture,
    plot_distances: bool = True,
    plot_each_component: bool = True,
    plot_overall_model_pdf: bool = False,
) -> None:
    """Helper to overlay GMM probability density function on a histogram."""
    x = np.linspace(distances.min() * 0.9, distances.max() * 1.1, 1000)

    if plot_distances:
        ax.hist(distances, density=True)

    # Plot the individual Gaussian components
    if plot_each_component:
        for i in range(model.n_components):
            mean = model.means_[i, 0]
            var = model.covariances_[i, 0, 0]
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
        pdf = np.exp(model.score_samples(x.reshape(-1, 1)))
        ax.plot(x, pdf, linestyle="--", linewidth=2, label="Aggregated GMM density")

    ax.set_ylabel("Density")
    ax.legend()
