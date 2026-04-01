import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Callable
from estimators.base import Estimator, Space


@torch.no_grad()
def weighted_average(y, weights, dim=0, eps: float = 0.0):
    return (weights * y).sum(dim=dim) / (weights.sum(dim=0) + eps)


class GMMEstimator(Estimator):
    def __init__(
        self,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        random_state: int | None = None,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.model = GaussianMixture(n_components=2, random_state=random_state)
        self.distance_function = distance_function

        # Possible extensions:
        # hard binary classification with a threshold
        # classify on more than one property
        # iterative recursive estimation

    def fit(self, images):
        # ensure images are a pytorch tensor on the correct device
        images = self._prepare_data(images)

        # Calculate average and distances
        reference_avg = images.mean(dim=0)
        mean_distances = self.distance_function(images, reference_avg)
        mean_distances_np = mean_distances.detach().cpu().numpy().reshape(-1, 1)

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


class RecursiveGMMEstimator(Estimator):
    def __init__(
        self,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        random_state: int | None = None,
        max_iter: int = 1,
        tol: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.model = GaussianMixture(
            n_components=2, random_state=random_state, init_params="k-means++"
        )
        self.distance_function = distance_function
        self.max_iter = max_iter
        self.n_its = None
        self.converged = False
        self.tol = tol

        # Possible extensions:
        # hard binary classification with a threshold
        # classify on more than one property
        # iterative recursive estimation

    def fit(self, images):
        # Ensure images are a pytorch tensor on the correct device
        images = self._prepare_data(images)

        reference_avg = images.mean(dim=0)

        self.converged = False
        for i in range(self.max_iter):
            mean_distances = self.distance_function(images, reference_avg)
            avg_distance = torch.mean(mean_distances)
            std_distance = torch.std(mean_distances)
            mean_distances = (mean_distances - avg_distance) / (std_distance)
            mean_distances_np = mean_distances.detach().cpu().numpy()
            if mean_distances_np.ndim == 1:
                mean_distances_np = mean_distances_np.reshape(-1, 1)
            n_features = mean_distances_np.shape[1]

            # Initiliaze means
            self.model.means_init = np.array(
                [-np.ones(n_features), np.ones(n_features)]
            )
            # print(
            #     "Initializing means to "
            #     f"{self.model.means_init}"
            # )

            self.model.weights_init = np.array([0.9, 0.1])
            # print(
            #     "Initializing weights to "
            #     f"{self.model.weights_init[0]:.4f}, {self.model.weights_init[1]:.4f}"
            # )

            self.model.fit(
                mean_distances_np,
            )

            # Identify "good" class and get predictions
            idx_good = np.argmin(self.model.means_.mean(axis=1))
            predicted_proba = self.model.predict_proba(mean_distances_np)
            weights = torch.tensor(predicted_proba.T[idx_good]).view(-1, 1, 1)

            # print(
            #     "Final means: "
            #     f"{self.model.means_[0, 0]:.4f}, {self.model.means_[1, 0]:.4f}"
            # )
            # print(
            #     "Final weights: "
            #     f"{self.model.weights_[0]:.4f}, {self.model.weights_[1]:.4f}\n"
            # )

            # Weighted average according to predicted responsibilities
            next_avg = weighted_average(images, weights)

            if torch.norm(next_avg - reference_avg) < self.tol:
                reference_avg = next_avg
                self.converged = True
                print(f"Achieved tolerance on iteration {i + 1}")
                break

            reference_avg = next_avg

        self.avg = reference_avg
        self.n_its = i + 1
        self.final_weights = {
            Space.REAL: weights,
            Space.FOURIER_REAL: weights,
            Space.FOURIER_IMAG: weights,
        }
