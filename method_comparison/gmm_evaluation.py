import numpy as np
import torch
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from method_comparison.visualization.plotting import LABEL_MAP


def plot_gmm_fit(ax, distances, model, title):
    """Helper to overlay GMM probability density function on a histogram."""
    x = np.linspace(distances.min() * 0.9, distances.max() * 1.1, 1000)

    # Plot the individual Gaussian components
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

    ax.set_title(title)
    ax.set_ylabel("Density")
    ax.legend()


def evaluate_gmm_fits(results: dict, estimators: dict, images, labels: np.ndarray):
    """
    Recalculates distances and plots GMM fits for initial and final averages.
    'images' should be a PyTorch tensor on the correct device.
    """
    # Calculate the initial average
    image_average = images.mean(dim=0)

    for method_name, estimator in estimators.items():
        # Skip M-estimators (they don't use distance_function)
        if not hasattr(estimator, "distance_function"):
            continue

        if results[method_name].get("reference") is not None:
            initial_avg = torch.tensor(results[method_name]["reference"])
        else:
            initial_avg = image_average

        print(f"Processing GMM visualizations for: {method_name}")

        # Recalculate Initial Distances
        initial_dist = estimator.distance_function(images, initial_avg)
        initial_dist = (initial_dist - initial_dist.mean()) / initial_dist.std()
        initial_dist_np = initial_dist.detach().cpu().numpy().reshape(-1, 1)

        # Recalculate Final Distances
        final_avg = torch.tensor(results[method_name]["avg"])
        final_dist = estimator.distance_function(images, final_avg)
        final_dist = (final_dist - final_dist.mean()) / final_dist.std()
        final_dist_np = final_dist.detach().cpu().numpy().reshape(-1, 1)

        # Re-fit GMMs strictly for the visualization curves
        # random_state=42 should ensure equal results in most cases
        gmm_initial = GaussianMixture(n_components=2, random_state=42).fit(
            initial_dist_np
        )
        gmm_final = GaussianMixture(n_components=2, random_state=42).fit(final_dist_np)

        # Plot initial state and final state
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 4), sharex=True)
        for label_idx, config in LABEL_MAP.items():
            mask = labels == label_idx
            if mask.any():
                # Plot histogram of initial distances for this label group
                axes[0].hist(
                    initial_dist_np[mask],
                    bins=30,
                    alpha=0.5,
                    label=config["name"],
                    color=config["color"],
                    density=True,
                )

                # Plot histogram of final distances for this label group
                axes[1].hist(
                    final_dist_np[mask],
                    bins=30,
                    alpha=0.5,
                    label=config["name"],
                    color=config["color"],
                    density=True,
                )

        # Overlay estimated densities
        plot_gmm_fit(
            axes[0],
            initial_dist_np,
            gmm_initial,
            f"Initial Distances & Fit: {method_name}",
        )
        plot_gmm_fit(
            axes[1],
            final_dist_np,
            gmm_final,
            f"Final Distances & Fit: {method_name}",
        )

        plt.tight_layout()
        plt.show()


def evaluate_gmm_fits_unlabeled(results: dict, estimators: dict, images):
    """Plots GMM distance distributions without class labels."""
    print("\n" + "=" * 50 + "\nGENERATING GMM PLOTS\n" + "=" * 50)
    initial_avg = images.mean(dim=0)

    for method_name, estimator in estimators.items():
        if not hasattr(estimator, "distance_function"):
            continue

        initial_dist = (
            estimator.distance_function(images, initial_avg).detach().cpu().numpy()
        )
        initial_dist = (initial_dist - initial_dist.mean()) / initial_dist.std()
        initial_dist = initial_dist.reshape(-1, 1)
        final_avg = results[method_name]["avg"]
        final_dist = (
            estimator.distance_function(images, final_avg).detach().cpu().numpy()
        )
        final_dist = (final_dist - final_dist.mean()) / final_dist.std()
        final_dist = final_dist.reshape(-1, 1)

        gmm_initial = GaussianMixture(n_components=2, random_state=42).fit(initial_dist)
        gmm_final = GaussianMixture(n_components=2, random_state=42).fit(final_dist)

        # Plot side-by-side Initial vs Final
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(initial_dist, bins=40, alpha=0.5, color="gray", density=True)
        plot_gmm_fit(
            ax1, initial_dist, gmm_initial, f"Initial Distances: {method_name}"
        )

        ax2.hist(final_dist, bins=40, alpha=0.5, color="teal", density=True)
        plot_gmm_fit(ax2, final_dist, gmm_final, f"Final Distances: {method_name}")

        plt.tight_layout()
        plt.show()
