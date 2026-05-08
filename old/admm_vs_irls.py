import numpy as np
import torch
import matplotlib.pyplot as plt

from estimators.admm import ADMMSolver

from method_comparison.domain.enums import Space
from method_comparison.visualization.plotting import LABEL_MAP

def compute_baseline_irls(
    admm_estimator: ADMMSolver, images_dict: dict
) -> torch.Tensor:
    """Clones the internal IRLS solver of an ADMM estimator and fits it without priors."""
    # Assuming standard IRLSSolver structure. We avoid deepcopying PyTorch modules/tensors directly.
    template = admm_estimator.irls_real

    # Create an identical, fresh instance
    baseline_solver = template.__class__(
        weight_function=template.weight_function,
        max_iter=template.max_iter,
        tol=template.tol,
        damping_coef=template.damping_coef,
        min_weight=template.min_weight,
        max_weight=template.max_weight,
        space=template.space,
        device=template.device,
        eps=template.eps,
    )

    # Fit strictly without the prior to establish the true baseline
    _, weights = baseline_solver.fit(
        images=images_dict[Space.REAL], prior_mean=None, prior_variance=None
    )
    return weights


def plot_admm_vs_irls_scatter(
    admm_scores: np.ndarray, irls_scores: np.ndarray, labels: np.ndarray, admm_name: str
):
    """Visualizes how the Fourier prior in ADMM changes the real-space weights compared to pure IRLS."""
    plt.figure(figsize=(7, 7))

    for label_idx, config in LABEL_MAP.items():
        mask = labels == label_idx
        if mask.any():
            plt.scatter(
                irls_scores[mask],
                admm_scores[mask],
                alpha=0.6,
                label=config["name"],
                color=config["color"],
                edgecolors="none",
            )

    # Identity line
    max_val = max(irls_scores.max(), admm_scores.max())
    min_val = min(irls_scores.min(), admm_scores.min())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="Identity (No change)",
    )

    plt.xlabel("Pure Real-Space IRLS Score")
    plt.ylabel("ADMM Real-Space Score")
    plt.title(admm_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.show()