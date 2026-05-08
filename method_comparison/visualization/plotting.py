import numpy as np
import torch
import matplotlib.pyplot as plt

from estimators.admm import ADMMSolver
from method_comparison.domain.reports import EvaluationReport
from method_comparison.domain.enums import Space

# Helper for consistent coloring
LABEL_MAP = {
    0: {"name": "Inliers", "color": "blue"},
    1: {"name": "Rotated Outliers", "color": "orange"},
    2: {"name": "Misclassified", "color": "red"},
}

def plot_report(
    report: EvaluationReport,
    max_subplots: int = 4,
    plot_weights: bool = True,
    density: bool = False,
    plot_fsc: bool = True,
) -> None:
    """Produce all diagnostic plots for an `EvaluationReport`.

    Generates two groups of plots:

    * **Weight distributions** — one histogram per (method, space, strategy)
      combination, with bars separated by class label.  Figures contain at
      most `max_subplots` subplots each so they remain readable.
    * **FSC curves** — all methods overlaid on a single axes, with a
      horizontal line at the experiment threshold.

    Parameters
    ----------
    report : EvaluationReport
        Populated report produced by `compute_metrics`.
    max_subplots : int, optional
        Maximum number of histogram subplots per figure.  Default is `4`.
    plot_weights: bool, optional.
        Whether to plot the weight distribution histograms.  Default is `True`.
    density : bool, optional
        If `True`, normalise histograms to probability density.  Default is `False`.
    plot_fsc : bool, optional
        Whether to render the FSC curve comparison plot.  Default is `True`.

    Returns
    -------
    None
    """
    labels = report.labels

    # Weight distribution histograms
    if plot_weights:
        # Flatten all (label, scores_array) pairs into an ordered dict for batching
        all_scores: dict[str, np.ndarray] = {}
        for method_result in report.method_results:
            for space, strategy_scores in method_result.scores.items():
                for strategy, scores in strategy_scores.items():
                    # Generate an informative plot key
                    key = f"{method_result.name} ({space.name} | {strategy})"
                    all_scores[key] = scores

        # Iterate over the scores dict to plot weight distributions
        items = list(all_scores.items())
        for batch_start in range(0, len(items), max_subplots):
            # Operate in batches to limit number of subplots
            chunk = items[batch_start : batch_start + max_subplots]
            n = len(chunk)
            fig, axes = plt.subplots(n, 1, figsize=(8.0, 3.0 * n), sharex=False)
            if n == 1:
                axes = [axes]

            # Plot weight distributions for every method in the batch
            for ax, (plot_key, scores) in zip(axes, chunk):
                ax.set_title(f"Weight Distribution: {plot_key}")
                min_val, max_val = scores.min(), scores.max()
                bins = (
                    np.linspace(min_val - 0.01, max_val + 0.01, 40)
                    if np.isclose(min_val, max_val)
                    else np.linspace(min_val, max_val, 40)
                )

                if labels is None:
                    # Plot overall distribution
                    ax.hist(scores, bins=bins, alpha=0.7, color="teal", density=density)
                else:
                    # Plot per-class distributions
                    for label_idx, config in LABEL_MAP.items():
                        mask = labels == label_idx
                        if mask.any():
                            ax.hist(
                                scores[mask],
                                bins=bins,
                                alpha=0.5,
                                label=config["name"],
                                color=config["color"],
                                density=density,
                            )
                    ax.legend()

            plt.tight_layout()
            plt.show()

    # FSC curves
    if not plot_fsc:
        return

    # Get fsc curve data from each method
    fsc_items = [
        (mr.name, mr.fsc_data)
        for mr in report.method_results
        if mr.fsc_data is not None
    ]
    if not fsc_items:
        return

    # Create figure and plot fsc curves for each method
    plt.figure(figsize=(8, 5))
    for name, (freqs, fsc_curve) in fsc_items:
        plt.plot(freqs, fsc_curve, label=name)

    # Plot the threshold as a horizontal line
    plt.axhline(
        report.fsc_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold ({report.fsc_threshold})",
    )

    # Labels, legends and titles
    plt.xlabel("Normalised Spatial Frequency")
    plt.ylabel("Fourier Shell Correlation")
    plt.title("Resolution Estimates (FSC/FRC)")
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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