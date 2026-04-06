import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, average_precision_score
from estimators.base import Space
from typing import Iterable, Tuple, Dict


# Helper for consistent coloring
LABEL_MAP = {
    0: {"name": "Inliers", "color": "blue"},
    1: {"name": "Rotated Outliers", "color": "orange"},
    2: {"name": "Misclassified", "color": "red"},
}

# List of all implemented recall methods
ALL_RECALL_METHODS = ["huang_tagare", "inlier_avg", "global_avg"]


def aggregate_weights(
    weights: Dict[Space, torch.Tensor] | torch.Tensor | None,
) -> np.ndarray:
    if weights is None:
        return None
    if isinstance(weights, torch.Tensor):
        # Move to CPU and convert to numpy
        w = weights.detach().cpu().numpy()

        # Check if weights are per-pixel (e.g., shape is N x H x W)
        if w.ndim == 3 and (w.shape[1] > 1 or w.shape[2] > 1):
            # Average across height and width
            return w.mean(axis=(1, 2))
        else:
            # Already per-image (e.g., shape is N x 1 x 1 or N)
            return w.flatten()

    # TODO: improve the following
    # If there are real-space weights, use those
    if weights[Space.REAL] is not None:
        return aggregate_weights(weights[Space.REAL])

    # Otherwise, take the average of fourier real part and imaginary part
    agg_fourier_real = aggregate_weights(weights[Space.FOURIER_REAL])
    agg_fourier_imag = aggregate_weights(weights[Space.FOURIER_IMAG])

    return (agg_fourier_real + agg_fourier_imag) / 2


def get_precision(weights: np.ndarray, idx_good: np.ndarray | torch.Tensor) -> float:
    """
    Calculates the precision metric \\hat{P} proposed in Huang and Tagare, 2016.
    """
    return float(weights[idx_good].sum(axis=0) / weights.sum(axis=0))


def get_recall(
    weights: torch.Tensor,
    idx_good: np.ndarray | torch.Tensor,
    average_type: str = "huang_tagare",
) -> float:
    """
    Calculates the recall metric \\hat{R} proposed in Huang and Tagare, 2016.
    weights should have shape (N, ).
    """
    n_in = idx_good.sum()

    if average_type == "inlier_avg":
        omega_bar = weights[idx_good].mean()
    elif average_type == "global_avg":
        omega_bar = weights.mean()
    elif average_type == "huang_tagare":
        omega_bar = weights.sum() / n_in
    else:
        warnings.warn(
            "Unrecognised average type in get_recall(): using 'huang_tagare' method"
        )
        omega_bar = weights.sum() / n_in

    return float(np.clip(weights[idx_good] / omega_bar, max=1).mean())


def calculate_soft_metrics(
    scores: np.ndarray, idx_good: np.ndarray, recall_methods: Iterable[str]
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calculates average precision together with soft precision and soft recall.
    Scores should ideally be bounded between [0, 1].
    """
    # Average Precision (Area under the PR curve - standard ML metric)
    # This evaluates how well the weights rank inliers above outliers.
    ap = average_precision_score(idx_good, scores)

    soft_precision = get_precision(scores, idx_good)

    soft_recall = {
        method: get_recall(scores, idx_good, method) for method in recall_methods
    }

    return ap, soft_precision, soft_recall


def plot_distributions(
    data_dict: dict,
    labels: np.ndarray | None = None,
    metric_name: str = "Final weight distribution",
    max_subplots: int = 4,
    subplot_height: float = 3.0,
    figure_width: float = 8.0,
):
    """Plots histograms separated by the 3 classes, creating new figures if necessary."""
    if not data_dict:
        return

    # Convert dict items to a list so we can slice it into chunks
    items = list(data_dict.items())

    # Iterate through the items in chunks of 'max_subplots'
    for i in range(0, len(items), max_subplots):
        chunk = items[i : i + max_subplots]
        n_items = len(chunk)

        # Create a new figure for each chunk
        fig, axes = plt.subplots(
            n_items, 1, figsize=(figure_width, subplot_height * n_items), sharex=False
        )

        # Ensure axes is always iterable, even if there's only 1 item in the chunk
        if n_items == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, chunk):
            # Calculate safe bins
            min_val, max_val = values.min(), values.max()
            bins = (
                np.linspace(min_val - 0.01, max_val + 0.01, 40)
                if np.isclose(min_val, max_val)
                else np.linspace(min_val, max_val, 40)
            )

            if labels is None:
                ax.hist(values, bins=bins, alpha=0.8, density=True)
            else:
                for label_idx, config in LABEL_MAP.items():
                    mask = labels == label_idx
                    if mask.any():
                        ax.hist(
                            values[mask],
                            bins=bins,
                            alpha=0.5,
                            label=config["name"],
                            color=config["color"],
                            density=True,
                        )
                    ax.legend()
            ax.set_title(f"{metric_name}: {name}")

        plt.tight_layout()
        plt.show()  # Display the current figure of up to 4 subplots before making the next one


def compare_and_report(
    results: dict,
    ground_truth_img: np.ndarray,
    labels: np.ndarray,
    plot_weights: bool,
    max_subplots: int = 4,
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
):
    print("\n" + "=" * 50 + "\nEVALUATION RESULTS\n" + "=" * 50)

    # Recreate idx_good for the soft metrics (Inliers = 0)
    idx_good = labels == 0
    all_scores = {}

    for method_name, data in results.items():
        # 1. Image Quality Metrics
        estimated_img = data["avg"].detach().cpu().numpy()

        # Root Mean Squared Error
        mse = mean_squared_error(ground_truth_img, estimated_img)
        rmse = np.sqrt(mse)

        # Pearson Correlation (requires flattening the 2D images)
        corr, _ = pearsonr(ground_truth_img.flatten(), estimated_img.flatten())

        # 2. Outlier Rejection Metrics
        weights = data["weights"]
        scores = aggregate_weights(weights)
        all_scores[method_name] = scores

        ap, soft_prec, soft_rec = calculate_soft_metrics(
            scores, idx_good, recall_methods
        )

        # Print Report
        print(f"--- {method_name.upper()} ---")
        print(f"  RMSE:           {rmse:.4f}")
        print(f"  Correlation:    {corr:.4f}")
        print(f"  Avg Precision:  {ap:.4f}")
        print(f"  Soft Precision: {soft_prec:.4f}")
        print(f"  Soft Recall:")
        max_len = max(len(method) for method in soft_rec)
        for recall_method, value in soft_rec.items():
            print(f"\t- {recall_method:<{max_len}}: {value:.4f}")
        print("")

    if plot_weights:
        # Plot Weight Distributions (3-class)
        plot_distributions(
            all_scores,
            labels,
            "Final weight distribution",
            max_subplots=max_subplots,
            subplot_height=2.5,
        )


def report_unlabeled(results: dict):
    """
    Evaluates results on unlabeled data by showing the overall weight distributions.
    """
    all_scores = {}

    for method_name, data in results.items():
        print(f"Processed: {method_name}")
        weights = data["weights"][Space.REAL]

        # Store weights for the next plot
        all_scores[method_name] = aggregate_weights(weights)

    # Plot overall weight distributions
    plot_distributions(all_scores, labels=None)
