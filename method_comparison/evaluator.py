import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from typing import Iterable, Tuple, Dict, Any
from estimators.admm import ADMMSolver
from utils.space import Space
from utils.evaluation import (
    aggregate_weights,
    ALL_RECALL_METHODS,
    LABEL_MAP,
    compute_soft_metrics,
    compute_fsc,
    get_resolution_from_fsc
)

### Plotting utilities ###


def plot_distributions(
    scores_dict: dict,
    labels: np.ndarray,
    metric_name: str = "Final Weight Distribution",
    max_subplots: int = 4,
    density: bool = False,
):
    """Plots histograms of image scores separated by class."""
    if not scores_dict:
        return

    items = list(scores_dict.items())
    for i in range(0, len(items), max_subplots):
        chunk = items[i : i + max_subplots]
        n_items = len(chunk)
        fig, axes = plt.subplots(n_items, 1, figsize=(8.0, 3.0 * n_items), sharex=False)
        if n_items == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, chunk):
            ax.set_title(f"{metric_name}: {name}")
            min_val, max_val = values.min(), values.max()
            bins = (
                np.linspace(min_val - 0.01, max_val + 0.01, 40)
                if np.isclose(min_val, max_val)
                else np.linspace(min_val, max_val, 40)
            )

            if labels is None:
                ax.hist(values, bins=bins, alpha=0.7, color="teal", density=False)
                continue

            for label_idx, config in LABEL_MAP.items():
                mask = labels == label_idx
                if mask.any():
                    ax.hist(
                        values[mask],
                        bins=bins,
                        alpha=0.5,
                        label=config["name"],
                        color=config["color"],
                        density=density,
                    )
            ax.legend()

        plt.tight_layout()
        plt.show()


### Fourier Ring Correlation ###


def plot_fsc_curves(fsc_data: dict, threshold: float = 0.5):
    """Plots the FSC curves for all estimators."""
    plt.figure(figsize=(8, 5))

    for name, (freqs, fsc) in fsc_data.items():
        plt.plot(freqs, fsc, label=name)

    plt.axhline(threshold, color="r", linestyle="--", label=f"Threshold ({threshold})")
    plt.xlabel("Normalized Spatial Frequency")
    plt.ylabel("Fourier Shell Correlation")
    plt.title("Resolution Estimates (FSC/FRC)")
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


### ADMM comparison with IRLS ###


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


### Main evaluation pipeline ###


def compare_and_report(
    results: dict[str, Any],
    images_dict: dict[Space, torch.Tensor],
    ground_truth_img: np.ndarray,
    labels: np.ndarray,
    plot_weights: bool = False,
    max_subplots: int = 4,
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
    real_agg_strategies: Iterable[str] = ("mean", "energy"),
    fourier_agg_strategies: Iterable[str] = ("energy",),
    energy_reference: str = "ground_truth",  # "ground_truth" or "global_avg"
    fsc_threshold: float = 0.143,
    mask: np.ndarray = np.array([1]),
    reapply_mask: bool = False,
):
    print("\n" + "-" * 25 + "EVALUATION RESULTS" + "-" * 25 + "\n")

    # Identify good images
    idx_good = labels == 0

    # Initialize scores and metrics dicts
    all_scores_for_plotting = {}  # stores weights
    fsc_data_for_plotting = {}  # stores fsc curves for plots
    metrics_summary = {}  # stores error metrics

    # Setup the references for energy calculations
    gt_tensor = torch.from_numpy(ground_truth_img).to(
        dtype=torch.float32, device=images_dict[Space.REAL].device
    )
    if energy_reference == "ground_truth":
        ref_real = gt_tensor
    elif energy_reference == "global_avg":
        ref_real = images_dict[Space.REAL].mean(dim=0)
    else:
        raise ValueError("energy_reference must be 'ground_truth' or 'global_avg'")
    ref_fourier = torch.fft.rfft2(ref_real, norm="ortho")

    # Iterate over methods to calculate metrics and get weights for plots
    for method_name, data in results.items():
        metrics_summary[method_name] = {}

        # Get the estimate from this method
        estimated_img = data["avg"].detach().cpu().numpy()
        if reapply_mask:
            estimated_img *= mask

        # Calculate error metrics
        rmse = np.sqrt(mean_squared_error(ground_truth_img, estimated_img))
        corr, _ = pearsonr(ground_truth_img.flatten(), estimated_img.flatten())

        # FSC Calculation
        freqs, fsc_curve = compute_fsc(estimated_img, ground_truth_img)
        resolution = get_resolution_from_fsc(freqs, fsc_curve, threshold=fsc_threshold)
        fsc_data_for_plotting[method_name] = (freqs, fsc_curve)

        # Save metrics in dict
        metrics_summary[method_name]["RMSE"] = rmse
        metrics_summary[method_name]["Pearson_Corr"] = corr
        metrics_summary[method_name][f"FSC_Resolution_{fsc_threshold}"] = resolution

        # Print summary to terminal
        print(f"--- {method_name.upper()} ---")
        print(
            f"  Reconstruction RMSE: {rmse:.4f} | "
            f"Corr: {corr:.4f} | Res ({fsc_threshold}): {resolution:.4f}"
        )

        # Outlier rejection metrics (precision and recall)
        weights_dict = data["weights"]
        for space, weights in weights_dict.items():
            if weights is None:
                continue

            # Select reference and strategies based on space
            if space == Space.REAL:
                ref = ref_real
                strategies = real_agg_strategies
            elif space == Space.FOURIER_REAL:
                ref = ref_fourier.real
                strategies = fourier_agg_strategies
            elif space == Space.FOURIER_IMAG:
                ref = ref_fourier.imag
                strategies = fourier_agg_strategies
            else:
                continue

            for strategy in strategies:
                scores = aggregate_weights(weights, strategy=strategy, reference=ref)

                # Tag clearly for plots
                plot_key = f"{method_name} ({space.name} | {strategy})"
                all_scores_for_plotting[plot_key] = scores

                space_metrics = compute_soft_metrics(scores, idx_good, recall_methods)

                # Store in summary
                for metric_k, metric_v in space_metrics.items():
                    metrics_summary[method_name][
                        f"{space.name}_{strategy}_{metric_k}"
                    ] = metric_v

                print(f"  Space: {space.name} (Agg: {strategy})")
                print(f"    Avg Precision:   {space_metrics['ap']:.4f}")
                print(f"    Soft Precision:  {space_metrics['soft_precision']:.4f}")
                for recall_method in recall_methods:
                    metric = space_metrics.get(f"soft_recall_{recall_method}", None)
                    if metric is not None:
                        print(f"    Soft Recall:  {metric:.4f}\t({recall_method})")

        # Handle ADMM Baseline Extraction
        estimator = data["estimator"]
        is_admm = isinstance(estimator, ADMMSolver)
        if is_admm and plot_weights:
            print(f"  -> Extracting baseline IRLS weights for {method_name}...")
            baseline_weights = compute_baseline_irls(estimator, images_dict)

            for strategy in real_agg_strategies:
                baseline_scores = aggregate_weights(
                    baseline_weights, strategy=strategy, reference=ref_real
                )
                admm_scores = all_scores_for_plotting[
                    f"{method_name} ({Space.REAL.name} | {strategy})"
                ]

                plot_admm_vs_irls_scatter(
                    admm_scores=admm_scores,
                    irls_scores=baseline_scores,
                    labels=labels,
                    admm_name=f"{method_name} (Agg: {strategy})",
                )
        print("")

    # Weight Distribution Visualizations
    if plot_weights:
        # # Plot FSC Curves
        # plot_fsc_curves(fsc_data_for_plotting, threshold=fsc_threshold)

        plot_distributions(
            all_scores_for_plotting,
            labels,
            metric_name="Weight Distribution",
            max_subplots=max_subplots,
            density=False,
        )

    return metrics_summary


def report_unlabeled(results: dict, plot_weights: bool = True):
    """
    Evaluates results on unlabeled data by showing the overall weight distributions.
    """
    if plot_weights:
        all_scores = {}

        for method_name, data in results.items():
            print(f"Processed: {method_name}")

            # Get relevant weights
            if data["weights"][Space.REAL] is not None:
                weights = data["weights"][Space.REAL]
            else:
                weights = 0.5 * (
                    data["weights"][Space.FOURIER_REAL]
                    + data["weights"][Space.FOURIER_IMAG]
                )

            # Store weights for the next plot
            all_scores[method_name] = aggregate_weights(weights)

        # Plot overall weight distributions
        plot_distributions(all_scores, labels=None)
