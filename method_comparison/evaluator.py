import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, average_precision_score
from typing import Iterable, Tuple, Dict, Any
from estimators.admm import ADMMSolver
from utils.space import Space
from utils.evaluation import (
    aggregate_weights,
    get_precision,
    get_recall,
    ALL_RECALL_METHODS,
    LABEL_MAP,
    compute_soft_metrics,
)


# Helper for consistent coloring
LABEL_MAP = {
    0: {"name": "Inliers", "color": "blue"},
    1: {"name": "Rotated Outliers", "color": "orange"},
    2: {"name": "Misclassified", "color": "red"},
}


### Plotting utilities ###


def plot_distributions(
    scores_dict: dict,
    labels: np.ndarray,
    metric_name: str = "Final Weight Distribution",
    max_subplots: int = 4,
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
            min_val, max_val = values.min(), values.max()
            bins = (
                np.linspace(min_val - 0.01, max_val + 0.01, 40)
                if np.isclose(min_val, max_val)
                else np.linspace(min_val, max_val, 40)
            )

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
        plt.show()


### ADMM comparison with IRLS ###

def compute_baseline_irls(admm_estimator: ADMMSolver, images_dict: dict) -> torch.Tensor:
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
        eps=template.eps
    )
    
    # Fit strictly without the prior to establish the true baseline
    _, weights = baseline_solver.fit(
        images=images_dict[Space.REAL], 
        prior_mean=None, 
        prior_variance=None
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
    estimators: Dict[str, Any],
    images_dict: Dict[Space, torch.Tensor],
    ground_truth_img: np.ndarray,
    labels: np.ndarray,
    plot_weights: bool = False,
    max_subplots: int = 4,
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
    real_agg_strategies: Iterable[str] = ("mean", "energy"),
    fourier_agg_strategies: Iterable[str] = ("energy",),
    energy_reference: str = "ground_truth"  # "ground_truth" or "global_avg"
) -> Dict[str, Any]:
    
    print("\n" + "=" * 60 + "\nEVALUATION RESULTS\n" + "=" * 60)

    idx_good = (labels == 0)
    all_scores_for_plotting = {}
    metrics_summary = {}

    # Setup the unbiased references for energy calculations
    gt_tensor = torch.from_numpy(ground_truth_img).to(dtype=torch.float32, device=images_dict[Space.REAL].device)
    
    if energy_reference == "ground_truth":
        ref_real = gt_tensor
    elif energy_reference == "global_avg":
        ref_real = images_dict[Space.REAL].mean(dim=0)
    else:
        raise ValueError("energy_reference must be 'ground_truth' or 'global_avg'")
        
    ref_fourier = torch.fft.rfft2(ref_real, norm="ortho")

    for method_name, estimator in estimators.items():
        metrics_summary[method_name] = {}
        
        # 1. Image Quality Metrics
        estimated_img = estimator.avg.detach().cpu().numpy()
        rmse = np.sqrt(mean_squared_error(ground_truth_img, estimated_img))
        corr, _ = pearsonr(ground_truth_img.flatten(), estimated_img.flatten())
        
        metrics_summary[method_name]["RMSE"] = rmse
        metrics_summary[method_name]["Pearson_Corr"] = corr
        
        print(f"--- {method_name.upper()} ---")
        print(f"  Reconstruction RMSE: {rmse:.4f} | Corr: {corr:.4f}")

        # 2. Outlier Rejection Metrics (Evaluating EVERY space)
        for space, w in estimator.final_weights.items():
            if w is None: continue
            
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
                scores = aggregate_weights(w, strategy=strategy, reference=ref)
                
                # Tag clearly for plots
                plot_key = f"{method_name} ({space.name} | {strategy})"
                all_scores_for_plotting[plot_key] = scores
                
                space_metrics = compute_soft_metrics(scores, idx_good, recall_methods)
                
                # Store in summary
                for metric_k, metric_v in space_metrics.items():
                    metrics_summary[method_name][f"{space.name}_{strategy}_{metric_k}"] = metric_v

                print(f"  Space: {space.name} (Agg: {strategy})")
                print(f"    Avg Precision:   {space_metrics['ap']:.4f}")
                print(f"    Soft Precision:  {space_metrics['soft_precision']:.4f}")
                for recall_method in ALL_RECALL_METHODS:
                    metric = space_metrics.get(f'soft_recall_{recall_method}', None)
                    if metric is not None:
                        print(f"    Soft Recall:  {metric:.4f}\t({recall_method})")

        # 3. Handle ADMM Baseline Extraction
        is_admm = isinstance(estimator, ADMMSolver)
        if is_admm and plot_weights:
            print(f"  -> Extracting baseline IRLS weights for {method_name}...")
            baseline_weights = compute_baseline_irls(estimator, images_dict)
            
            for strategy in real_agg_strategies:
                baseline_scores = aggregate_weights(baseline_weights, strategy=strategy, reference=ref_real)
                admm_scores = all_scores_for_plotting[f"{method_name} ({Space.REAL.name} | {strategy})"]
                
                plot_admm_vs_irls_scatter(
                    admm_scores=admm_scores,
                    irls_scores=baseline_scores,
                    labels=labels,
                    admm_name=f"{method_name} (Agg: {strategy})"
                )
        print("")

    # 4. Standard Distribution Visualizations
    if plot_weights:
        plot_distributions(
            all_scores_for_plotting, 
            labels, 
            metric_name="Score Distribution", 
            max_subplots=max_subplots
        )

    return metrics_summary


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
