from typing import Any, Iterable

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch

from method_comparison.results_classes import (
    MethodMetrics,
    MethodResults,
    EvaluationReport,
)
from utils.evaluation import (
    ALL_RECALL_METHODS,
    LABEL_MAP,
    compute_fsc,
    get_resolution_from_fsc,
    aggregate_weights,
    compute_soft_metrics,
)
from utils.space import Space


def _setup_energy_reference(
    ground_truth_img: np.ndarray | None,
    images_dict: dict[Space, torch.Tensor],
    energy_reference: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct the real- and Fourier-space reference tensors for energy aggregation.

    Parameters
    ----------
    ground_truth_img : np.ndarray or None
        Ground truth image array, shape `(H, W)`.
        `None` if ground truth is not available.
    images_dict : dict of {Space: torch.Tensor}
        Image tensors keyed by space; used to infer the target device.
    energy_reference : {"ground_truth", "global_avg"}
        Strategy for building the real-space reference.  `"ground_truth"`
        uses the known clean image; `"global_avg"` uses the mean of all
        input images.

    Returns
    -------
    ref_real : torch.Tensor
        Real-space reference image, shape `(H, W)`.
    ref_fourier : torch.Tensor
        Fourier transform of `ref_real` via `torch.fft.rfft2`, shape
        `(H, W//2 + 1)`.

    Raises
    ------
    ValueError
        If `energy_reference` is `'ground_truth'` but `ground_truth_img` is `None`.
        If `energy_reference` is not one of the accepted values.
    """
    if ground_truth_img is not None:
        gt_tensor = torch.from_numpy(ground_truth_img).to(
            dtype=torch.float32, device=images_dict[Space.REAL].device
        )
    elif energy_reference == "ground_truth":
        raise ValueError(
            "ground_truth energy reference requested, but ground_truth_img is None"
        )

    if energy_reference == "ground_truth":
        ref_real = gt_tensor
    elif energy_reference == "global_avg":
        ref_real = images_dict[Space.REAL].mean(dim=0)
    else:
        raise ValueError(
            "energy_reference must be one of 'ground_truth' or 'global_avg'"
        )
    ref_fourier = torch.fft.rfft2(ref_real, norm="ortho")

    return ref_real, ref_fourier


def _get_space_reference(
    space: Space,
    ref_real: torch.Tensor,
    ref_fourier: torch.Tensor,
) -> tuple[torch.Tensor, None] | None:
    """
    Return the reference tensor and aggregation strategies for a given space.

    Parameters
    ----------
    space : Space
        The weight space to look up.
    ref_real : torch.Tensor
        Real-space reference tensor.
    ref_fourier : torch.Tensor
        Complex Fourier-space reference tensor.

    Returns
    -------
    torch.Tensor or None
        The appropriate reference slice, or `None` if the space is not
        handled or no reference is available.
    """
    if space == Space.REAL:
        return ref_real  # may be None for unlabeled
    elif space == Space.FOURIER_REAL:
        return ref_fourier.real if ref_fourier is not None else None
    elif space == Space.FOURIER_IMAG:
        return ref_fourier.imag if ref_fourier is not None else None
    return None


def _compute_scores(
    weights_dict: dict[Space, torch.Tensor | None],
    real_agg_strategies: Iterable[str],
    fourier_agg_strategies: Iterable[str],
    ref_real: torch.Tensor | None = None,
    ref_fourier: torch.Tensor | None = None,
) -> dict[Space, dict[str, np.ndarray]]:
    """Aggregate per-image weights into scalar scores for all spaces and strategies.

    Parameters
    ----------
    weights_dict : dict of {Space: torch.Tensor or None}
        Raw weight tensors keyed by space, as returned by the estimator.
    real_agg_strategies : iterable of str
        Aggregation strategies to apply to real-space weights.
    fourier_agg_strategies : iterable of str
        Aggregation strategies to apply to Fourier-space weights.
    ref_real : torch.Tensor or None, optional
        Real-space reference tensor, required if `"energy"` is among
        `real_agg_strategies`. Default is `None`.
    ref_fourier : torch.Tensor or None, optional
        Complex Fourier reference tensor, required if `"energy"` is among
        `fourier_agg_strategies`. Default is `None`.

    Returns
    -------
    dict of {Space: dict of {str: np.ndarray}}
        Aggregated scores keyed by space then strategy. Spaces with
        `None` weights or no matching reference are omitted.
    """
    scores: dict[Space, dict[str, np.ndarray]] = {}

    for space, weights in weights_dict.items():
        if weights is None:
            continue
        ref = _get_space_reference(space, ref_real, ref_fourier)
        strategies = (
            real_agg_strategies if space == Space.REAL else fourier_agg_strategies
        )

        scores[space] = {
            strategy: aggregate_weights(weights, strategy=strategy, reference=ref)
            for strategy in strategies
            if not (
                ref is None and strategy == "energy"
            )  # energy strategy requested but no reference available; skip
        }

    return scores


def compute_report_labeled(
    results: dict[str, Any],
    images_dict: dict[Space, torch.Tensor],
    ground_truth_img: np.ndarray,
    labels: np.ndarray,
    reapply_mask: bool = False,
    mask: np.ndarray = np.array([1]),
    fsc_threshold: float = 0.143,
    recall_methods: Iterable[str] = ALL_RECALL_METHODS,
    real_agg_strategies: Iterable[str] = ("mean",),
    fourier_agg_strategies: Iterable[str] = ("mean", "energy"),
    energy_reference: str = "ground_truth",
) -> EvaluationReport:
    """
    Compute all quantitative metrics for a set of estimation results.

    Iterates over every method in `results`, computes reconstruction quality
    (RMSE, Pearson correlation, FSC resolution) and outlier-rejection quality
    (average precision, soft precision/recall) for each weight space and
    aggregation strategy, and packages everything into an `EvaluationReport`.

    No printing or plotting is performed here; use `print_report` and `plot_report`
    for those.

    Parameters
    ----------
    results : dict of {str: dict}
        Output of `run_estimators`.  Each value must contain the keys
        `"avg"` (estimated image tensor) and `"weights"` (`dict[Space, Tensor | None]`).
    images_dict : dict of {Space: torch.Tensor}
        Full image tensors keyed by space; used to build the global-average
        reference when `energy_reference="global_avg"`.
    ground_truth_img : np.ndarray
        Clean reference image, shape `(H, W)`.
    labels : np.ndarray
        Integer class label per image; `0` denotes a good image.
    reapply_mask : bool, optional
        If `True`, multiply every estimated image by `mask` before
        computing metrics.  Default is `False`.
    mask : np.ndarray, optional
        Binary mask to apply when `reapply_mask=True`.  Default is a scalar
        `[1]` (no-op).
    fsc_threshold : float, optional
        FSC value at which resolution is read off.  Default is `0.143`
        (gold-standard half-bit criterion).
    recall_methods : iterable of str, optional
        Soft-recall variants to compute; defaults to `ALL_RECALL_METHODS`.
    real_agg_strategies : iterable of str, optional
        Aggregation strategies to apply to real-space weights.  Default is
        `("mean",)`.
    fourier_agg_strategies : iterable of str, optional
        Aggregation strategies for Fourier-space weights.  Default is
        `("energy",)`.
    energy_reference : {"ground_truth", "global_avg"}, optional
        Reference image used for energy-based aggregation.  Default is
        `"ground_truth"`.

    Returns
    -------
    EvaluationReport
        Fully populated report object ready for printing or plotting.
    """
    # Get energy reference for weight aggregating
    ref_real, ref_fourier = _setup_energy_reference(
        ground_truth_img, images_dict, energy_reference
    )

    # Identify good images
    idx_good = labels == 0

    all_results = []
    # Iterate over methods to compute the metrics
    for method_name, data in results.items():
        # Get the estimated image for this method
        estimated_img = data["avg"].detach().cpu().numpy()
        if reapply_mask:
            estimated_img *= mask

        # Reconstruction quality metrics
        rmse = np.sqrt(mean_squared_error(ground_truth_img, estimated_img))
        corr, _ = pearsonr(ground_truth_img.flatten(), estimated_img.flatten())
        freqs, corr_values = compute_fsc(estimated_img, ground_truth_img)
        resolution = get_resolution_from_fsc(
            freqs, corr_values, threshold=fsc_threshold
        )

        # Iterate over spaces to calculate space weight metrics
        weights_dict = data["weights"]
        space_metrics: dict[Space, dict[str, dict]] = {}
        scores = _compute_scores(
            weights_dict,
            real_agg_strategies,
            fourier_agg_strategies,
            ref_real,
            ref_fourier,
        )
        for space, weights in weights_dict.items():
            if weights is None:
                continue
            space_metrics[space] = {}
            strategies = (
                real_agg_strategies if space == Space.REAL else fourier_agg_strategies
            )

            # Iterate over aggregating strategies and compute metrics
            for strategy in strategies:
                agg_weights = scores[space][strategy]
                space_metrics[space][strategy] = compute_soft_metrics(
                    agg_weights, idx_good, recall_methods
                )

        method_metrics = MethodMetrics(
            rmse=rmse,
            pearson_corr=corr,
            fsc_resolution=resolution,
            space_metrics=space_metrics,
        )
        all_results.append(
            MethodResults(
                name=method_name,
                metrics=method_metrics,
                scores=scores,
                fsc_data=(freqs, corr_values),
                estimated_img=estimated_img,
            )
        )

    return EvaluationReport(
        method_results=all_results, labels=labels, fsc_threshold=fsc_threshold
    )


def print_report(report: EvaluationReport) -> None:
    """
    Print a structured summary of an `EvaluationReport`.

    For each method, the output contains reconstruction metrics (RMSE, Pearson
    correlation, FSC resolution) followed by outlier-rejection metrics
    (average precision, soft precision, soft recall) broken down by weight
    space and aggregation strategy.

    Parameters
    ----------
    report : EvaluationReport
        Populated report produced by `compute_metrics`.

    Returns
    -------
    None
    """
    separator = "-" * 25
    print(f"\n{separator} EVALUATION RESULTS {separator}\n")

    for method_result in report.method_results:
        print(f"--- {method_result.name.upper()} ---")

        m = method_result.metrics
        if m is None:
            print("  No ground-truth metrics available.\n")
            continue

        print(
            f"  RMSE: {m.rmse:.4f} | "
            f"Pearson: {m.pearson_corr:.4f} | "
            f"FSC Resolution ({report.fsc_threshold}): {m.fsc_resolution:.4f}"
        )

        for space, strategy_metrics in m.space_metrics.items():
            for strategy, metrics in strategy_metrics.items():
                print(f"  Space: {space.name}  |  Aggregation: {strategy}")
                print(f"    Avg Precision:   {metrics['ap']:.4f}")
                print(f"    Soft Precision:  {metrics['soft_precision']:.4f}")
                for key, value in metrics.items():
                    if key.startswith("soft_recall_"):
                        recall_method = key[len("soft_recall_") :]
                        print(f"    Soft Recall ({recall_method}): {value:.4f}")

        print()


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


def compute_report_unlabeled(
    results: dict[str, Any],
    images_dict: dict[Space, torch.Tensor],
    real_agg_strategies: Iterable[str] = ("mean",),
    fourier_agg_strategies: Iterable[str] = ("energy",),
) -> EvaluationReport:
    """
    Compute weight-distribution scores for unlabeled data.

    No ground truth is available, so reconstruction metrics (RMSE, Pearson,
    FSC) and outlier-rejection metrics (precision/recall) are not computed.
    The report is populated with aggregated per-image scores only, which can
    be passed to `plot_report` to inspect weight distributions.

    Parameters
    ----------
    results : dict of {str: dict}
        Output of `run_estimators`.
    images_dict : dict of {Space: torch.Tensor}
        Full image tensors keyed by space; used to build the global-average
        reference when energy weight aggregation is used.
    real_agg_strategies : iterable of str, optional
        Aggregation strategies for real-space weights. Default is `("mean",)`.
    fourier_agg_strategies : iterable of str, optional
        Aggregation strategies for Fourier-space weights. Default is `("energy",)`.

    Returns
    -------
    EvaluationReport
        Report with `metrics=None` and `fsc_data=None` for every method,
        and `labels=None` at the report level.
    """
    ref_real, ref_fourier = _setup_energy_reference(None, images_dict, "global_avg")
    all_results = []
    for method_name, data in results.items():
        scores = _compute_scores(
            data["weights"],
            real_agg_strategies,
            fourier_agg_strategies,
            ref_real=ref_real,
            ref_fourier=ref_fourier,
        )

        all_results.append(
            MethodResults(
                name=method_name,
                metrics=None,
                scores=scores,
                fsc_data=None,
                estimated_img=data["avg"].detach().cpu().numpy(),
            )
        )

    return EvaluationReport(
        method_results=all_results,
        labels=None,
        fsc_threshold=None,  # unused, no ground truth
    )
