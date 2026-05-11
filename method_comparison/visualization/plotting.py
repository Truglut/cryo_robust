from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from method_comparison.domain.reports import EvaluationReport

# Helper for consistent coloring
LABEL_MAP = {
    0: {"name": "Inliers", "color": "blue"},
    1: {"name": "Rotated Outliers", "color": "orange"},
    2: {"name": "Misclassified", "color": "red"},
    3: {"name": "Noise", "color": "darkorange"},
}

AVERAGE_NAME = "Average"

BASE_PLOT_OPTIONS = {
    "max_subplots": 3,
    "density": False,
    "dpi": 150,
}


def _plot_weight_histogram(
    ax: plt.Axes,
    scores: np.ndarray,
    title: str,
    labels: np.ndarray | None,
    density: bool,
) -> None:
    """
    Render a single weight distribution histogram onto `ax`.

    Parameters
    ----------
    ax : plt.Axes
        The axes to draw on.
    scores : np.ndarray
        Score values to histogram.
    title : str
        Axes title.
    labels : np.ndarray | None
        Per-sample class labels. If None, the overall distribution is plotted.
    density : bool
        Whether to normalise to probability density.
    """
    ax.set_title(f"Weight Distribution: {title}")

    min_val, max_val = scores.min(), scores.max()
    bins = (
        np.linspace(min_val - 0.01, max_val + 0.01, 40)
        if np.isclose(min_val, max_val)
        else np.linspace(min_val, max_val, 40)
    )

    if labels is None:
        ax.hist(scores, bins=bins, alpha=0.7, color="teal", density=density)
        return

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


def _collect_weight_scores(
    report: EvaluationReport,
) -> dict[str, np.ndarray]:
    """
    Extract and flatten all (method, space, strategy) score arrays from a report.

    The average method is excluded.

    Parameters
    ----------
    report : EvaluationReport
        Populated evaluation report.

    Returns
    -------
    dict[str, np.ndarray]
        Ordered mapping from a human-readable plot key to score array.
    """
    all_scores: dict[str, np.ndarray] = {}

    for method_result in report.method_results:
        if method_result.name == AVERAGE_NAME:
            continue

        for space, strategy_scores in method_result.scores.items():
            for strategy, scores in strategy_scores.items():
                key = f"{method_result.name} ({space.name} | {strategy})"
                all_scores[key] = scores

    return all_scores


def _plot_weight_distributions(
    all_scores: dict[str, np.ndarray],
    labels: np.ndarray | None,
    max_subplots: int,
    density: bool,
) -> list[plt.Figure]:
    """
    Produce batched weight distribution figures.

    Parameters
    ----------
    all_scores : dict[str, np.ndarray]
        Mapping from plot key to score array, as returned by
        `_collect_weight_scores`.
    labels : np.ndarray | None
        Per-sample class labels.
    max_subplots : int
        Maximum subplots per figure.
    density : bool
        Whether to normalise histograms to probability density.

    Returns
    -------
    list[plt.Figure]
        One figure per batch.
    """
    figures = []
    items = list(all_scores.items())

    for batch_start in range(0, len(items), max_subplots):
        chunk = items[batch_start : batch_start + max_subplots]
        n = len(chunk)

        fig, axes = plt.subplots(n, 1, figsize=(8.0, 3.0 * n), sharex=False)
        if n == 1:
            axes = [axes]

        for ax, (plot_key, scores) in zip(axes, chunk):
            _plot_weight_histogram(ax, scores, plot_key, labels, density)

        fig.tight_layout()
        figures.append(fig)

    return figures


def _plot_fsc_curves(report: EvaluationReport) -> plt.Figure | None:
    """Produce the FSC curve comparison figure.

    Parameters
    ----------
    report : EvaluationReport
        Populated evaluation report.

    Returns
    -------
    plt.Figure | None
        The figure, or None if no FSC data is available.
    """
    fsc_items = [
        (mr.name, mr.fsc_data)
        for mr in report.method_results
        if mr.fsc_data is not None
    ]
    if not fsc_items:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, (freqs, fsc_curve) in fsc_items:
        ax.plot(freqs, fsc_curve, label=name)

    ax.axhline(
        report.fsc_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold ({report.fsc_threshold})",
    )

    ax.set_xlabel("Normalised Spatial Frequency")
    ax.set_ylabel("Fourier Shell Correlation")
    ax.set_title("Resolution Estimates (FSC/FRC)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_report(
    report: EvaluationReport,
    max_subplots: int,
    plot_weights: bool = True,
    density: bool = False,
    plot_fsc: bool = True,
) -> None:
    """Produce all diagnostic plots for an `EvaluationReport`.

    Parameters
    ----------
    report : EvaluationReport
        Populated report produced by `compute_metrics`.
    max_subplots : int, optional
        Maximum number of histogram subplots per figure. Default is 4.
    plot_weights : bool, optional
        Whether to plot weight distribution histograms. Default is True.
    density : bool, optional
        If True, normalise histograms to probability density. Default is False.
    plot_fsc : bool, optional
        Whether to render the FSC curve comparison plot. Default is True.
    """
    if plot_weights:
        all_scores = _collect_weight_scores(report)
        _ = _plot_weight_distributions(all_scores, report.labels, max_subplots, density)
        plt.show()

    if plot_fsc:
        fig = _plot_fsc_curves(report)
        if fig is not None:
            fig.show()
            plt.show()


def save_report_figures(
    report: EvaluationReport,
    output_path: Path,
    max_subplots: int,
    density: bool = False,
    dpi: int = 150,
) -> dict[str, list[Path]]:
    """Save all report figures to disk and return their paths.

    Parameters
    ----------
    report : EvaluationReport
        Populated evaluation report.
    output_path : Path
        Directory in which figures are saved. Created if absent.
    max_subplots : int
        Maximum subplots per weight-distribution figure.
    density : bool, optional
        Whether to normalise histograms to probability density.
    dpi : int, optional
        Output resolution in dots per inch. Default is 150.

    Returns
    -------
    dict[str, list[Path]]
        Keys are ``"weight_distributions"`` and ``"fsc_curves"``.
        Values are lists of saved file paths (FSC list has 0 or 1 entries).
    """
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[Path]] = {"weight_distributions": [], "fsc_curves": []}

    all_scores = _collect_weight_scores(report)
    for i, fig in enumerate(
        _plot_weight_distributions(all_scores, report.labels, max_subplots, density)
    ):
        path = output_path / f"weight_distribution_{i}.pdf"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["weight_distributions"].append(path)

    fsc_fig = _plot_fsc_curves(report)
    if fsc_fig is not None:
        path = output_path / "fsc_curves.pdf"
        fsc_fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fsc_fig)
        saved["fsc_curves"].append(path)

    return saved
