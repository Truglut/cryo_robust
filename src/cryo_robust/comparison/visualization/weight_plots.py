"""Weight distribution plotting and visualization utilities"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cryo_robust.comparison.domain.enums import AggregationStrategy
from cryo_robust.comparison.domain.reports import EvaluationReport
from cryo_robust.comparison.domain.runs import AVERAGE_NAME
from cryo_robust.comparison.visualization.plot_utils import (
    LABEL_MAP,
    HISTOGRAM_TYPE,
)


import numpy as np
from matplotlib.axes import Axes

from cryo_robust.domain import ImageSpace

WEIGHT_PLOT_TITLE_OPTIONS = {
    "show_description": True,
    "show_method": True,
    "show_space": True,
    "show_aggregation": False,
}


def _plot_weight_histogram(
    ax: Axes,
    scores: np.ndarray,
    title: str,
    labels: np.ndarray | None,
    density: bool,
) -> None:
    """
    Render a single weight distribution histogram onto `ax`.

    Parameters
    ----------
    ax : Axes
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
    ax.set_title(title)

    min_val, max_val = scores.min(), scores.max()
    bins = (
        np.linspace(min_val - 0.01, max_val + 0.01, 40)
        if np.isclose(min_val, max_val)
        else np.linspace(min_val, max_val, 40)
    )

    if labels is None:
        ax.hist(
            scores,
            bins=bins,
            alpha=0.7,
            color="teal",
            density=density,
            histtype=HISTOGRAM_TYPE,
        )
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
                edgecolor=config["color"],
                linewidth=1.2,
                density=density,
                histtype=HISTOGRAM_TYPE,
            )
    ax.legend()


def collect_weight_scores(
    report: EvaluationReport,
) -> dict[tuple[str, ImageSpace, AggregationStrategy], np.ndarray]:
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
    all_scores: dict[tuple[str, ImageSpace, AggregationStrategy], np.ndarray] = {}

    for method_result in report.method_results:
        if method_result.name == AVERAGE_NAME:
            continue

        for space, strategy_scores in method_result.scores.items():
            for strategy, scores in strategy_scores.items():
                key = (method_result.name, space, strategy)
                all_scores[key] = scores

    return all_scores


def _build_weight_plot_title(
    method: str,
    space: ImageSpace,
    strategy,
    suffix: str | None = None,
) -> str:
    """Build a weight-distribution plot title from the configured components."""
    parts = []

    if WEIGHT_PLOT_TITLE_OPTIONS["show_description"]:
        parts.append("Weight distribution")

    if WEIGHT_PLOT_TITLE_OPTIONS["show_method"]:
        parts.append(method)

    if WEIGHT_PLOT_TITLE_OPTIONS["show_space"]:
        parts.append(space.name)

    if WEIGHT_PLOT_TITLE_OPTIONS["show_aggregation"]:
        parts.append(str(strategy))

    title = " — ".join(parts)

    if suffix:
        title = f"{title} — {suffix}" if title else suffix

    return title


def plot_weight_distributions(
    all_scores: dict[tuple[str, ImageSpace, AggregationStrategy], np.ndarray],
    labels: np.ndarray | None,
    max_subplots: int,
    density: bool,
    title_suffix: str | None = None,
    fig_width: float = 4.0,
    fig_height: float = 3.0,
) -> list[Figure]:
    """
    Produce batched weight distribution figures.

    Parameters
    ----------
    all_scores : dict[tuple[str, ImageSpace, AggregationStrategy], np.ndarray]
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
    list[Figure]
        One figure per batch.
    """
    figures = []
    items = list(all_scores.items())

    for batch_start in range(0, len(items), max_subplots):
        chunk = items[batch_start : batch_start + max_subplots]
        n = len(chunk)

        fig, axes = plt.subplots(
            n, 1, figsize=(fig_width, fig_height * n), sharex=False
        )
        if n == 1:
            axes = [axes]

        for ax, ((method, space, strategy), scores) in zip(axes, chunk):
            title = _build_weight_plot_title(
                method=method, space=space, strategy=strategy, suffix=title_suffix
            )
            _plot_weight_histogram(ax, scores, title, labels, density)

        fig.tight_layout()
        figures.append(fig)

    return figures
