from typing import Sequence, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scipy.stats as stats

from .plot_utils import GOOD_BAD_PLOT_OPTIONS, HISTOGRAM_TYPE

from cryo_robust.estimators.results import GMMDiagnostics
from cryo_robust.comparison.domain.reports import EvaluationReport

PLOT_COMPONENT_PDFS = True
PLOT_FULL_MODEL_PDF = False
PLOT_INITIAL_REFERENCE = True
GMM_DISTANCE_PLOT_TYPE_DEFAULT = "overlayed-proportional"


def _plot_gmm_fit(
    diagnostics: GMMDiagnostics,
    idx_good: np.ndarray | None,
    idx_bad: np.ndarray | None,
    plot_initial_reference: bool,
    distance_plot_type: Literal["stacked", "overlayed-full", "overlayed-proportional"],
    plot_component_pdfs: bool,
    plot_full_model_pdf: bool,
    title: str | None = None,
) -> Figure:
    """
    Produce a plot of the GMM state given in the diagnostics object. The plot shows
    the distribution of distances from each image to the reference with the GMM
    density overlayed.

    Parameters
    ----------
    diagnostics : GMMDiagnostics
        Diagnostics object containing information about the GMM fit. It should have
        a ``distances`` property with the distances from the images to the initial
        reference and ``means``, ``vars`` and ``component_weights`` properties
        detailing the state of the GMM model after the fit.
    idx_good : np.ndarray, optional
        Indices of the inlier images, by default None.
    idx_bad : np.ndarray, optional
        Indices of the outlier images, by default None.
    title : str, optional
        Plot title, by default None.
    plot_initial_reference: bool, optional
        If True, make two subplots: one shows the initial reference, the other shows
        the distance distribution. Default is False.

    Returns
    -------
    Figure
        Figure object containing the plot.
    """
    ncols = 2 if plot_initial_reference else 1

    # figsize default is (6.4, 4.8), and it is (width, height)
    figsize = (ncols * 5.6, 4.0)

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, squeeze=False)

    col = 0
    if plot_initial_reference:
        initial_reference = diagnostics.initial_reference.detach().cpu().numpy()

        ax = axes[0, col]
        ax.imshow(initial_reference, cmap="gray")
        ax.set_title("Initial reference")
        ax.set_axis_off()

        col += 1

    _plot_gmm_distances_fit(
        ax=axes[0, col],
        distances_np=diagnostics.distances.detach().cpu().numpy(),
        standardized_distances=diagnostics.standardized_distances,
        model_means=diagnostics.means,
        model_vars=diagnostics.vars,
        model_component_weights=diagnostics.component_weights,
        idx_good=idx_good,
        idx_bad=idx_bad,
        title="Distances and GMM components",
        distance_plot_type=distance_plot_type,
        plot_component_pdfs=plot_component_pdfs,
        plot_full_model_pdf=plot_full_model_pdf,
    )

    fig.suptitle(title)

    return fig


def _plot_gmm_distances_fit(
    ax: Axes,
    distances_np: np.ndarray,
    *,
    standardized_distances: bool,
    model_means: Sequence[float],
    model_vars: Sequence[float],
    model_component_weights: Sequence[float],
    idx_good: np.ndarray | None,
    idx_bad: np.ndarray | None,
    title: str | None,
    bins: int = 30,
    distance_plot_type: Literal["stacked", "overlayed_full", "overlayed_proportional"],
    plot_full_model_pdf: bool,
    plot_component_pdfs: bool,
):
    dist_min = distances_np.min()
    dist_max = distances_np.max()
    length = dist_max - dist_min

    x = np.linspace(dist_min - 0.1 * length, dist_max + 0.1 * length, 1000)
    if standardized_distances:
        multiplier = 1.0 / distances_np.std()
        x_for_model = (x - distances_np.mean()) * multiplier
    else:
        multiplier = 1.0
        x_for_model = x

    if idx_good is None or idx_bad is None:
        ax.hist(distances_np, bins=bins, density=True, alpha=0.7)
    else:
        plot_distances_function = GMM_PLOT_FUNCTIONS[distance_plot_type]

        plot_distances_function(
            ax,
            distances_np,
            dist_min=dist_min,
            dist_max=dist_max,
            idx_good=idx_good,
            idx_bad=idx_bad,
            bins=bins,
        )

    if plot_full_model_pdf or plot_component_pdfs:
        means = np.asarray(model_means)[:, None]
        stds = np.sqrt(model_vars)[:, None]
        weights = np.asarray(model_component_weights)[:, None]

        # Calculate densities for all components simultaneously (shape: [num_components, len(x_for_model)])
        component_matrix = (
            multiplier * weights * stats.norm.pdf(x_for_model, loc=means, scale=stds)
        )

        if plot_component_pdfs:
            for i, (pdf, w) in enumerate(
                zip(component_matrix, model_component_weights), start=1
            ):
                ax.plot(
                    x,
                    pdf,
                    linestyle="--",
                    linewidth=2,
                    label=f"Gaussian {i} (w={w:.2f})",
                )

        if plot_full_model_pdf:
            full_pdf = component_matrix.sum(axis=0)
            ax.plot(x, full_pdf, linestyle="--", linewidth=2, label="Full GMM density")

    if title is not None:
        ax.set_title(title)

    ax.legend()


def plot_report_gmm_fits(
    report: EvaluationReport,
    labels: np.ndarray | None = None,
    plot_initial_reference: bool = False,
    gmm_distance_plot_type: Literal[
        "stacked", "overlayed-full", "overlayed-proportional"
    ] = GMM_DISTANCE_PLOT_TYPE_DEFAULT,
) -> list[Figure]:
    method_evaluations = report.method_results

    gmm_figures: list[Figure] = []

    if labels is not None:
        idx_good = labels == 0
        idx_bad = ~idx_good
    else:
        idx_good = None
        idx_bad = None

    for evaluation in method_evaluations:
        name = evaluation.name
        diagnostics = evaluation.result.diagnostics

        # Skip non-gmm estimators: they won't have gmm diagnostics
        if not isinstance(diagnostics, GMMDiagnostics):
            continue

        gmm_fig = _plot_gmm_fit(
            diagnostics,
            idx_good=idx_good,
            idx_bad=idx_bad,
            title=name,
            plot_initial_reference=plot_initial_reference,
            distance_plot_type=gmm_distance_plot_type,
            plot_component_pdfs=PLOT_COMPONENT_PDFS,
            plot_full_model_pdf=PLOT_FULL_MODEL_PDF,
        )

        gmm_figures.append(gmm_fig)

    return gmm_figures


### ===========================
### Types of GMM distances plot
### ===========================


def _plot_gmm_distances_stacked(
    ax: Axes,
    distances_np: np.ndarray,
    *,
    dist_min: float,
    dist_max: float,
    idx_good: np.ndarray,
    idx_bad: np.ndarray,
    bins: int,
) -> None:
    # Define explicit bin boundaries so both classes share identical edges
    bin_edges = np.linspace(dist_min, dist_max, bins + 1)

    ax.hist(
        [distances_np[idx_good], distances_np[idx_bad]],
        bins=bin_edges,
        density=True,
        stacked=True,
        color=[
            GOOD_BAD_PLOT_OPTIONS["good"]["color"],
            GOOD_BAD_PLOT_OPTIONS["bad"]["color"],
        ],
        label=[
            GOOD_BAD_PLOT_OPTIONS["good"]["label"],
            GOOD_BAD_PLOT_OPTIONS["bad"]["label"],
        ],
        alpha=0.7,
    )


def _plot_gmm_distances_overlayed_full_density(
    ax: Axes,
    distances_np: np.ndarray,
    *,
    dist_min: float,  # unused, maintained for consistency
    dist_max: float,  # unused, maintained for consistency
    idx_good: np.ndarray,
    idx_bad: np.ndarray,
    bins: int,
) -> None:
    for idx, image_type in ((idx_good, "good"), (idx_bad, "bad")):
        ax.hist(
            distances_np[idx],
            bins=bins,
            density=True,
            histtype=HISTOGRAM_TYPE,
            color=GOOD_BAD_PLOT_OPTIONS[image_type]["color"],
            label=GOOD_BAD_PLOT_OPTIONS[image_type]["label"],
            alpha=0.5,
        )


def _plot_gmm_distances_overlayed_proportional_density(
    ax: Axes,
    distances_np: np.ndarray,
    *,
    dist_min: float,
    dist_max: float,
    idx_good: np.ndarray,
    idx_bad: np.ndarray,
    bins: int,
) -> None:
    bin_edges = np.linspace(dist_min, dist_max, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    total_n = len(distances_np)

    for idx, image_type in ((idx_good, "good"), (idx_bad, "bad")):
        sub_data = distances_np[idx]
        # Weights scale each subgroup relative to total dataset count and bin width
        weights = np.ones_like(sub_data) / (total_n * bin_width)

        ax.hist(
            sub_data,
            bins=bin_edges,
            weights=weights,
            density=False,  # Disable density since weights handle scaling
            histtype=HISTOGRAM_TYPE,
            color=GOOD_BAD_PLOT_OPTIONS[image_type]["color"],
            label=GOOD_BAD_PLOT_OPTIONS[image_type]["label"],
            alpha=0.5,
        )


GMM_PLOT_FUNCTIONS = {
    "stacked": _plot_gmm_distances_stacked,
    "overlayed-full": _plot_gmm_distances_overlayed_full_density,
    "overlayed-proportional": _plot_gmm_distances_overlayed_proportional_density,
}
