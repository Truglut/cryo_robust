import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scipy.stats as stats

from .plot_utils import GOOD_BAD_PLOT_COLORS, HISTOGRAM_TYPE

from cryo_robust.estimators.results import GMMDiagnostics
from cryo_robust.comparison.domain.reports import EvaluationReport


def _plot_gmm_fit(
    diagnostics: GMMDiagnostics,
    idx_good: np.ndarray | None = None,
    idx_bad: np.ndarray | None = None,
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

    Returns
    -------
    Figure
        Figure object containing the plot.
    """
    fig = plt.figure()

    distances_np = diagnostics.distances.detach().cpu().numpy()

    dist_min = distances_np.min()
    dist_max = distances_np.max()
    length = dist_max - dist_min

    x = np.linspace(dist_min - 0.1 * length, dist_max + 0.1 * length, 1000)
    if diagnostics.standardized_distances:
        multiplier = 1.0 / distances_np.std()
        x_for_model = (x - distances_np.mean()) * multiplier
    else:
        multiplier = 1.0
        x_for_model = x

    bins = 40
    if idx_good is not None and idx_bad is not None:
        for idx, image_type in ((idx_good, "good"), (idx_bad, "bad")):
            plt.hist(
                distances_np[idx],
                bins=bins,
                density=True,
                histtype=HISTOGRAM_TYPE,
                color=GOOD_BAD_PLOT_COLORS[image_type],
                alpha=0.4,
            )
    else:
        plt.hist(distances_np, density=True, alpha=0.7)

    for i in range(2):
        mean = diagnostics.means[i]
        var = diagnostics.vars[i]
        weight = diagnostics.component_weights[i]

        # Calculate the component's density over the grid, accounting for the
        # possible change of variables when standardizing
        pdf = multiplier * weight * stats.norm.pdf(x_for_model, mean, np.sqrt(var))
        plt.plot(
            x,
            pdf,
            linestyle="--",
            linewidth=2,
            label=f"Gaussian {i+1} (w={weight:.2f})",
        )

    if title is not None:
        plt.title(title)

    return fig


def plot_report_gmm_fits(
    report: EvaluationReport,
    idx_good: np.ndarray | None = None,
    idx_bad: np.ndarray | None = None,
) -> list[Figure]:
    method_evaluations = report.method_results

    gmm_figures: list[Figure] = []

    for evaluation in method_evaluations:
        name = evaluation.name
        diagnostics = evaluation.result.diagnostics

        # Skip non-gmm estimators: they won't have gmm diagnostics
        if not isinstance(diagnostics, GMMDiagnostics):
            continue

        gmm_fig = _plot_gmm_fit(
            diagnostics, idx_good=idx_good, idx_bad=idx_bad, title=name
        )

        gmm_figures.append(gmm_fig)

    return gmm_figures
