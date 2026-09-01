"""Frequency-domain visualization utilities."""

from typing import Iterable

from cryo_robust.comparison.domain.frc import FRCData, FRCThreshold
from cryo_robust.comparison.domain.metrics import ClassificationMetrics
from cryo_robust.comparison.domain.reports import EvaluationReport, MethodEvaluation
from cryo_robust.comparison.evaluation.frc import get_threshold


import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cryo_robust.domain import ImageSpace

THRESHOLD_COLORS = {
    FRCThreshold.ONE_OVER_SEVEN: "tomato",
    FRCThreshold.ONE_HALF: "orange",
    FRCThreshold.HALF_BIT: "seagreen",
}


def _plot_frc_curves(
    data_items: list[tuple[str, FRCData]],
    frc_thresholds: list[FRCThreshold] | None = None,
    title: str = "Resolution Estimates (FRC)",
    x_axis_freqs: bool = True,
) -> Figure | None:
    """
    Plot Fourier Ring Correlation (FRC) curves.

    Parameters
    ----------
    data_items : list[tuple[str, FRCData]]
        A list of tuples containing the method name and its corresponding FRC data.
    frc_threshold : list[FRCThreshold] | None, optional
        A threshold value to draw as a horizontal dashed line. Default is None.
    title : str, optional
        The title of the axes. Default is "Resolution Estimates (FRC)".
    x_axis_freqs: bool, optional.
        Plot spatial frequencies instead of resolutions on the x-axis. Default is True.

    Returns
    -------
    Figure | None
        The generated figure, or None if `data_items` is empty.
    """
    # Early exit if no data is provided to avoid generating empty figures
    if not data_items:
        return None

    if frc_thresholds is None:
        frc_thresholds = []

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each curve with its corresponding method name as the legend label
    for name, frc_data in data_items:
        x = frc_data.freqs if x_axis_freqs else frc_data.spatial_resolutions
        ax.plot(x, frc_data.frc, label=name)

    frc_data = data_items[0][1]
    x_thresh = frc_data.freqs if x_axis_freqs else frc_data.spatial_resolutions

    # Optionally draw threshold lines
    for threshold in frc_thresholds:
        thr = get_threshold(frc_data, threshold)

        ax.plot(
            x_thresh,
            thr,
            linestyle="--",
            label=threshold,
            color=THRESHOLD_COLORS.get(threshold, "gray"),
        )

    xlabel = "Spatial Frequency (1/Å)" if x_axis_freqs else "Spatial Resolution"
    ax.set_xlabel(xlabel)

    ax.set_ylabel("Fourier Shell Correlation")
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_report_frc_curves(
    report: EvaluationReport, x_axis_freqs: bool = True
) -> tuple[Figure | None, Figure | None]:
    """
    Generate FRC curve figures for ground truth and half-set data from a report.

    Parameters
    ----------
    report : EvaluationReport
        The evaluation report containing the method results and FRC data.
    x_axis_freqs: bool, optional.
        Plot spatial frequencies instead of resolutions on the x-axis. Default is True.

    Returns
    -------
    tuple[Figure | None, Figure | None]
        A tuple containing the ground truth FRC figure and the half-set FRC figure.
        Either or both can be `None` if the respective data is not present.
    """
    # Extract ground truth FRC data only for methods where it exists
    gt_frc_items = [
        (mr.name, mr.ground_truth_frc_data)
        for mr in report.method_results
        if mr.ground_truth_frc_data is not None
    ]

    # Extract half-set FRC data only for methods where it exists
    hs_frc_items = [
        (mr.name, mr.half_set_frc_data)
        for mr in report.method_results
        if mr.half_set_frc_data is not None
    ]

    # Plot both sets of curves
    gt_fig = _plot_frc_curves(
        gt_frc_items,
        frc_thresholds=report.frc_thresholds,
        title="Ground Truth Resolution Estimates (FRC)",
        x_axis_freqs=x_axis_freqs,
    )
    hs_fig = _plot_frc_curves(
        hs_frc_items,
        frc_thresholds=report.frc_thresholds,
        title="Half-set Resolution Estimates (FRC)",
        x_axis_freqs=x_axis_freqs,
    )

    # Return the figures
    return gt_fig, hs_fig


def _extract_ring_data(
    ring_metrics_dict: dict[int, ClassificationMetrics], pixel_size: float = 1.0
) -> tuple[list[float], dict[str, list[float]]]:
    """
    Extract spatial frequencies and associated classification metrics from ring-based data.

    This helper sorts the ring indices, converts them into spatial frequencies using
    the inferred Fourier box size and pixel size, and collects selected metric values
    into parallel lists for downstream analysis or plotting.

    Parameters
    ----------
    ring_metrics_dict : dict[int, ClassificationMetrics]
        Mapping of ring (spatial frequency index) to its corresponding
        ``ClassificationMetrics`` object.
    pixel_size : float, optional
        Physical pixel size used to scale ring indices into spatial frequencies.
        Frequencies are computed as::

            frequency = ring / (box_size * pixel_size)

        where ``box_size = 2 * max(ring_metrics_dict.keys())``. Defaults to 1.0.

    Returns
    -------
    tuple[list[float], dict[str, list[float]]]
        A tuple containing:

        - ``freqs`` : list[float]
            Spatial frequencies corresponding to the sorted ring indices.
        - ``extracted`` : dict[str, list[float]]
            Dictionary of metric values aligned with ``freqs``. Keys include:

            - ``"ap"`` : Average precision values.
            - ``"roc_auc"`` : ROC AUC values.
            - ``"soft_precision"`` : Soft precision values.
            - ``"soft_recall_ht"`` : Soft recall values for the
              ``"huang_tagare"`` thresholding method.

        Missing attributes or metric values default to ``0.0``.

    Notes
    -----
    If ``ring_metrics_dict`` is empty, a fallback ``box_size`` of 1 is used and
    both returned collections will be empty.
    """
    sorted_rings = sorted(ring_metrics_dict.keys())
    box_size = 2 * max(sorted_rings) if sorted_rings else 1
    freqs = [ring / (box_size * pixel_size) for ring in sorted_rings]

    extracted = {"ap": [], "roc_auc": [], "soft_precision": [], "soft_recall_ht": []}
    for r in sorted_rings:
        m = ring_metrics_dict[r]
        extracted["ap"].append(getattr(m, "ap", 0.0))
        extracted["roc_auc"].append(getattr(m, "roc_auc", 0.0))
        extracted["soft_precision"].append(getattr(m, "soft_precision", 0.0))
        extracted["soft_recall_ht"].append(m.soft_recall.get("huang_tagare", 0.0))

    return freqs, extracted


def plot_method_fourier_ring_curves(
    method_results: MethodEvaluation,
    space: ImageSpace = ImageSpace.FOURIER_REAL,
    pixel_size: float = 1.0,
    figsize: tuple[float, float] = (11, 4.5),
) -> Figure | None:
    """
    Generates one classification metrics vs. Fourier frequency for one estimation method.
    Returns ``None`` if the estimator does not have valid weights in the requested space.

    Parameters
    ----------
    method_results : MethodResults
        MethodResults object containing the information about the requested method
    space : Space, optional
        Space to extract the weights from, by default Space.FOURIER_REAL
    pixel_size : float, optional
        Image pixel size, by default 1.0
    figsize : tuple[float, float], optional
        Figure size, by default (11, 4.5)

    Returns
    -------
    Figure | None
        Figure object containing the plot, or ``None`` if the estimation method
        did not have valid weights for the requested space
    """
    ring_metrics_dict = getattr(method_results, "fourier_ring_metrics", {}).get(space)
    if not ring_metrics_dict:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    freqs, data = _extract_ring_data(ring_metrics_dict, pixel_size=pixel_size)

    # Left subplot: Precision & Recall
    ax1.plot(freqs, data["soft_precision"], label="Soft Precision", color="teal", lw=2)
    ax1.plot(
        freqs,
        data["soft_recall_ht"],
        label="Soft Recall (Huang-Tagare)",
        color="darkorange",
        lw=2,
        linestyle="--",
    )
    ax1.set_title(
        f"Detection Metrics vs Frequency\n({method_results.name} - {space.label})"
    )
    ax1.set_xlabel(r"Spatial Frequency ($1/\mathrm{\AA}$)")
    ax1.set_ylabel("Metric Score")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left")

    # Right subplot: AP & ROC-AUC
    ax2.plot(freqs, data["ap"], label="Average Precision (AP)", color="crimson", lw=2)
    ax2.plot(
        freqs, data["roc_auc"], label="ROC-AUC", color="royalblue", lw=2, linestyle="-."
    )
    ax2.set_title(f"Classification Capacity vs Frequency\n({method_results.name})")
    ax2.set_xlabel(r"Spatial Frequency ($1/\mathrm{\AA}$)")
    ax2.set_ylabel("Metric Score")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    return fig


def plot_fourier_ring_summary(
    all_method_results: Iterable[MethodEvaluation],
    space: ImageSpace = ImageSpace.FOURIER_REAL,
    pixel_size: float = 1.0,
    figsize: tuple[int, int] = (8, 5),
) -> Figure | None:
    """
    Generates a single summary plot comparing all models across the spectrum.
    Solid line = Soft Precision, Dashed line = Soft Recall (Huang-Tagare).

    Parameters
    ----------
    all_method_results : Iterable[MethodResults]
        Iterable containing the MethodResults object for each of the estimation methods
    space : Space, optional
        Space from which weights will be extracted to calculate the metrics,
        by default Space.FOURIER_REAL
    pixel_size : float, optional
        Image pixel size, by default 1.0
    figsize : tuple[int, int], optional
        Figure size, by default (8, 5)

    Returns
    -------
    Figure | None
        The Figure object containing the plot, or
        ``None`` if no methods with valid weights for the requested space were provided.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # One color per method, same cmap as in plot_vs_snr
    cmap = plt.get_cmap("tab10")

    any_plots = False
    for idx, method_results in enumerate(all_method_results):
        ring_metrics_dict = getattr(method_results, "fourier_ring_metrics", {}).get(
            space
        )
        if not ring_metrics_dict:
            continue

        any_plots = True

        color = cmap(idx % cmap.N)
        freqs, data = _extract_ring_data(ring_metrics_dict, pixel_size=pixel_size)

        # Plot soft precision as solid line and soft recall as dashed line
        ax.plot(
            freqs,
            data["soft_precision"],
            label=f"{method_results.name}",
            color=color,
        )
        ax.plot(freqs, data["soft_recall_ht"], color=color, linestyle="--")

    if not any_plots:
        return None

    ax.set_title(f"Frequency evaluation comparison - {space.label}")
    ax.set_xlabel(r"Spatial Frequency ($1 / \mathrm{\AA}$)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add a custom helper legend text specifying line styles
    ax.text(
        0.02,
        0.05,
        "Solid = Precision\nDashed = Recall",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
        fontsize=9,
    )
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    return fig
