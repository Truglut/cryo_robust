from pathlib import Path
from typing import Sequence, Iterable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from method_comparison.domain.enums import Space
from method_comparison.domain.metrics import ClassificationMetrics
from method_comparison.domain.reports import EvaluationReport, MethodResults
from method_comparison.evaluation.frc import FRCData, FRCThreshold, get_threshold

# Helper for consistent coloring
LABEL_MAP = {
    0: {"name": "Inliers", "color": "blue"},
    1: {"name": "Rotated Outliers", "color": "orange"},
    2: {"name": "Misclassified", "color": "red"},
    3: {"name": "Noise", "color": "darkorange"},
}

THRESHOLD_COLORS = {
    FRCThreshold.ONE_OVER_SEVEN: "tomato",
    FRCThreshold.ONE_HALF: "orange",
    FRCThreshold.HALF_BIT: "seagreen",
}

AVERAGE_NAME = "Average"

BASE_PLOT_OPTIONS = {
    "max_subplots": 3,
    "density": False,
    "dpi": 150,
}


### ====================
### Weight distributions
### ====================


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


### ==========
### FRC curves
### ==========


def _plot_frc_curves(
    data_items: list[tuple[str, FRCData]],
    frc_thresholds: list[FRCThreshold] = [],
    title: str = "Resolution Estimates (FRC)",
    x_axis_freqs: bool = True,
) -> plt.Figure | None:
    """
    Plot Fourier Ring Correlation (FRC) curves.

    Parameters
    ----------
    data_items : list[tuple[str, FRCData]]
        A list of tuples containing the method name and its corresponding FRC data.
    frc_threshold : float | None, optional
        A threshold value to draw as a horizontal dashed line. Default is None.
    title : str, optional
        The title of the axes. Default is "Resolution Estimates (FRC)".
    x_axis_freqs: bool, optional.
        Plot spatial frequencies instead of resolutions on the x-axis. Default is True.

    Returns
    -------
    plt.Figure | None
        The generated figure, or None if `data_items` is empty.
    """
    # Early exit if no data is provided to avoid generating empty figures
    if not data_items:
        return None

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
) -> tuple[plt.Figure | None, plt.Figure | None]:
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
    tuple[plt.Figure | None, plt.Figure | None]
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


### ===================================
### Fourier ring classification metrics
### ===================================


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
    method_results: MethodResults,
    space: Space = Space.FOURIER_REAL,
    pixel_size: float = 1.0,
    figsize: tuple[int, int] = (11, 4.5),
) -> plt.Figure | None:
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
    figsize : tuple[int, int], optional
        Figure size, by default (11, 4.5)

    Returns
    -------
    plt.Figure | None
        plt.Figure object containing the plot, or ``None`` if the estimation method
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
    all_method_results: Iterable[MethodResults],
    space: Space = Space.FOURIER_REAL,
    pixel_size: float = 1.0,
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure | None:
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
    plt.Figure | None
        The plt.Figure object containing the plot, or
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


### ========================
### Complete report plotting
### ========================


def plot_report(
    report: EvaluationReport,
    max_subplots: int,
    plot_weights: bool = True,
    density: bool = False,
    plot_frc: bool = True,
) -> None:
    """
    Produce all diagnostic plots for an `EvaluationReport`.

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
    plot_frc : bool, optional
        Whether to render the FRC curve comparison plot. Default is True.

    Returns
    -------
    `None`
    """
    if plot_weights:
        all_scores = _collect_weight_scores(report)
        _ = _plot_weight_distributions(all_scores, report.labels, max_subplots, density)
        plt.show()

    if plot_frc:
        gt_fig, hs_fig = plot_report_frc_curves(report)

        if gt_fig is not None:
            gt_fig.show()
        if hs_fig is not None:
            hs_fig.show()
        if gt_fig is not None or hs_fig is not None:
            plt.show()


### ===============================
### Figure saving for LaTeX reports
### ===============================


def save_report_figures(
    report: EvaluationReport,
    report_figure_path: Path,
    max_subplots: int,
    density: bool = False,
    dpi: int = 150,
    frc_x_axis_freqs: bool = True,
    pixel_size: float = 1.0,
) -> dict[str, list[Path]]:
    """
    Save all report figures to disk and return their paths.

    Parameters
    ----------
    report : EvaluationReport
        Populated evaluation report.
    report_figure_path : Path
        Directory in which figures are saved. Created if absent.
    max_subplots : int
        Maximum subplots per weight-distribution figure.
    density : bool, optional
        Whether to normalise histograms to probability density.
    dpi : int, optional
        Output resolution in dots per inch. Default is 150.
    frc_x_axis_freqs: bool, optional
        Plot frequencies instead of spatial resolution on the x-axis in FRC plots.
    pixel_size: float, optional
        Image pixel size. Default is 1.0.

    Returns
    -------
    dict[str, list[Path]]
        Keys are
        - ``"weight_distributions"``,
        - ``"frc_curves"``,
        - ``"fourier_ring_classification"``, and
        - ``"fourier_ring_summary"``.

        Values are lists of saved file paths.

        - Weight distributions list has one entry per valid space, method and aggregation
        strategy combination.
        - FRC list has 0, 1 or 2 entries (ground truth FRC and/or half-set FRC or none)
        - Fourier ring classification list has one entry per valid fourier-space (real or
        imaginary) and method combination.
        - Fourier ring summary has 0, 1 or 2 entries (real and/or imaginary or none)
    """
    report_figure_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[Path]] = {
        "weight_distributions": [],
        "frc_curves": [],
        "fourier_ring_classification": [],
        "fourier_ring_summary": [],
    }

    all_scores = _collect_weight_scores(report)
    for i, fig in enumerate(
        _plot_weight_distributions(all_scores, report.labels, max_subplots, density)
    ):
        path = report_figure_path / f"weight_distribution_{i}.pdf"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["weight_distributions"].append(path)

    # FRC curves
    gt_frc_fig, hs_frc_fig = plot_report_frc_curves(
        report, x_axis_freqs=frc_x_axis_freqs
    )
    if gt_frc_fig is not None:
        path = report_figure_path / "gt_frc_curves.pdf"
        gt_frc_fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(gt_frc_fig)
        saved["frc_curves"].append(path)
    if hs_frc_fig is not None:
        path = report_figure_path / "hs_frc_curves.pdf"
        hs_frc_fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(hs_frc_fig)
        saved["frc_curves"].append(path)

    ## Fourier ring classification metrics
    for space in [Space.FOURIER_REAL, Space.FOURIER_IMAG]:
        space_str = "real" if space == Space.FOURIER_REAL else "imag"

        # 1. Output individual method subplot figures
        for res in report.method_results:
            fig = plot_method_fourier_ring_curves(
                res, space=space, pixel_size=pixel_size
            )

            if fig is None:
                continue

            clean_name = res.name.lower().replace(" ", "_")
            fig_filename = f"fourier_{space_str}_rings_{clean_name}.pdf"
            fig_save_path = report_figure_path / fig_filename

            fig.savefig(fig_save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            saved["fourier_ring_classification"].append(fig_save_path)

        # 2. Output global multi-method summary
        summary_fig = plot_fourier_ring_summary(
            report.method_results, space=space, pixel_size=pixel_size
        )

        if summary_fig is None:
            continue

        summary_filename = f"fourier_{space_str}_rings_summary.pdf"
        summary_save_path = report_figure_path / summary_filename

        summary_fig.savefig(summary_save_path, dpi=dpi, bbox_inches="tight")
        plt.close(summary_fig)

        saved["fourier_ring_summary"].append(summary_save_path)

    return saved


def save_snr_reports_figures(
    snr_reports: dict[float, EvaluationReport],
    output_path: Path,
    figures_path: Path,
    max_subplots: int,
    density: bool = False,
    dpi: int = 150,
    frc_x_axis_freqs: bool = True,
    pixel_size: float = 1.0,
) -> dict[float, dict[str, list[Path]]]:
    """
    Save all report figures to disk and return their paths.

    Parameters
    ----------
    snr_reports : dict[float, EvaluationReport]
        Dict mapping every SNR value to its corresponding evaluation report.
    output_path : Path
        Directory in which figures are saved. Created if absent.
    max_subplots : int
        Maximum subplots per weight-distribution figure.
    density : bool, optional
        Whether to normalise histograms to probability density.
    dpi : int, optional
        Output resolution in dots per inch. Default is 150.
    frc_x_axis_freqs: bool, optional
        Plot frequencies instead of spatial resolution on the x-axis in FRC plots.
    pixel_size: float, optional
        Image pixel size. Default is 1.0.

    Returns
    -------
    dict[float, dict[str, list[Path]]]
        Maps every SNR value to a dict with keys ``"weight_distributions"`` and ``"frc_curves"``,
        whose values are lists of saved file paths (FRC list has 0, 1 or 2 entries).
    """
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[float, dict[str, list[Path]]] = dict()
    for snr, report in snr_reports.items():
        snr_str = f"snr_{snr:.3f}".replace(".", "p")
        snr_figures_output = figures_path / snr_str
        snr_figures_output.mkdir(parents=True, exist_ok=True)

        saved[snr] = save_report_figures(
            report=report,
            report_figure_path=snr_figures_output,
            max_subplots=max_subplots,
            density=density,
            dpi=dpi,
            frc_x_axis_freqs=frc_x_axis_freqs,
            pixel_size=pixel_size,
        )

    return saved

### ===============================
### Plotting metrics vs. SNR levels
### ===============================

def plot_vs_snr(
    df: pd.DataFrame,
    metrics: str | Sequence[str],
    save_path: str | Path,
    *,
    metric_labels: Sequence[str] | str | None = None,
    method_column: str = "method",
    snr_column: str = "snr",
    dpi: int = 150,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
    ylabel: str = "Score",
) -> Path:
    """
    Plot one or more metrics as a function of SNR for each method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the method, SNR, and metric columns.
    metrics : str | Sequence[str]
        Metric column name or sequence of metric column names to plot.
    save_path : str | Path
        Full output path of the saved figure.
    metric_labels : Sequence[str] | None, optional
        Display names for metrics in the legend. If ``None``, metric column
        names are used directly.
    method_column : str, optional
        Name of the dataframe column identifying reconstruction or evaluation
        methods. Default is ``"method"``.
    snr_column : str, optional
        Name of the dataframe column containing SNR values.
        Default is ``"snr"``.
    dpi : int, optional
        Output resolution in dots per inch. Default is 150.
    figsize : tuple[int, int], optional
        Figure size in inches as ``(width, height)``.
        Default is ``(10, 6)``.
    title : str | None, optional
        Figure title. If ``None``, a title is generated automatically from
        the selected metrics.
    ylabel : str, optional
        Label of the y-axis. Default is ``"Score"``.

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    ValueError
        If required dataframe columns are missing.
    ValueError
        If ``metrics`` and ``metric_labels`` have different lengths.
    """

    # Normalize inputs
    if isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)

    if metric_labels is None:
        metric_labels = metrics
    elif isinstance(metric_labels, str):
        metric_labels = [metric_labels]
    else:
        metric_labels = list(metric_labels)

    if len(metrics) != len(metric_labels):
        raise ValueError("`metrics` and `metric_labels` must have the same length")

    # Validate columns
    required_columns = {
        method_column,
        snr_column,
        *metrics,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    methods = sorted(df[method_column].unique())

    # One color per method
    cmap = plt.get_cmap("tab10")

    # Reusable style cycles
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v"]

    # For each method, plot metrics vs snr
    for method_idx, method in enumerate(methods):
        method_df = df[df[method_column] == method].sort_values(snr_column)

        color = cmap(method_idx % cmap.N)

        # Plot every metric in a different style
        for metric_idx, (metric, metric_label) in enumerate(
            zip(metrics, metric_labels)
        ):
            ax.plot(
                method_df[snr_column],
                method_df[metric],
                label=f"{method} — {metric_label}",
                color=color,
                linestyle=linestyles[metric_idx % len(linestyles)],
                marker=markers[metric_idx % len(markers)],
            )

    ax.set_xscale("log")

    ax.set_xlabel("SNR")
    ax.set_ylabel(ylabel)

    if title is None:
        if len(metric_labels) == 1:
            title = f"{metric_labels[0]} vs SNR"
        else:
            title = "Metrics vs SNR"

    ax.set_title(title)

    ax.grid(True, which="both", linestyle=":")
    ax.legend()

    fig.tight_layout()

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return save_path


### =================================================================
### Image generation and saving (for ground truth/estimated averages)
### =================================================================

def generate_image_plots(
    images: Iterable[np.ndarray],
    save_paths: Iterable[Path],
    link_contrast: bool = True,
    *,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 150,
) -> list[Path]:
    """
    Generate and save individual plots for a sequence of images.

    Parameters
    ----------
    images : Iterable[np.ndarray]
        An iterable of 2D arrays representing the images to plot.
    save_paths : Iterable[Path]
        An iterable of file paths where the corresponding images should be saved.
    link_contrast : bool, optional
        If True, applies a global minimum and maximum contrast across all images.
        Default is True.
    figsize : tuple[int, int], optional
        The dimensions of each generated figure in inches. Default is (6, 6).
    dpi : int, optional
        The resolution of the saved figures in dots per inch. Default is 150.

    Returns
    -------
    list[Path]
        A list of paths where the images were saved.

    Raises
    ------
    ValueError
        If the number of images and save paths do not match.
    """
    # Convert iterables to lists to safely calculate length and iterate multiple times
    images = list(images)
    save_paths = list(save_paths)

    if len(images) != len(save_paths):
        raise ValueError("images and save_paths must have the same length")

    # Determine global contrast limits if requested
    if link_contrast:
        vmin = min([image.min() for image in images])
        vmax = max([image.max() for image in images])
    else:
        vmin = None
        vmax = None

    for image, save_path in zip(images, save_paths):
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(image, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)

        # Remove axes and whitespace for a clean image output
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save cleanly
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)

        plt.close(fig)

    return save_paths
