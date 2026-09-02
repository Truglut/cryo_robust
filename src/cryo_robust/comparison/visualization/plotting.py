from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from cryo_robust.comparison.visualization.plot_utils import ALL_PLOT_TYPES, save_figure
from cryo_robust.comparison.visualization.fourier_plots import (
    plot_fourier_ring_summary,
    plot_method_fourier_ring_curves,
    plot_report_frc_curves,
)
from cryo_robust.comparison.visualization.weight_plots import (
    collect_weight_scores,
    plot_weight_distributions,
)
from cryo_robust.comparison.visualization.gmm_plots import plot_report_gmm_fits

from cryo_robust.domain import ImageSpace
from cryo_robust.comparison.domain.reports import EvaluationReport

### ========================
### Complete report plotting
### ========================


def plot_report(
    report: EvaluationReport,
    max_subplots: int,
    plot_weights: bool = True,
    density: bool = False,
    plot_frc: bool = True,
    plot_gmm: bool = True,
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
    plot_gmm : bool, optional
        Whether to render the GMM distribution plots. Default is True.

    Returns
    -------
    `None`
    """
    if plot_weights:
        all_scores = collect_weight_scores(report)
        _ = plot_weight_distributions(all_scores, report.labels, max_subplots, density)
        plt.show()

    if plot_frc:
        gt_fig, hs_fig = plot_report_frc_curves(report)

        if gt_fig is not None:
            gt_fig.show()
        if hs_fig is not None:
            hs_fig.show()
        if gt_fig is not None or hs_fig is not None:
            plt.show()

    if plot_gmm:
        _ = plot_report_gmm_fits(report, labels=report.labels)
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
    title_suffix: str | None = None,
    plot_types: set[str] | None = None,
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
    title_suffix: str, optional.
        Suffix to append to weight plot titles.
    plot_types : set[str]
        Set of plots to include in the LaTeX report. This will also
        determine which plots get generated. The relevant options for this section
        are
        - "weights": weight histograms for each estimator and set of experimental
        conditions
        - "frc": FRC curves with ground-truth at each set of experimental conditions
        - "fourier-rings": classification metrics by Fourier ring for fourier-space
        methods.
        - "gmm": Plots of GMM fits

    Returns
    -------
    dict[str, list[Path]]
        Keys are
        - ``"weight_distributions"``,
        - ``"frc_curves"``,
        - ``"fourier_ring_classification"``,
        - ``"fourier_ring_summary"``, and
        - ``"gmm"``.

        Values are lists of saved file paths.

        - Weight distributions list has one entry per valid space, method and aggregation
        strategy combination.
        - FRC list has 0, 1 or 2 entries (ground truth FRC and/or half-set FRC or none)
        - Fourier ring classification list has one entry per valid fourier-space (real or
        imaginary) and method combination.
        - Fourier ring summary has 0, 1 or 2 entries (real and/or imaginary or none)
        - GMM plot list has one entry per estimator in the experiment that returned
         a valid GMMDiagnostics object (should be every gmm estimator).
    """
    report_figure_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[Path]] = {
        "weight_distributions": [],
        "frc_curves": [],
        "fourier_ring_classification": [],
        "fourier_ring_summary": [],
        "gmm": [],
    }

    if plot_types is None:
        plot_types = ALL_PLOT_TYPES

    # Weight distribution histograms
    if "weights" in plot_types:
        all_scores = collect_weight_scores(report)
        for i, fig in enumerate(
            plot_weight_distributions(
                all_scores,
                report.labels,
                max_subplots,
                density,
                title_suffix=title_suffix,
            )
        ):
            path = report_figure_path / f"weight_distribution_{i}.pdf"
            save_figure(fig=fig, path=path, dpi=dpi)
            saved["weight_distributions"].append(path)

    # FRC curves
    if "frc" in plot_types:
        gt_frc_fig, hs_frc_fig = plot_report_frc_curves(
            report, x_axis_freqs=frc_x_axis_freqs
        )
        if gt_frc_fig is not None:
            path = report_figure_path / "gt_frc_curves.pdf"
            save_figure(fig=gt_frc_fig, path=path, dpi=dpi)
            saved["frc_curves"].append(path)
        if hs_frc_fig is not None:
            path = report_figure_path / "hs_frc_curves.pdf"
            save_figure(fig=hs_frc_fig, path=path, dpi=dpi)
            saved["frc_curves"].append(path)

    ## Fourier ring classification metrics
    if "fourier-rings" in plot_types:
        for space in [ImageSpace.FOURIER_REAL, ImageSpace.FOURIER_IMAG]:
            space_str = "real" if space == ImageSpace.FOURIER_REAL else "imag"

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

                save_figure(fig=fig, path=fig_save_path, dpi=dpi)

                saved["fourier_ring_classification"].append(fig_save_path)

            # 2. Output global multi-method summary
            summary_fig = plot_fourier_ring_summary(
                report.method_results, space=space, pixel_size=pixel_size
            )

            if summary_fig is None:
                continue

            summary_filename = f"fourier_{space_str}_rings_summary.pdf"
            summary_save_path = report_figure_path / summary_filename

            save_figure(fig=summary_fig, path=summary_save_path, dpi=dpi)

            saved["fourier_ring_summary"].append(summary_save_path)

    if "gmm" in plot_types:
        gmm_figures = plot_report_gmm_fits(
            report=report, labels=report.labels, plot_initial_reference=True
        )

        for i, fig in enumerate(gmm_figures):
            save_name = f"gmm_fit_{i}.pdf"
            save_path = report_figure_path / save_name

            save_figure(fig=fig, path=save_path, dpi=dpi)

            saved["gmm"].append(save_path)

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
    title_suffix: str | None = None,
    plot_types: set[str] | None = None,
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
    pixel_size : float, optional
        Image pixel size. Default is 1.0.
    title_suffix : str, optional
        Suffix to append to weight plot titles.
    plot_types : set[str]
        Set of plots to include in the LaTeX report. This will also
        determine which plots get generated. The relevant options for this section
        are
        - "weights": weight histograms for each estimator and set of experimental
        conditions
        - "frc": FRC curves with ground-truth at each set of experimental conditions
        - "fourier-rings": classification metrics by Fourier ring for fourier-space
        methods.
        - "gmm": GMM fit plots

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
            title_suffix=title_suffix,
            plot_types=plot_types,
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
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ylabel: str = "Score",
    aggregated_data: bool = False,
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
    figsize : tuple[float, float], optional
        Figure size in inches as ``(width, height)``.
    title : str | None, optional
        Figure title. If ``None``, a title is generated automatically from
        the selected metrics.
    ylabel : str, optional
        Label of the y-axis. Default is ``"Score"``.
    aggregated_data: bool, optional
        Whether the data in the DataFrame is the result of aggregating multiple runs.
        If True, the requested metrics are plotted with error bars centered at the
        mean value for that metric, and radius equal to the std for that metric.

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
    if aggregated_data:
        required_metrics = [metric + "_mean" for metric in metrics]
    else:
        required_metrics = metrics
    required_columns = {
        method_column,
        snr_column,
        *required_metrics,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    methods: list[str] = sorted(df[method_column].unique())

    N_COLS = 2

    # Total items = (number of methods) * (number of metrics)
    total_legend_items = len(methods) * len(metrics)

    # Calculate rows (ceiling division)
    legend_rows = -(-total_legend_items // N_COLS)

    # Dynamically adjust height: base height + extra space per row
    if figsize is None:
        base_width = 5
        base_height = 3 + (legend_rows * 0.18)
        figsize = (base_width, base_height)

    # Create the figure with the adjusted dynamic size
    fig, ax = plt.subplots(figsize=(base_width, base_height))

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
            label = f"{method} — {metric_label}" if metric_label else method

            if aggregated_data:
                ax.errorbar(
                    x=method_df[snr_column],
                    y=method_df[metric + "_mean"],
                    yerr=method_df[metric + "_std"],
                    label=label,
                    color=color,
                    linestyle=linestyles[metric_idx % len(linestyles)],
                    marker=markers[metric_idx % len(markers)],
                    linewidth=1.2,
                )
            else:
                ax.plot(
                    method_df[snr_column],
                    method_df[metric],
                    label=label,
                    color=color,
                    linestyle=linestyles[metric_idx % len(linestyles)],
                    marker=markers[metric_idx % len(markers)],
                    linewidth=1.2,
                )

    # Set log scale
    ax.set_xscale("log")

    # 1. Extract exactly which SNR values exist in the data
    unique_snrs = sorted(df[snr_column].unique())

    # 2. Force matplotlib to place ticks exactly at these data points
    ax.set_xticks(unique_snrs)

    # 3. Format to 3 decimal places max, rounding 0.0067 to 0.007
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{round(x, 3):g}")
    )

    # 4. Clear out minor ticks so they don't create visual clutter
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    # Labels and Font sizes
    ax.set_xlabel("SNR", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    # Increase font size of tick numbers themselves
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(axis="x", which="minor", labelsize=8)

    # Lighten the grid lines
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.35)

    # Legend
    ax.legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=N_COLS,
    )

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    save_figure(fig=fig, path=save_path, dpi=dpi, pad_inches=0.02)

    return save_path
