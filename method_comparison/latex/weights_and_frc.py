from pathlib import Path
from typing import Any

from method_comparison.domain.reports import EvaluationReport, EvaluationStudy
from method_comparison.latex.figures import create_figure_section
from method_comparison.visualization.plotting import save_snr_reports_figures


def weights_and_frc_plots_latex(
    saved_figures: dict[str, list[Path]], output_path: Path
) -> str:
    """
    Generate a complete LaTeX plots subsection from saved figure paths.

    Parameters
    ----------
    saved_figures : dict[str, list[Path]]
        Mapping returned by `save_report_figures`, with keys
        `"weight_distributions"` and `"frc_curves"`.
    output_path: Path
        Path to the directory that contains the `report.tex` file.

    Returns
    -------
    str
        LaTeX section string ready to be written into a document.
    """
    text = ""
    weight_paths = [
        p.relative_to(output_path)
        for p in saved_figures.get("weight_distributions", [])
    ]
    if weight_paths:
        text += "\n\\subsubsection{Weight Distributions}\n"
        text += create_figure_section(weight_paths, "Weight distribution")

    frc_paths = [
        p.relative_to(output_path) for p in saved_figures.get("frc_curves", [])
    ]
    if frc_paths:
        text += "\n\\subsubsection{FRC Curves}\n"
        text += create_figure_section(frc_paths, "FRC curves")

    return text


def generate_weight_and_frc_plots_section(
    results: dict[float, EvaluationReport] | dict[float, EvaluationStudy],
    output_path: Path,
    figures_path: Path,
    plot_options: dict[str, Any],
    frc_x_axis_freqs: bool = True,
) -> str:
    """
    Generates the LaTeX text for the plots section.

    Parameters
    ----------
    results : dict[float, EvaluationReport]
        Dict mapping every SNR level to its evaluation report or dict mapping
        each SNR level to its evaluation study containing one report per simulation
        run.
    output_path : Path
        Path to the directory where the `report.tex` will be generated.
    figures_path : Path
        Path to the directory where the figures will be saved.
    plot_options : dict[str, Any]
        Dict containing the following keyword arguments for figure generation:
            - max_subplots: int
            - density: bool
            - dpi: int
    frc_x_axis_freqs: bool, optional
        Plot frequencies instead of spatial resolution on the x-axis in FRC plots.

    Returns
    -------
    str
        LaTeX text for the 'diagnostic plots' section. This section contains one
        subsection per SNR level. Each of these subsection contains:
            - Weight distribution histograms, for each method, space and aggregation strategy.
            - One plot representing the FRC curves for all methods.
    """
    # If results are EvaluationStudy, take the first report for each snr
    report = list(results.values())[0]
    if isinstance(report, EvaluationStudy):
        results = {snr: study.reports[0] for snr, study in results.items()}

    plots = save_snr_reports_figures(
        results,
        output_path=output_path,
        figures_path=figures_path,
        frc_x_axis_freqs=frc_x_axis_freqs,
        **plot_options,
    )

    text = "\n\\section{Diagnostic plots}\n"

    for snr in results:
        text += f"\n\\subsection{{SNR {snr:.3f}}}\n"
        text += weights_and_frc_plots_latex(
            saved_figures=plots[snr], output_path=output_path
        )

        fourier_ring_classification_figpaths = [
            p.relative_to(output_path)
            for p in plots[snr].get("fourier_ring_classification", [])
        ]
        fourier_ring_summary_figpaths = [
            p.relative_to(output_path)
            for p in plots[snr].get("fourier_ring_summary", [])
        ]

        if fourier_ring_classification_figpaths or fourier_ring_summary_figpaths:
            text += "\n\\subsubsection{Fourier Ring Classification Metrics}\n"

            text += "\n\\textbf{Per-method metrics vs frequency}\n"
            text += create_figure_section(
                fourier_ring_classification_figpaths,
                caption_prefix="Classification metrics in each Fourier ring",
                width="0.85\\textwidth",
            )

            text += "\n\\textbf{Classification metrics vs. Frequency: summary}\n"
            text += create_figure_section(
                fourier_ring_summary_figpaths,
                caption_prefix="Classification metrics vs. Frequency summary",
                width="0.65\\textwidth",
            )

    return text
