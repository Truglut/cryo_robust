from pathlib import Path

from .figures import create_figure_section


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
    plots: dict[float, dict[str, list[Path]]], output_path: Path
) -> str:
    """
    Generates the LaTeX text for the plots section.

    Parameters
    ----------
    plots : dict[float, dict[str, list[Path]]]
        Dict mapping every SNR level to a dict which maps every figure type
        to a list of paths to its corresponding figures.
    output_path : Path
        Path to the directory where the `report.tex` will be generated.

    Returns
    -------
    str
        LaTeX text for the 'diagnostic plots' section. This section contains one
        subsection per SNR level. Each of these subsection contains:
            - Weight distribution histograms, for each method, space and aggregation strategy.
            - One plot representing the FRC curves for all methods.
    """
    text = "\n\\section{Diagnostic plots}\n"

    for snr in plots:
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
