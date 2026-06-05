from pathlib import Path

import pandas as pd

from method_comparison.domain.reports import EvaluationReport
from method_comparison.visualization.plotting import plot_vs_snr
from method_comparison.latex.tables import format_dataframe
from method_comparison.latex.figures import create_figure_block


def create_reconstruction_table(
    report: EvaluationReport,
    caption: str = "Métricas de reconstrucción para cada método",
) -> str:
    """
    Generate the reconstruction metrics section of the report.

    Parameters
    ----------
    report : EvaluationReport
        Evaluation report containing reconstruction metrics.

    Returns
    -------
    str
        LaTeX-formatted reconstruction metrics table.
    """
    reconstruction_df = report.reconstruction_metrics_dataframe()

    return format_dataframe(
        reconstruction_df,
        caption=caption,
    )


def generate_reconstruction_section(
    snr_reports: dict[float, EvaluationReport],
    output_path: Path,
    figures_path: Path,
    dpi: int = 150,
) -> str:
    """
    Generate the reconstruction metrics section of the LaTeX report.

    Produces one subsection per SNR level with a formatted metrics table,
    followed by a 'metrics vs. SNR' subsection containing plots for RMSE,
    Pearson correlation, and FRC resolution.

    Parameters
    ----------
    snr_reports : dict[float, EvaluationReport]
        Dict mapping every SNR level to its evaluation report.
    output_path : Path
        Path to the directory where ``report.tex`` will be generated.
        Used to compute relative figure paths.
    figures_path : Path
        Directory where generated figure files will be saved.
    dpi : int, optional
        Resolution for saved figures, by default 150.

    Returns
    -------
    str
        LaTeX text for the reconstruction section, ready to be written
        into the document.
    """
    if len(snr_reports) == 0:
        return ""

    text = "\n\\section{Reconstruction metrics}\n"

    reconstruction_dfs: dict[float, pd.DataFrame] = {}
    for snr, report in snr_reports.items():
        text += f"\n\\subsection{{SNR {snr:.3f}}}\n"
        text += create_reconstruction_table(
            report, caption=f"Reconstruction metrics for each method at SNR {snr:.3f}"
        )

        df = report.reconstruction_metrics_dataframe()
        df["snr"] = snr
        reconstruction_dfs[snr] = df

        frc_thresholds = report.frc_thresholds

    overall_rec_df = pd.concat(reconstruction_dfs.values())

    text += "\n\\subsection{Reconstruction metrics vs. SNR graphs}\n"

    snr_vs_rmse_plot = plot_vs_snr(
        df=overall_rec_df,
        metrics=["rmse"],
        save_path=figures_path / "snr_vs_rmse.pdf",
        metric_labels=[""],
        dpi=dpi,
        title="RMSE de reconstrucción frente a SNR",
        ylabel="RMSE",
    ).relative_to(output_path)

    text += "\n\\textbf{RMSE}\n"
    text += create_figure_block(
        snr_vs_rmse_plot, caption="Reconstruction RMSE vs SNR", width="0.8\\textwidth"
    )

    snr_vs_corr_plot = plot_vs_snr(
        df=overall_rec_df,
        metrics=["pearson_corr"],
        save_path=figures_path / "snr_vs_corr.pdf",
        metric_labels=[""],
        dpi=dpi,
        title="Correlación de la reconstrucción con original frente a SNR",
        ylabel="Correlación",
    ).relative_to(output_path)

    text += "\n\\textbf{Correlation}\n"
    text += create_figure_block(
        snr_vs_corr_plot,
        caption="Correlation with ground truth vs SNR",
        width="0.8\\textwidth",
    )

    snr_vs_gt_frc_plot = plot_vs_snr(
        df=overall_rec_df,
        metrics=["GT Resolution" + f"({thr.value})" for thr in frc_thresholds],
        save_path=figures_path / "snr_vs_gt_frc.pdf",
        metric_labels=[f"Resolution ({thr.value})" for thr in frc_thresholds],
        dpi=dpi,
        title="Ground truth reconstruction resolution vs SNR",
        ylabel="Resolution",
    ).relative_to(output_path)

    text += "\n\\textbf{Ground truth FRC}\n"
    text += "\n\\textbf{Resolution}\n"
    text += create_figure_block(
        snr_vs_gt_frc_plot,
        caption="FRC resolution vs SNR (comparing global average to ground truth)",
        width="0.8\\textwidth",
    )

    text += "\n\\textbf{AUFRC}\n"
    snr_vs_gt_aufrc_plot = plot_vs_snr(
        df=overall_rec_df,
        metrics=["AUFRC (GT)"],
        save_path=figures_path / "snr_vs_gt_aufrc.pdf",
        metric_labels=[""],
        title="Area under the ground-truth FRC curve vs. SNR",
        ylabel="AUFRC",
    ).relative_to(output_path)
    text += create_figure_block(
        snr_vs_gt_aufrc_plot,
        caption="Area under the ground-truth FRC curve vs. SNR",
        width="0.8\\textwidth",
    )

    snr_vs_hs_frc_plot = plot_vs_snr(
        df=overall_rec_df,
        metrics=["HS Resolution" + f"({thr.value})" for thr in frc_thresholds],
        save_path=figures_path / "snr_vs_hs_frc.pdf",
        metric_labels=[f"Resolution ({thr.value})" for thr in frc_thresholds],
        dpi=dpi,
        title="Half-set reconstruction resolution vs SNR",
        ylabel="Resolution",
    ).relative_to(output_path)

    text += "\n\\textbf{Half-set FRC}\n"
    text += "\n\\textbf{Resolution}\n"
    text += create_figure_block(
        snr_vs_hs_frc_plot,
        caption="FRC resolution vs SNR (comparing half-set averages)",
        width="0.8\\textwidth",
    )

    text += "\n\\textbf{AUFRC}\n"
    snr_vs_hs_aufrc_plot = plot_vs_snr(
        df=overall_rec_df,
        metrics=["AUFRC (HS)"],
        save_path=figures_path / "snr_vs_hs_aufrc.pdf",
        metric_labels=[""],
        title="Area under the half-set FRC curve vs. SNR",
        ylabel="AUFRC",
    ).relative_to(output_path)
    text += create_figure_block(
        snr_vs_hs_aufrc_plot,
        caption="Area under the half-set FRC curve vs. SNR",
        width="0.8\\textwidth",
    )

    return text
