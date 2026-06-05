from pathlib import Path

import pandas as pd

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.domain.reports import EvaluationReport
from method_comparison.visualization.plotting import AVERAGE_NAME, plot_vs_snr
from method_comparison.latex.tables import format_dataframe
from method_comparison.latex.figures import create_figure_block


def get_classification_table(
    classification_df: pd.DataFrame, space: Space, strategy: AggregationStrategy
) -> pd.DataFrame | None:
    """
    Extract a classification metrics sub-table for a given space and strategy.

    Filters ``classification_df`` to rows matching ``space`` and
    ``strategy``, and returns ``None`` when the result would be empty or
    consist solely of the baseline average method.

    Parameters
    ----------
    classification_df : pd.DataFrame
        Full classification metrics dataframe from
        ``EvaluationReport.classification_metrics_dataframe()``.
    space : Space
        Reconstruction space to filter on (matched against the ``"space"``
        column by name).
    strategy : AggregationStrategy
        Aggregation strategy to filter on (matched against the
        ``"aggregation_strategy"`` column by value).

    Returns
    -------
    pd.DataFrame | None
        Filtered dataframe, or ``None`` if no relevant rows exist.
    """
    space_rows = classification_df["space"] == space.name

    if not space_rows.any():
        return None

    space_df = classification_df[space_rows]

    # Skip sections containing only the baseline average method.
    if len(space_df.index) == 1 and space_df["method"].iloc[0] == AVERAGE_NAME:
        return None

    rows = space_df["aggregation_strategy"] == strategy.value

    if not rows.any():
        return None

    return space_df[rows]


def generate_classification_tables(report: EvaluationReport) -> str:
    """
    Generate the classification metrics section of the report for one SNR level.

    Tables are grouped by:
    - reconstruction space
    - aggregation strategy

    Empty groups and groups containing only the baseline average
    method are skipped.

    Parameters
    ----------
    report : EvaluationReport
        Evaluation report containing classification metrics.

    Returns
    -------
    str
        LaTeX-formatted classification section.
    """
    classification_df = report.classification_metrics_dataframe()
    text = ""

    for space in Space:
        # Filter out spaces that have no associated methods or only the baseline average
        space_rows = classification_df["space"] == space.name
        space_df = classification_df[space_rows]
        if len(space_df.index) == 0:
            continue
        if len(space_df.index) == 1 and space_df["method"].iloc[0] == AVERAGE_NAME:
            continue

        text += f"\n\\subsubsection{{{space.label}}}\n"

        for strategy in AggregationStrategy:
            df = get_classification_table(classification_df, space, strategy)
            if df is None:
                continue

            text += f"\n\\textbf{{{strategy.label}}}\n\n\\smallskip\n\n"

            # Remove grouping columns before rendering the table.
            df = df.drop(["space", "aggregation_strategy"], axis=1)

            text += format_dataframe(df)

    return text


def generate_classification_section(
    snr_reports: dict[float, EvaluationReport],
    output_path: Path,
    figures_path: Path,
    dpi: int = 150,
) -> str:
    """
    Generates the complete classification section of the report, including all SNR
    subsections and metrics-vs-snr plots.

    Parameters
    ----------
    snr_reports : dict[float, EvaluationReport]
        Dict mapping every SNR level to its evaluation report.
    output_path : Path
        Path to the directory where the `report.tex` will be generated.
    figures_path : Path
        Path to the directory where the figures will be saved
    dpi : int, optional
        Resolution for saved figures, by default 150.

    Returns
    -------
    str
        LaTeX text for the classification section of the report, ready to be written
        into the document. This section contains:
            - One subsection per SNR level. Each of these subsections has a
            classification metrics table per space and aggregation strategy where
            methods have been tested.
            - One 'metrics vs. SNR' section, containing a plot of soft precision
            and soft recall (calculated according to Huang and Tagare) vs. SNR.
    """
    text = "\n\\section{Classification metrics}\n"

    classification_dfs: dict[float, pd.DataFrame] = {}
    for snr, report in snr_reports.items():
        text += f"\n\\subsection{{SNR {snr:.3f}}}\n"
        text += "\n"
        text += generate_classification_tables(report)

        classification_dfs[snr] = report.classification_metrics_dataframe()
        classification_dfs[snr]["snr"] = snr

    overall_df = pd.concat(classification_dfs.values())

    # For every space and aggregation strategy, plot precision and recall vs snr
    # and average precision vs snr
    text += "\n\\subsection{Classification metrics vs SNR}"
    for space in Space:
        space_rows = overall_df["space"] == space.name
        space_df = overall_df[space_rows]

        non_average = space_df[space_df["method"] != AVERAGE_NAME]

        if non_average.empty:
            continue

        text += f"\n\\subsubsection{{{space.label}}}\n"

        for strategy in AggregationStrategy:
            df = get_classification_table(overall_df, space, strategy)
            if df is None:
                continue

            text += f"\n\\textbf{{{strategy.label}}}\n"

            space_strategy_identifier = f"{space.name.lower()}_{strategy.value.lower()}"

            precision_curve = plot_vs_snr(
                df,
                metrics=["soft_precision"],
                save_path=figures_path
                / f"snr_vs_softprec_{space_strategy_identifier}.pdf",
                metric_labels=[""],
                dpi=dpi,
                title="Precisión ($\\hat{P}$) frente a SNR",
                ylabel="Precisión",
            ).relative_to(output_path)
            text += create_figure_block(
                precision_curve,
                caption=f"Soft precision vs. SNR ({space.label} - {strategy.label})",
            )

            recall_curve = plot_vs_snr(
                df,
                metrics=["soft_recall_huang_tagare"],
                save_path=figures_path
                / f"snr_vs_softrec_{space_strategy_identifier}.pdf",
                metric_labels=[""],
                dpi=dpi,
                title="Sensibilidad ($\\hat{R}$) frente a SNR",
                ylabel="Sensibilidad",
            ).relative_to(output_path)
            text += create_figure_block(
                recall_curve,
                caption=f"Soft precision vs. SNR ({space.label} - {strategy.label})",
            )

            precision_and_recall_curves = plot_vs_snr(
                df,
                metrics=["soft_precision", "soft_recall_huang_tagare"],
                save_path=figures_path / f"snr_vs_pr_{space_strategy_identifier}.pdf",
                metric_labels=["Soft precision", "Soft recall"],
                dpi=dpi,
                title="Precision and recall vs SNR",
                ylabel="Score",
            ).relative_to(output_path)
            text += create_figure_block(
                precision_and_recall_curves,
                caption=f"Soft precision and soft recall vs. SNR ({space.label} - {strategy.label})",
            )

            average_precision_curve = plot_vs_snr(
                df,
                metrics=["ap"],
                save_path=figures_path / f"snr_vs_ap_{space_strategy_identifier}.pdf",
                metric_labels=[""],
                dpi=dpi,
                title="Average precision frente a SNR",
                ylabel="AP",
            ).relative_to(output_path)
            text += create_figure_block(
                average_precision_curve,
                caption=f"Average precision and AUC vs. SNR ({space.label} - {strategy.label})",
            )

            roc_auc_curve = plot_vs_snr(
                df,
                metrics=["roc_auc"],
                save_path=figures_path
                / f"snr_vs_roc_auc_{space_strategy_identifier}.pdf",
                metric_labels=[""],
                dpi=dpi,
                title="ROC-AUC frente a SNR",
                ylabel="AUC",
            ).relative_to(output_path)
            text += create_figure_block(
                roc_auc_curve,
                caption=f"Average precision and AUC vs. SNR ({space.label} - {strategy.label})",
            )

            average_precision_auc_curves = plot_vs_snr(
                df,
                metrics=["ap", "roc_auc"],
                save_path=figures_path
                / f"snr_vs_ap_and_auc_{space_strategy_identifier}.pdf",
                metric_labels=["Average precision", "AUC-ROC"],
                dpi=dpi,
                title="Average precision and AUC vs. SNR",
                ylabel="Score",
            ).relative_to(output_path)
            text += create_figure_block(
                average_precision_auc_curves,
                caption=f"Average precision and AUC vs. SNR ({space.label} - {strategy.label})",
            )

    return text
