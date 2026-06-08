from pathlib import Path

import pandas as pd

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.domain.reports import EvaluationReport, EvaluationStudy
from method_comparison.visualization.plotting import AVERAGE_NAME, plot_vs_snr
from method_comparison.latex.tables import format_dataframe
from method_comparison.latex.figures import create_figure_block


def generate_classification_section(
    results: dict[float, EvaluationReport] | dict[float, EvaluationStudy],
    output_path: Path,
    figures_path: Path,
    dpi: int = 150,
) -> str:
    """
    Generates the complete classification section of the report, including all SNR
    subsections and metrics-vs-snr plots.
    Decides whether to call ``generate_reconstruction_section_from_report`` or
    ``generate_reconstruction_section_from_study`` based on the properties of
    ``results``.

    Parameters
    ----------
    results : dict[float, EvaluationReport] or dict[float, EvaluationStudy]
        Dict mapping every SNR level to its evaluation report, or dict mapping
        every SNR level to its evaluation study (mixed configurations are not
        supported).
        If values are of type EvaluationReport,
        ``generate_reconstruction_section_from_report`` is called.
        If values of type EvaluationStudy, then every study's reports list should
        have the same length. Three cases are possible based on the length of
        the reports lists:
          - If the report lists are empty, an empty string is returned.
          - If the report lists are length one, a dict[float, EvaluationReport] is
          built and then ``generate_reconstruction_section_from_report`` is called.
          - If the report lists have length greater than one, then
          ``generate_reconstruction_section_from_study`` is called.
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

    if not results:
        return ""

    value = list(results.values())[0]

    if isinstance(value, EvaluationReport):
        return generate_classification_section_from_report(
            snr_reports=results,
            output_path=output_path,
            figures_path=figures_path,
            dpi=dpi,
        )
    if not value.reports:
        return ""
    if len(value.reports) == 1:
        results = {snr: study.reports[0] for snr, study in results.items()}
        return generate_classification_section_from_report(
            snr_reports=results,
            output_path=output_path,
            figures_path=figures_path,
            dpi=dpi,
        )

    return generate_classification_section_from_study(
        snr_studies=results, output_path=output_path, figures_path=figures_path, dpi=dpi
    )


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


def generate_classification_tables(classification_df: pd.DataFrame) -> str:
    """
    Generate the classification metrics section of the report for one SNR level.

    Tables are grouped by:
    - reconstruction space
    - aggregation strategy

    Empty groups and groups containing only the baseline average
    method are skipped.

    Parameters
    ----------
    classification_df : pd.DataFrame
        DataFrame containing classification metrics.

    Returns
    -------
    str
        LaTeX-formatted classification section.
    """
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


def generate_classification_section_from_report(
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

    classification_dfs: list[pd.DataFrame] = []
    for snr, report in snr_reports.items():
        text += f"\n\\subsection{{SNR {snr:.3f}}}\n"
        text += "\n"
        classification_df = report.classification_metrics_dataframe()
        text += generate_classification_tables(classification_df)

        classification_df["snr"] = snr
        classification_dfs.append(classification_df)

    overall_df = pd.concat(classification_dfs)

    # For every space and aggregation strategy, plot precision, recall,
    # average precision and roc-auc vs SNR
    text += "\n\\subsection{Classification metrics vs SNR}"
    for space in Space:
        space_rows = overall_df["space"] == space.name
        space_df = overall_df[space_rows]

        non_average = space_df[space_df["method"] != AVERAGE_NAME]

        if non_average.empty:
            continue

        text += f"\n\\subsubsection{{{space.label}}}\n"

        for strategy in AggregationStrategy:
            text += generate_classification_plots(
                overall_df=overall_df,
                space=space,
                strategy=strategy,
                output_path=output_path,
                figures_path=figures_path,
                dpi=dpi,
                aggregated_data=False,
            )

    return text


def generate_classification_section_from_study(
    snr_studies: dict[float, EvaluationStudy],
    output_path: Path,
    figures_path: Path,
    dpi: int = 150,
) -> str:
    """
    Generates the complete classification section of the report, including all SNR
    subsections and metrics-vs-snr plots.

    Parameters
    ----------
    snr_reports : dict[float, EvaluationStudy]
        Dict mapping every SNR level to its evaluation study, which contains a
        list of evaluation reports, one for each simulation run.
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

    classification_dfs: list[pd.DataFrame] = []
    for snr, study in snr_studies.items():
        text += f"\n\\subsection{{SNR {snr:.3f}}}\n"
        text += "\n"

        classification_df = study.aggregate_classification_metrics()
        text += generate_classification_tables(classification_df)

        classification_df["snr"] = snr
        classification_dfs.append(classification_df)

    overall_df = pd.concat(classification_dfs)

    # For every space and aggregation strategy, plot precision, recall,
    # average precision and roc-auc vs SNR
    text += "\n\\subsection{Classification metrics vs SNR}"
    for space in Space:
        space_rows = overall_df["space"] == space.name
        space_df = overall_df[space_rows]

        non_average = space_df[space_df["method"] != AVERAGE_NAME]

        if non_average.empty:
            continue

        text += f"\n\\subsubsection{{{space.label}}}\n"

        for strategy in AggregationStrategy:
            text += generate_classification_plots(
                overall_df=overall_df,
                space=space,
                strategy=strategy,
                output_path=output_path,
                figures_path=figures_path,
                dpi=dpi,
                aggregated_data=True,
            )

    return text


def generate_classification_plots(
    overall_df: pd.DataFrame,
    space: Space,
    strategy: AggregationStrategy,
    output_path: Path,
    figures_path: Path,
    dpi: int,
    aggregated_data: bool = False,
) -> str:
    df = get_classification_table(overall_df, space, strategy)
    if df is None:
        return ""

    text = f"\n\\textbf{{{strategy.label}}}\n"

    space_strategy_identifier = f"{space.name.lower()}_{strategy.value.lower()}"

    precision_curve = plot_vs_snr(
        df,
        metrics=["soft_precision"],
        save_path=figures_path / f"snr_vs_softprec_{space_strategy_identifier}.pdf",
        metric_labels=[""],
        dpi=dpi,
        title="Precisión ($\\hat{P}$) frente a SNR",
        ylabel="Precisión",
        aggregated_data=aggregated_data,
    ).relative_to(output_path)
    text += create_figure_block(
        precision_curve,
        caption=f"Soft precision vs. SNR ({space.label} - {strategy.label})",
    )

    recall_curve = plot_vs_snr(
        df,
        metrics=["soft_recall_huang_tagare"],
        save_path=figures_path / f"snr_vs_softrec_{space_strategy_identifier}.pdf",
        metric_labels=[""],
        dpi=dpi,
        title="Sensibilidad ($\\hat{R}$) frente a SNR",
        ylabel="Sensibilidad",
        aggregated_data=aggregated_data,
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
        aggregated_data=aggregated_data,
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
        aggregated_data=aggregated_data,
    ).relative_to(output_path)
    text += create_figure_block(
        average_precision_curve,
        caption=f"Average precision and AUC vs. SNR ({space.label} - {strategy.label})",
    )

    roc_auc_curve = plot_vs_snr(
        df,
        metrics=["roc_auc"],
        save_path=figures_path / f"snr_vs_roc_auc_{space_strategy_identifier}.pdf",
        metric_labels=[""],
        dpi=dpi,
        title="ROC-AUC frente a SNR",
        ylabel="AUC",
        aggregated_data=aggregated_data,
    ).relative_to(output_path)
    text += create_figure_block(
        roc_auc_curve,
        caption=f"Average precision and AUC vs. SNR ({space.label} - {strategy.label})",
    )

    average_precision_auc_curves = plot_vs_snr(
        df,
        metrics=["ap", "roc_auc"],
        save_path=figures_path / f"snr_vs_ap_and_auc_{space_strategy_identifier}.pdf",
        metric_labels=["Average precision", "AUC-ROC"],
        dpi=dpi,
        title="Average precision and AUC vs. SNR",
        ylabel="Score",
        aggregated_data=aggregated_data,
    ).relative_to(output_path)
    text += create_figure_block(
        average_precision_auc_curves,
        caption=f"Average precision and AUC vs. SNR ({space.label} - {strategy.label})",
    )

    return text