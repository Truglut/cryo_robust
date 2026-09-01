from cryo_robust.comparison.domain.reports import EvaluationReport
from cryo_robust.comparison.domain.metrics import ReconstructionMetrics


def format_reconstruction_metrics(metrics: ReconstructionMetrics) -> str:
    s = ""
    if metrics.rmse is not None:
        s += f"RMSE:              {metrics.rmse:.4f}\n"
    if metrics.pearson_corr is not None:
        s += f"Correlation:       {metrics.pearson_corr:.4f}\n"
    if metrics.gt_frc_resolutions is not None:
        s += f"GT FRC Resolution:\n"
        for threshold, value in metrics.gt_frc_resolutions.items():
            s += f"\t{threshold}: {value:.4f}\n"
    if metrics.hs_frc_resolutions is not None:
        s += f"HS FRC Resolution:\n"
        for threshold, value in metrics.hs_frc_resolutions.items():
            s += f"\t{threshold}: {value:.4f}\n"
    if metrics.gt_aufrc is not None:
        s += f"AUFRC (GT): {metrics.gt_aufrc:.4f}\n"
    if metrics.hs_aufrc is not None:
        s += f"AUFRC (HS): {metrics.hs_aufrc:.4f}\n"
    return s


def print_report(report: EvaluationReport) -> None:
    """
    Print a structured summary of an `EvaluationReport`.

    For each method, the output contains available reconstruction metrics (RMSE, Pearson
    correlation, ground truth FRC resolution and half-set FRC resolution)
    followed by outlier-rejection metrics (average precision, soft precision,
    soft recall) broken down by weight space and aggregation strategy.

    Parameters
    ----------
    report : EvaluationReport
        Populated report produced by `compute_metrics`.

    Returns
    -------
    None
    """
    separator = "-" * 25
    print(f"\n{separator} EVALUATION RESULTS {separator}\n")

    for method_result in report.method_results:
        print(f"--- {method_result.name.upper()} ---")

        metrics = method_result.metrics
        if metrics is None:
            print("  No metrics available.\n")
            continue

        if metrics.reconstruction_metrics is not None:
            print(format_reconstruction_metrics(metrics.reconstruction_metrics))

        if metrics.space_metrics is not None:
            for space, strategy_metrics in metrics.space_metrics.items():
                for strategy, metrics in strategy_metrics.items():
                    print(f"  Space: {space.name}  |  Aggregation: {strategy}")
                    print(f"    Avg Precision:   {metrics.ap:.4f}")
                    print(f"    ROC-AUC:         {metrics.roc_auc:.4f}")
                    print(f"    Soft Precision:  {metrics.soft_precision:.4f}")
                    for recall_method, value in metrics.soft_recall.items():
                        print(f"    Soft Recall ({recall_method}): {value:.4f}")

        print()
