from cryo_robust.comparison.domain.reports import EvaluationReport


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
            print(metrics.reconstruction_metrics.print_text())

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
