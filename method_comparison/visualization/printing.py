from method_comparison.domain.reports import EvaluationReport

def print_report(report: EvaluationReport) -> None:
    """
    Print a structured summary of an `EvaluationReport`.

    For each method, the output contains reconstruction metrics (RMSE, Pearson
    correlation, FSC resolution) followed by outlier-rejection metrics
    (average precision, soft precision, soft recall) broken down by weight
    space and aggregation strategy.

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

        m = method_result.metrics
        if m is None:
            print("  No ground-truth metrics available.\n")
            continue
        
        reconstruction_m = m.reconstruction_metrics
        print(
            f"  RMSE: {reconstruction_m.rmse:.4f} | "
            f"Pearson: {reconstruction_m.pearson_corr:.4f} | "
            f"FSC Resolution ({report.fsc_threshold}): {reconstruction_m.fsc_resolution:.4f}"
        )

        for space, strategy_metrics in m.space_metrics.items():
            for strategy, metrics in strategy_metrics.items():
                print(f"  Space: {space.name}  |  Aggregation: {strategy}")
                print(f"    Avg Precision:   {metrics.ap:.4f}")
                print(f"    Soft Precision:  {metrics.soft_precision:.4f}")
                for recall_method, value in metrics.soft_recall.items():
                    print(f"    Soft Recall ({recall_method}): {value:.4f}")

        print()
