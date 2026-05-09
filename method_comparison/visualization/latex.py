from pathlib import Path

import pandas as pd

from method_comparison.domain.reports import EvaluationReport


def generate_latex_report(report: EvaluationReport, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "report.tex"

    with report_path.open("w") as f:
        reconstruction_df = report.reconstruction_metrics_dataframe()
        f.write(reconstruction_df.to_latex(index=False))
        
        classification_df = report.classification_metrics_dataframe()
        f.write(classification_df.to_latex(index=False))

    return
