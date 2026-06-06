from dataclasses import dataclass

import numpy as np
import pandas as pd

from method_comparison.domain.enums import Space
from method_comparison.domain.metrics import MethodMetrics, ClassificationMetrics
from method_comparison.evaluation.frc import FRCData, FRCThreshold

ID_COLS = ["method", "space", "aggregation_strategy", "run"]


@dataclass
class MethodResults:
    """
    All outputs produced for a single estimation method.

    Parameters
    ----------
    name : str
        Human-readable method identifier.
    metrics : MethodMetrics or None
        Quantitative metrics. `None` for unlabeled / real-data runs where
        no ground truth is available.
    scores : dict of {Space: dict of {str: np.ndarray}}
        Aggregated per-image scalar weights, keyed by space then aggregation
        strategy.  Shape of each array is `(n_images,)`.  Used directly by
        plotting and report-generation code.
    ground_truth_frc_data : FRCData or None
        Computed by comparing the estimated class average against ground truth.
        `FRCData` returned by `compute_frc`.
        `None` when no ground truth is available.
    half_set_frc_data : FRCData
        Computed by comparing the weighted averages resulting from splitting the
        images into two half-sets.
        Returned by `compute_frc`.
    estimated_img : np.ndarray
        The reconstructed average image produced by this method.
    fourier_ring_metrics: dict[Space, dict[int, ClassificationMetrics]]
        Dict mapping Space.FOURIER_REAL and Space.FOURIER_IMAG to a dict that maps
        each integer key to the classification metrics obtained using the weights
        in the corresponding Fourier ring.
    """

    name: str
    metrics: MethodMetrics | None
    scores: dict[Space, np.ndarray]
    ground_truth_frc_data: FRCData | None
    half_set_frc_data: FRCData
    estimated_img: np.ndarray
    fourier_ring_metrics: dict[Space, dict[int, ClassificationMetrics]]

    def reconstruction_metrics_record(self) -> dict:
        if self.metrics is None:
            return {}

        return {"method": self.name, **self.metrics.reconstruction_record()}

    def classification_metrics_records(self) -> list[dict]:
        """
        Retrieves flattened space metrics and injects the method name
        into each record.
        """
        if self.metrics is None:
            return []

        # Get the flat records from the metrics dataclass
        records = self.metrics.classification_metrics_records()

        # Inject the method name into every row
        return [{"method": self.name, **record} for record in records]


@dataclass
class EvaluationReport:
    """
    Container for the full evaluation output across all methods.

    Parameters
    ----------
    method_results : list of MethodResults
        One entry per estimation method, in the order they were evaluated.
    labels : np.ndarray or None
        Per-image ground-truth class labels.  `None` for unlabeled data.
    frc_thresholds : list[FRCThreshold]
        The FRC thresholds used to define resolution (e.g. 0.143 or 0.5).
    """

    method_results: list[MethodResults]
    labels: np.ndarray | None
    frc_thresholds: list[FRCThreshold]

    def reconstruction_metrics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            mr.reconstruction_metrics_record() for mr in self.method_results
        )

    def classification_metrics_dataframe(self) -> pd.DataFrame:
        """
        Builds a global pandas DataFrame containing all space metrics
        for all evaluated methods in a long format.
        """
        all_records = []
        for mr in self.method_results:
            all_records.extend(mr.classification_metrics_records())

        return pd.DataFrame(all_records)


@dataclass
class EvaluationStudy:
    reports: list[EvaluationReport]

    def reconstruction_metrics_dataframe(self) -> pd.DataFrame:
        if not self.reports:
            return pd.DataFrame()
        
        # Iterate over reports and build overall dataframe by concatenating
        dfs = []
        for run_idx, report in enumerate(self.reports):
            df = report.reconstruction_metrics_dataframe()
            df["run"] = run_idx
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def classification_metrics_dataframe(self) -> pd.DataFrame:
        if not self.reports:
            return pd.DataFrame()
        
        # Iterate over reports and build overall dataframe by concatenating
        dfs = []
        for run_idx, report in enumerate(self.reports):
            df = report.classification_metrics_dataframe()
            df["run"] = run_idx
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def _metric_columns(df: pd.DataFrame) -> pd.Index:
        """Selects the columns that contain metrics from a dataframe"""
        numeric_cols = df.select_dtypes(include="number").columns
        return pd.Index([c for c in numeric_cols if c not in ID_COLS])

    @staticmethod
    def _aggregate(
        df: pd.DataFrame,
        groupby: str | list[str],
    ) -> pd.DataFrame:
        """
        Aggregates a dataframe grouping by certain columns, calculating mean
        and std for each metric.
        """
        metric_cols = EvaluationStudy._metric_columns(df)

        grouped = df.groupby(groupby)

        summary = grouped[metric_cols].agg(["mean", "std"])
        summary["n"] = grouped.size()

        summary = summary.reset_index()
        
        # Flatten multiindex
        summary.columns = [
            "_".join(col).rstrip("_") if isinstance(col, tuple) else col
            for col in summary.columns
        ]

        return summary

    def aggregate_reconstruction_metrics(self) -> pd.DataFrame:
        df = self.reconstruction_metrics_dataframe()

        return EvaluationStudy._aggregate(df, groupby="method")

    def aggregate_classification_metrics(self) -> pd.DataFrame:
        df = self.classification_metrics_dataframe()

        return EvaluationStudy._aggregate(
            df, groupby=["method", "space", "aggregation_strategy"]
        )
