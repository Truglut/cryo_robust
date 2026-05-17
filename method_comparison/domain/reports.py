from dataclasses import dataclass

import numpy as np
import pandas as pd

from method_comparison.domain.enums import Space
from method_comparison.domain.metrics import MethodMetrics


@dataclass
class FRCData:
    """
    Fourier Ring Correlation curve data.

    Parameters
    ----------
    resolutions : np.ndarray
        1D array of the spatial resolutions FRC was computed at.
    freqs: np.ndarray
        1D array of the spatial frequencies FRC was computed at.
    frc: np.ndarray
        1D array containing the FRC values at the specified resolutions/frequencies.
    """

    resolutions: np.ndarray
    freqs: np.ndarray
    frc: np.ndarray


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
    """

    name: str
    metrics: MethodMetrics | None
    scores: dict[Space, np.ndarray]
    ground_truth_frc_data: FRCData | None
    half_set_frc_data: FRCData
    estimated_img: np.ndarray

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
    fsc_threshold : float
        The FSC threshold used to define resolution (e.g. 0.143 or 0.5).
        `None` for unlabeled data.
    """

    method_results: list[MethodResults]
    labels: np.ndarray | None
    frc_threshold: float | None

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
