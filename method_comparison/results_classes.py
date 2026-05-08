from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.space import Space


@dataclass
class MethodMetrics:
    """
    Quantitative metrics for a single estimation method.

    Parameters
    ----------
    rmse : float
        Root mean squared error between the estimated and ground truth images.
    pearson_corr : float
        Pearson correlation between the estimated and ground truth images.
    fsc_resolution : float
        Normalised spatial frequency at which the FSC curve drops below the
        experiment-level threshold.
    space_metrics : dict of {Space: dict of {str: dict}}
        Nested mapping `space -> aggregation_strategy -> metric_dict`, where
        each `metric_dict` contains keys such as `"ap"`,
        `"soft_precision"`, and `"soft_recall_<method>"`.
    """

    rmse: float
    pearson_corr: float
    fsc_resolution: float
    # space -> agg_strategy -> metric
    space_metrics: dict[Space, dict[str, dict[str, float]]]


    def to_record(self) -> dict:
        return {
            "rmse": self.rmse,
            "pearson_corr": self.pearson_corr,
            "fsc_resolution": self.fsc_resolution
        }


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
    fsc_data : tuple of (np.ndarray, np.ndarray) or None
        `(freqs, fsc_curve)` arrays returned by `compute_fsc`.
        `None` when no ground truth is available.
    estimated_img : np.ndarray
        The reconstructed average image produced by this method.
    """

    name: str
    metrics: MethodMetrics | None
    scores: dict[Space, np.ndarray]
    fsc_data: tuple[np.ndarray, np.ndarray] | None
    estimated_img: np.ndarray


    def metrics_record(self) -> dict:
        base = {"method": self.name}

        if self.metrics is None:
            return base
        
        return {
            **base,
            **self.metrics.to_record()
        }


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
    fsc_threshold: float | None


    def metrics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            mr.metrics_record()
            for mr in self.method_results
        )