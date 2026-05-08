from dataclasses import dataclass
from method_comparison.domain.enums import Space


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