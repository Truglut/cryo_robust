from dataclasses import dataclass

from method_comparison.domain.enums import Space, AggregationStrategy


@dataclass
class ReconstructionMetrics:
    """
    Global quantitative metrics evaluating the final class estimation against a ground truth.

    Parameters
    ----------
    rmse : float
        Root Mean Square Error between the estimated map and the ground truth.
    pearson_corr : float
        Pearson correlation coefficient between the estimated and ground truth images.
    fsc_resolution : float
        The spatial resolution (typically in angstroms or normalized spatial frequency) 
        at which the Fourier Shell Correlation (FSC) curve drops below the experiment-level 
        threshold (e.g., the 0.143 "gold standard" or 0.5 criterion).
    """
    rmse: float
    pearson_corr: float
    fsc_resolution: float

    def to_record(self) -> dict:
        return {
            "rmse": self.rmse,
            "pearson_corr": self.pearson_corr,
            "fsc_resolution": self.fsc_resolution,
        }
    

@dataclass
class SpaceMetrics:
    """
    Metrics evaluating the per-image weight estimation or classification for a
    specific space and aggregation strategy.

    Parameters
    ----------
    ap : float
        Average Precision of the estimated weights/scores.
    soft_precision : float
        Soft precision metric evaluating the correctness of assigned weights.
    soft_recall : dict of {str: float}
        Dictionary mapping specific method names to their corresponding
        soft recall values.
    """
    ap: float
    soft_precision: float
    soft_recall: dict[str, float]

    def to_record(self) -> dict:
        """
        Serializes the metrics, dynamically flattening the soft_recall dictionary
        so it can be easily ingested by pandas.
        """
        record = {
            "ap": self.ap,
            "soft_precision": self.soft_precision,
        }
        for method, value in self.soft_recall.items():
            record[f"soft_recall_{method}"] = value
        
        return record


@dataclass
class MethodMetrics:
    """
    Quantitative metrics for a single estimation method, covering both
    global reconstruction quality and per-image evaluation in different spaces.

    Parameters
    ----------
    reconstruction_metrics : ReconstructionMetrics
        Metrics evaluating the final reconstructed map (e.g. RMSE, FSC resolution).
    space_metrics : dict of {Space: dict of {AggregationStrategy: SpaceMetrics}}
        Nested mapping `space -> aggregation_strategy -> SpaceMetrics` containing
        detailed evaluations of weight estimation.
    """

    reconstruction_metrics: ReconstructionMetrics
    space_metrics: dict[Space, dict[AggregationStrategy, SpaceMetrics]]

    def reconstruction_record(self) -> dict:
        """Serialize the global reconstruction metrics to a dictionary."""
        return self.reconstruction_metrics.to_record()

    def classification_metrics_records(self) -> list[dict]:
        """
        Serialize the space metrics into a flat list of dictionaries,
        suitable for conversion into a distinct pandas DataFrame.
        """
        records = []
        for space, agg_dict in self.space_metrics.items():
            for agg_strategy, metrics in agg_dict.items():
                base_record = {
                    "space": space.name,
                    "aggregation_strategy": agg_strategy.value
                }
                # Merge the space/strategy identifiers with the actual metrics
                records.append({**base_record, **metrics.to_record()})
                
        return records
