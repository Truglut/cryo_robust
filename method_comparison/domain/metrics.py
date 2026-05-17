from dataclasses import dataclass

from method_comparison.domain.enums import Space, AggregationStrategy


@dataclass
class ReconstructionMetrics:
    """
    Global quantitative metrics evaluating the final class estimation against a ground truth.

    Parameters
    ----------
    rmse : float or None
        Root Mean Square Error between the estimated map and the ground truth.
        `None` when no ground-truth is available.
    pearson_corr : float or None
        Pearson correlation coefficient between the estimated and ground truth images.
        `None` when no ground-truth is available.
    gt_frc_resolution : float or None
        The spatial resolution (typically in angstroms or normalized spatial frequency)
        at which the Fourier Ring Correlation (FRC) curve drops below the experiment-level
        threshold (e.g., the 0.143 "gold standard" or 0.5 criterion).
        Computed by comparing the estimated images against ground truth in synthetic datasets.
        `None` when no ground-truth is available.
    hs_frc_resolution : float
        Half-set FRC resolution.
        Computed by splitting the images into two half-sets, computing their weighted
        averages (using the globally computed weights), and then comparing the two
        resulting weighted averages against each other.
    """

    rmse: float | None
    pearson_corr: float | None
    gt_frc_resolution: float | None
    hs_frc_resolution: float

    def to_record(self) -> dict:
        if self.rmse is not None:
            return {
                "rmse": self.rmse,
                "pearson_corr": self.pearson_corr,
                "gt_frc_resolution": self.gt_frc_resolution,
                "hs_frc_resolution": self.hs_frc_resolution,
            }
        else:
            return {"hs_frc_resolution": self.hs_frc_resolution}
        
    
    def print_text(self) -> str:
        s = ""
        if self.rmse is not None:
            s += f"RMSE:              {self.rmse:.4f}\n"
        if self.pearson_corr is not None:
            s += f"Correlation:       {self.pearson_corr:.4f}\n"
        if self.gt_frc_resolution is not None:
            s += f"GT FRC Resolution: {self.gt_frc_resolution:.4f}\n"
        if self.hs_frc_resolution is not None:
            s += f"HS FRC Resolution: {self.hs_frc_resolution:.4f}\n"
        return s


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
        Metrics evaluating the final reconstructed map (e.g. RMSE, FRC resolution).
        Only contains `hs_frc_resolution` for data where no ground truth is 
        available.
    space_metrics : dict of {Space: dict of {AggregationStrategy: SpaceMetrics}}
        Nested mapping `space -> aggregation_strategy -> SpaceMetrics` containing
        detailed evaluations of weight estimation.
        `None` for data where no labels are available.
    """

    reconstruction_metrics: ReconstructionMetrics
    space_metrics: dict[Space, dict[AggregationStrategy, SpaceMetrics]] | None

    def reconstruction_record(self) -> dict:
        """Serialize the global reconstruction metrics to a dictionary."""
        return self.reconstruction_metrics.to_record()

    def classification_metrics_records(self) -> list[dict] | None:
        """
        Serialize the space metrics into a flat list of dictionaries,
        suitable for conversion into a distinct pandas DataFrame.
        """
        if self.space_metrics is None:
            return None
        
        records = []
        for space, agg_dict in self.space_metrics.items():
            for agg_strategy, metrics in agg_dict.items():
                base_record = {
                    "space": space.name,
                    "aggregation_strategy": agg_strategy.value,
                }
                # Merge the space/strategy identifiers with the actual metrics
                records.append({**base_record, **metrics.to_record()})

        return records
