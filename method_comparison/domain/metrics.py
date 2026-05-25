from dataclasses import dataclass

from method_comparison.domain.enums import Space, AggregationStrategy
from method_comparison.evaluation.frc import FRCThreshold


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
    gt_frc_resolutions : float or None
        The spatial resolution (typically in angstroms or normalized spatial frequency)
        at which the Fourier Ring Correlation (FRC) curve drops below the experiment-level
        threshold (e.g., the 0.143 "gold standard" or 0.5 criterion).
        Computed by comparing the estimated images against ground truth in synthetic datasets.
        `None` when no ground-truth is available.
    hs_frc_resolutions : float
        Half-set FRC resolution.
        Computed by splitting the images into two half-sets, computing their weighted
        averages (using the globally computed weights), and then comparing the two
        resulting weighted averages against each other.
    gt_aufrc: float or None
        Area under the ground-truth FRC curve.
    hs_aufrc: float
        Area under the half-set FRC curve.
    """

    rmse: float | None
    pearson_corr: float | None
    gt_frc_resolutions: dict[FRCThreshold, float] | None
    hs_frc_resolutions: dict[FRCThreshold, float]
    gt_aufrc: float | None
    hs_aufrc: float

    def to_record(self) -> dict:
        record = {}
        if self.rmse is not None:
            record["rmse"] = self.rmse
        if self.pearson_corr is not None:
            record["pearson_corr"] = self.pearson_corr
        if self.gt_frc_resolutions is not None:
            for threshold, value in self.gt_frc_resolutions.items():
                record["GT Resolution" + f"({threshold.value})"] = value
        if self.hs_frc_resolutions is not None:
            for threshold, value in self.hs_frc_resolutions.items():
                record["HS Resolution" + f"({threshold.value})"] = value
        if self.gt_aufrc is not None:
            record["AUFRC (GT)"] = self.gt_aufrc
        if self.hs_aufrc is not None:
            record["AUFRC (HS)"] = self.hs_aufrc
        return record

    def print_text(self) -> str:
        s = ""
        if self.rmse is not None:
            s += f"RMSE:              {self.rmse:.4f}\n"
        if self.pearson_corr is not None:
            s += f"Correlation:       {self.pearson_corr:.4f}\n"
        if self.gt_frc_resolutions is not None:
            s += f"GT FRC Resolution:\n"
            for threshold, value in self.gt_frc_resolutions.items():
                s += f"\t{threshold}: {value:.4f}\n"
        if self.hs_frc_resolutions is not None:
            s += f"HS FRC Resolution:\n"
            for threshold, value in self.hs_frc_resolutions.items():
                s += f"\t{threshold}: {value:.4f}\n"
        if self.gt_aufrc is not None:
            s += f"AUFRC (GT): {self.gt_aufrc:.4f}\n"
        if self.hs_aufrc is not None:
            s += f"AUFRC (HS): {self.hs_aufrc:.4f}\n"
        return s


@dataclass
class ClassificationMetrics:
    """
    Metrics evaluating the per-image weight estimation or classification for a
    specific space and aggregation strategy.

    Parameters
    ----------
    ap : float
        Average Precision of the estimated weights/scores.
    roc_auc: float
        ROC-AUC of the estimated weights/scores.
    soft_precision : float
        Soft precision metric evaluating the correctness of assigned weights.
    soft_recall : dict of {str: float}
        Dictionary mapping specific method names to their corresponding
        soft recall values.
    """

    ap: float
    roc_auc: float
    soft_precision: float
    soft_recall: dict[str, float]

    def to_record(self) -> dict:
        """
        Serializes the metrics, dynamically flattening the soft_recall dictionary
        so it can be easily ingested by pandas.
        """
        record = {
            "ap": self.ap,
            "roc_auc": self.roc_auc,
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
    space_metrics: dict[Space, dict[AggregationStrategy, ClassificationMetrics]] | None

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
