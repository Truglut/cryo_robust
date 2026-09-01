import torch

from cryo_robust.domain import ImageSpace

from .data import ImageBatch
from .results import EstimatorResult, WeightSet


class Estimator:
    max_iter: int

    def fit(self, images: ImageBatch) -> EstimatorResult:
        raise NotImplementedError("Subclasses must implement the fit method.")

    def reconstruct_from_weights(
        self, images: ImageBatch, weights: WeightSet, space: ImageSpace
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement the reconstruct from weights method"
        )
