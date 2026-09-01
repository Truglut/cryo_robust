import torch

from .data import ImageBatch
from .results import EstimatorResult, WeightSet


class Estimator:
    def __init__(self, device: torch.device | str):
        self.device = device

    def fit(self, images: ImageBatch) -> EstimatorResult:
        raise NotImplementedError("Subclasses must implement the fit method.")

    def reconstruct_from_weights(
        self,
        images: ImageBatch,
        weights: WeightSet,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement the reconstruct from weights method"
        )
