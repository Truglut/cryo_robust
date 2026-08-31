from dataclasses import dataclass

import torch

from cryo_robust.estimators.base import Estimator
from cryo_robust.estimators.results import EstimatorResult


@dataclass
class MethodRun:
    estimator: Estimator | None
    result: EstimatorResult
    initial_reference: torch.Tensor | None = None
