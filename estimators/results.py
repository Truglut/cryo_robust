from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from method_comparison.domain.enums import ImageSpace


@dataclass
class WeightSet:
    """Standard container for real and Fourier estimator weights."""

    real: torch.Tensor | None = None
    fourier_real: torch.Tensor | None = None
    fourier_imag: torch.Tensor | None = None

    @classmethod
    def from_space_dict(
        cls, weights: Mapping[ImageSpace, torch.Tensor | None]
    ) -> WeightSet:
        """Create a WeightSet from the current Space-indexed dictionary format."""
        return cls(
            real=weights.get(ImageSpace.REAL),
            fourier_real=weights.get(ImageSpace.FOURIER_REAL),
            fourier_imag=weights.get(ImageSpace.FOURIER_IMAG),
        )

    @classmethod
    def shared_fourier(cls, weights: torch.Tensor | None) -> WeightSet:
        """Create Fourier weights shared by real and imaginary parts."""
        return cls(fourier_real=weights, fourier_imag=weights)

    def as_space_dict(self) -> dict[ImageSpace, torch.Tensor | None]:
        """Return weights in the current Space-indexed dictionary format."""
        return {
            ImageSpace.REAL: self.real,
            ImageSpace.FOURIER_REAL: self.fourier_real,
            ImageSpace.FOURIER_IMAG: self.fourier_imag,
        }

    def canonical_weights(self) -> torch.Tensor | None:
        """
        Return a single representative weight tensor.

        Real-space weights have priority. If only Fourier weights are available,
        the real and imaginary weights are averaged when both exist.
        """
        if self.real is not None:
            return self.real
        if self.fourier_real is not None and self.fourier_imag is not None:
            return 0.5 * (self.fourier_real + self.fourier_imag)
        return self.fourier_real if self.fourier_real is not None else self.fourier_imag

    @classmethod
    def for_irls_space(
        cls,
        space: ImageSpace,
        weights: torch.Tensor | None,
    ) -> "WeightSet":
        if space == ImageSpace.REAL:
            return cls(real=weights)
        if space == ImageSpace.FOURIER_REAL:
            return cls(fourier_real=weights)
        if space == ImageSpace.FOURIER_IMAG:
            return cls(fourier_imag=weights)
        if space == ImageSpace.FOURIER_COMPLEX:
            return cls.shared_fourier(weights)

        raise ValueError(f"Unsupported IRLS space: {space}")


@dataclass
class EstimatorResult:
    """Standard output returned by estimators."""

    average: torch.Tensor | None = None
    estimate: torch.Tensor | None = None
    weights: WeightSet = field(default_factory=WeightSet)
    converged: bool | None = None
    n_iter: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def final_weights(self) -> dict[ImageSpace, torch.Tensor | None]:
        """Compatibility alias for old code expecting Space-indexed weights."""
        return self.weights.as_space_dict()
