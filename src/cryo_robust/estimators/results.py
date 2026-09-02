from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from cryo_robust.domain import ImageSpace


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
    ) -> WeightSet:
        if space == ImageSpace.REAL:
            return cls(real=weights)
        if space == ImageSpace.FOURIER_REAL:
            return cls(fourier_real=weights)
        if space == ImageSpace.FOURIER_IMAG:
            return cls(fourier_imag=weights)
        if space == ImageSpace.FOURIER_COMPLEX:
            return cls.shared_fourier(weights)

        raise ValueError(f"Unsupported IRLS space: {space}")

    def select_space(self, space: ImageSpace):
        if space == ImageSpace.REAL:
            return self.real
        if space == ImageSpace.FOURIER_REAL:
            return self.fourier_real
        if space == ImageSpace.FOURIER_IMAG:
            return self.fourier_imag
        if space == ImageSpace.FOURIER_COMPLEX:
            # Estimators with space FOURIER_COMPLEX only produce one set of weights
            # can return either self.fourier_real or self.fourier_imag, since they
            # should be equal
            return self.fourier_real

    def subset(self, indices: torch.Tensor) -> WeightSet:
        """
        Select a subset of image weights.

        Parameters
        ----------
        indices : torch.Tensor
            Indices of the images to keep.

        Returns
        -------
        WeightSet
            WeightSet containing only the selected images.
        """
        return WeightSet(
            real=self.real[indices] if self.real is not None else None,
            fourier_real=(
                self.fourier_real[indices] if self.fourier_real is not None else None
            ),
            fourier_imag=(
                self.fourier_imag[indices] if self.fourier_imag is not None else None
            ),
        )


@dataclass
class GMMDiagnostics:
    """
    Additional information on the GMM state on a given iteration

    Attributes
    ----------
    initial_reference : torch.Tensor
        Tensor of shape ``(h, w)`` containing the initial reference used for
        this iteration.
    distances : torch.Tensor
        Tensor of shape ``(n,)``, where each element is the distance from the
        corresponding image to the initial reference of this iteration.
        These are the original distances even if the estimator then standardized
        them before fitting the GMM.
    standardized_distances: torch.Tensor
        Whether the original distances were standardized before fitting the GMM.
    component_weights : tuple[float, float]
        Weight of each of the two components of the GMM after fitting.
    means : tuple[float, float]
        Mean of each of the two components of the GMM after fitting.
    vars : tuple[float, float]
        Variance of each of the two components of the GMM after fitting.
    converged : bool
        Whether the GMM reached its tolerance for the change in reference on this
        iteration.
    weights : torch.Tensor, optional
        Tensor of shape ``(n,)`` containing the weight each image received after
        the GMM fit.
        The weights are computed as the posterior probability of the image belonging
        to the GMM component with a lower mean distance, given the distance of said
        image to the initial reference. This means the ``weights`` tensor can be
        computed from the information in ``distances``, ``component_weights``,
        ``means`` and ``stds``.
        Default is None.
    weighted_average : torch.Tensor, optional
        Tensor of shape ``(h, w)`` containing the average of all the images weighted
        according to ``weights``. This would be the output of the estimator on the
        present iteration.
    """

    initial_reference: torch.Tensor
    distances: torch.Tensor
    standardized_distances: bool
    component_weights: tuple[float, float]
    means: tuple[float, float]
    vars: tuple[float, float]
    converged: bool
    weights: torch.Tensor | None = None
    weighted_average: torch.Tensor | None = None


@dataclass
class EstimatorResult:
    """Standard output returned by estimators."""

    estimate: torch.Tensor
    average: torch.Tensor | None = None
    weights: WeightSet = field(default_factory=WeightSet)
    gmm_diagnostics: GMMDiagnostics | None = None
