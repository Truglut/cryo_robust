from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from estimators.spaces import IRLSSpace

from method_comparison.domain.enums import Space

ArrayLike = torch.Tensor | np.ndarray


def to_tensor(
    x: ArrayLike | None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """
    Convert an array-like object to a torch tensor.

    ``None`` is returned unchanged. If ``dtype`` is not provided, floating-point
    tensors are converted to float32 and complex tensors to complex64.
    """
    if x is None:
        return None

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if dtype is None:
        if torch.is_complex(x):
            dtype = torch.complex64
        elif torch.is_floating_point(x):
            dtype = torch.float32

    return x.to(device=device, dtype=dtype)


@dataclass
class FourierVariance:
    """
    Fourier-space variance information.

    ``real`` and ``imag`` are component-wise variances, used by separate
    real/imaginary Fourier estimators.

    ``modulus`` is a per-frequency variance scale for complex residual moduli,
    used by joint complex Fourier estimators.
    """

    real: torch.Tensor | None = None
    imag: torch.Tensor | None = None
    modulus: torch.Tensor | None = None
    eps: float = 1.0e-8

    @classmethod
    def estimate(cls, fourier: torch.Tensor, *, eps: float = 1.0e-8) -> FourierVariance:
        """Estimate component and modulus variances from complex Fourier images."""
        if not torch.is_complex(fourier):
            raise TypeError("Expected complex Fourier images.")

        centered = fourier - fourier.mean(dim=0)
        return cls(
            real=fourier.real.var(dim=0, unbiased=False),
            imag=fourier.imag.var(dim=0, unbiased=False),
            modulus=centered.abs().square().mean(dim=0),
            eps=eps,
        )

    @classmethod
    def from_components(
        cls,
        real: torch.Tensor,
        imag: torch.Tensor,
        *,
        modulus: torch.Tensor | None = None,
        eps: float = 1.0e-8,
    ) -> FourierVariance:
        """Create component variances, using real + imag as default modulus variance."""
        if modulus is None:
            modulus = real + imag
        return cls(real=real, imag=imag, modulus=modulus, eps=eps)

    def components(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real and imaginary component variances."""
        if self.real is None or self.imag is None:
            raise ValueError("Fourier real/imaginary variances are not available.")
        return self.real, self.imag

    def component_stds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real and imaginary component standard deviations."""
        real, imag = self.components()
        return real.sqrt().clamp_min(self.eps), imag.sqrt().clamp_min(self.eps)

    def modulus_variance(self) -> torch.Tensor:
        """Return complex-modulus variance."""
        if self.modulus is None:
            raise ValueError("Fourier modulus variance is not available.")
        return self.modulus

    def modulus_std(self) -> torch.Tensor:
        """Return complex-modulus standard deviation."""
        return self.modulus_variance().sqrt().clamp_min(self.eps)


@dataclass
class ImageBatch:
    """
    Canonical image container for estimator inputs.

    The preferred internal representation is:

    - real-space images as ``batch.real``;
    - Fourier-space images as one complex tensor, ``batch.fourier``;
    - Fourier variance as ``FourierVariance``.
    """

    real: torch.Tensor | None = None
    fourier: torch.Tensor | None = None
    ctf: torch.Tensor | None = None
    real_variance_value: torch.Tensor | None = None
    fourier_variance_value: FourierVariance | None = None
    real_shape: tuple[int, int] | None = None
    norm: str = "ortho"
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        """Ensures provided data is sufficient and performs basic checks"""
        if self.real is None and self.fourier is None:
            raise ValueError("ImageBatch requires real or Fourier images.")

        if self.fourier is not None and not torch.is_complex(self.fourier):
            raise TypeError("Fourier images must be stored as a complex tensor.")

        if self.real_shape is None and self.real is not None:
            self.real_shape = tuple(self.real.shape[1:])

    @classmethod
    def from_real(
        cls,
        images: ArrayLike,
        *,
        variance: ArrayLike | None = None,
        ctf: ArrayLike | None = None,
        device: torch.device | str | None = None,
    ) -> ImageBatch:
        """Create an ImageBatch from real-space images."""
        return cls(
            real=to_tensor(images, device=device, dtype=torch.float32),
            real_variance_value=to_tensor(variance, device=device, dtype=torch.float32),
            ctf=to_tensor(ctf, device=device, dtype=torch.float32),
        )

    @classmethod
    def from_fourier(
        cls,
        fourier: ArrayLike,
        *,
        variance: FourierVariance | torch.Tensor | np.ndarray | None = None,
        ctf: ArrayLike | None = None,
        real_shape: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
    ) -> ImageBatch:
        """Create an ImageBatch from complex Fourier-space images."""
        fourier = to_tensor(fourier, device=device, dtype=torch.complex64)

        if isinstance(variance, FourierVariance):
            fourier_variance = variance
        else:
            variance = to_tensor(variance, device=device, dtype=torch.float32)
            fourier_variance = (
                FourierVariance(modulus=variance) if variance is not None else None
            )

        return cls(
            fourier=fourier,
            fourier_variance_value=fourier_variance,
            ctf=to_tensor(ctf, device=device, dtype=torch.float32),
            real_shape=real_shape,
        )

    @classmethod
    def from_space_dict(
        cls,
        images: Mapping[Space, ArrayLike],
        *,
        variance: Mapping[Space, ArrayLike] | None = None,
        ctf: ArrayLike | None = None,
        device: torch.device | str | None = None,
    ) -> ImageBatch:
        """Create an ImageBatch from the current Space-indexed dictionary format."""
        real = to_tensor(images.get(Space.REAL), device=device, dtype=torch.float32)

        fourier = None
        if Space.FOURIER_REAL in images and Space.FOURIER_IMAG in images:
            fourier = torch.complex(
                to_tensor(images[Space.FOURIER_REAL], device=device, dtype=torch.float32),
                to_tensor(images[Space.FOURIER_IMAG], device=device, dtype=torch.float32),
            )

        real_var = None
        fourier_var = None
        if variance is not None:
            real_var = to_tensor(variance.get(Space.REAL), device=device, dtype=torch.float32)

            if Space.FOURIER_REAL in variance and Space.FOURIER_IMAG in variance:
                fourier_var = FourierVariance.from_components(
                    to_tensor(variance[Space.FOURIER_REAL], device=device, dtype=torch.float32),
                    to_tensor(variance[Space.FOURIER_IMAG], device=device, dtype=torch.float32),
                )

        return cls(
            real=real,
            fourier=fourier,
            ctf=to_tensor(ctf, device=device, dtype=torch.float32),
            real_variance_value=real_var,
            fourier_variance_value=fourier_var,
        )
    
    @property
    def n_images(self) -> int:
        """Number of images in the batch."""
        return self.ensure_real().shape[0] if self.real is not None else self.ensure_fourier().shape[0]

    @property
    def device(self) -> torch.device:
        """Device where the image tensors are stored."""
        return self.ensure_real().device if self.real is not None else self.ensure_fourier().device

    def ensure_real(self) -> torch.Tensor:
        """Return real-space images, computing them from Fourier images if needed."""
        if self.real is None:
            self.real = torch.fft.irfft2(self.fourier, s=self.real_shape, norm=self.norm)
        return self.real

    def ensure_fourier(self) -> torch.Tensor:
        """Return complex Fourier-space images, computing them from real images if needed."""
        if self.fourier is None:
            self.fourier = torch.fft.rfft2(self.real, norm=self.norm)
        return self.fourier
    
    def fourier_real(self) -> torch.Tensor:
        """Return real part of Fourier images."""
        return self.ensure_fourier().real

    def fourier_imag(self) -> torch.Tensor:
        """Return imaginary part of Fourier images."""
        return self.ensure_fourier().imag

    def real_variance(self) -> torch.Tensor:
        """Return or estimate real-space variance."""
        if self.real_variance_value is None:
            self.real_variance_value = self.ensure_real().var(dim=0, unbiased=False)
        return self.real_variance_value

    def real_std(self) -> torch.Tensor:
        """Return real-space standard deviation."""
        return self.real_variance().sqrt().clamp_min(self.eps)

    def fourier_variance(self) -> FourierVariance:
        """Return or estimate FourierVariance."""
        if self.fourier_variance_value is None:
            self.fourier_variance_value = FourierVariance.estimate(
                self.ensure_fourier(),
                eps=self.eps,
            )
        return self.fourier_variance_value
    
    def fourier_component_variances(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Fourier real/imaginary component variances."""
        return self.fourier_variance().components()

    def fourier_component_stds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Fourier real/imaginary component standard deviations."""
        return self.fourier_variance().component_stds()

    def fourier_modulus_variance(self) -> torch.Tensor:
        """Return Fourier modulus variance."""
        fv = self.fourier_variance()
        if fv.modulus is None and fv.real is not None and fv.imag is not None:
            fv.modulus = fv.real + fv.imag
        return fv.modulus_variance()

    def fourier_modulus_std(self) -> torch.Tensor:
        """Return Fourier modulus standard deviation."""
        return self.fourier_modulus_variance().sqrt().clamp_min(self.eps)
    
    def as_space_dict(self) -> dict[Space, torch.Tensor]:
        """Return images in the current Space-indexed dictionary format."""
        fourier = self.ensure_fourier()
        return {
            Space.REAL: self.ensure_real(),
            Space.FOURIER_REAL: fourier.real,
            Space.FOURIER_IMAG: fourier.imag,
        }

    def variance_space_dict(self) -> dict[Space, torch.Tensor]:
        """Return variances in the current Space-indexed dictionary format."""
        fourier_real_var, fourier_imag_var = self.fourier_component_variances()
        return {
            Space.REAL: self.real_variance(),
            Space.FOURIER_REAL: fourier_real_var,
            Space.FOURIER_IMAG: fourier_imag_var,
        }

    def std_space_dict(self) -> dict[Space, torch.Tensor]:
        """Return standard deviations in the current Space-indexed dictionary format."""
        fourier_real_std, fourier_imag_std = self.fourier_component_stds()
        return {
            Space.REAL: self.real_std(),
            Space.FOURIER_REAL: fourier_real_std,
            Space.FOURIER_IMAG: fourier_imag_std,
        }
    
    def select_space_data(
        self,
        irls_space: IRLSSpace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Selects the data from the batch that corresponds to the requested space

        Parameters
        ----------
        irls_space : IRLSSpace
            Space the IRLS estimator will operate on

        Returns
        -------
        torch.Tensor
            The batch of images in the appropriate space representation
        torch.Tensor
            Image variance tensor
        torch.Tensor
            Image std tensor
        torch.Tensor | None
            CTF tensor, or None if the images have no CTF.

        Raises
        ------
        ValueError
            If ``irls_space`` takes an unknown value.
        """
        if irls_space == IRLSSpace.REAL:
            return (
                self.ensure_real(),
                self.real_variance(),
                self.real_std(),
                None,
            )

        if irls_space == IRLSSpace.FOURIER_REAL:
            var_real, _ = self.fourier_component_variances()
            std_real, _ = self.fourier_component_stds()
            return (
                self.fourier_real(),
                var_real,
                std_real,
                self.ctf,
            )

        if irls_space == IRLSSpace.FOURIER_IMAG:
            _, var_imag = self.fourier_component_variances()
            _, std_imag = self.fourier_component_stds()
            return (
                self.fourier_imag(),
                var_imag,
                std_imag,
                self.ctf,
            )

        if irls_space == IRLSSpace.FOURIER_COMPLEX:
            return (
                self.ensure_fourier(),
                self.fourier_modulus_variance(),
                self.fourier_modulus_std(),
                self.ctf,
            )

        raise ValueError(f"Unsupported IRLS space: {irls_space}")