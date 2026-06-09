from __future__ import annotations
from enum import Enum

from method_comparison.domain.enums import Space

class IRLSSpace(str, Enum):
    REAL = "real"
    FOURIER_REAL = "fourier_real"
    FOURIER_IMAG = "fourier_imag"
    FOURIER_COMPLEX = "fourier_complex"

    @classmethod
    def from_space(cls, space: Space) -> IRLSSpace:
        if space == Space.REAL:
            return cls.REAL
        if space == Space.FOURIER_REAL:
            return cls.FOURIER_REAL
        if space == Space.FOURIER_IMAG:
            return cls.FOURIER_IMAG
        raise ValueError(f"Cannot convert Space value to IRLSSpace: {space}")
    

def normalize_irls_space(space: IRLSSpace | Space | str) -> IRLSSpace:
    """Normalize strings, Space values and IRLSSpace values to IRLSSpace."""
    if isinstance(space, IRLSSpace):
        return space
    if isinstance(space, Space):
        return IRLSSpace.from_space(space)
    return IRLSSpace(space)