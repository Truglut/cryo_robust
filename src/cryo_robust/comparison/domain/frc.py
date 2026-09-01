from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FRCThreshold(str, Enum):
    ONE_OVER_SEVEN = "0.143"
    ONE_HALF = "0.5"
    HALF_BIT = "half-bit"


@dataclass
class FRCData:
    """
    Fourier Ring Correlation curve data.

    Parameters
    ----------
    freqs: np.ndarray
        1D array of the spatial frequencies FRC was computed at.
        Expressed in inverse distance units (e.g. 1/Å).
    frc: np.ndarray
        1D array containing the FRC values at the specified resolutions/frequencies.
    pixel_size : float
        Physical pixel size of the input image data, typically in Å/pixel
        or another spatial unit per pixel.
    box_size : int
        Size of the square image region used for the FRC computation,
        in pixels.
    resolutions : dict, optional
        Dictionary containing estimated resolution values derived from the
        FRC curve, keyed by threshold criterion: "0.143", "0.5" or "half-bit".
        Defaults to an empty dictionary.
    Attributes
    ----------
    spatial_resolutions : np.ndarray
        Spatial resolutions corresponding to ``freqs``. Computed as
        ``1 / freqs`` for nonzero frequencies. The zero-frequency entry
        is set to ``np.inf``.
    """

    freqs: np.ndarray  # spatial frequency [1/Å], shape (n_rings,)
    frc: np.ndarray  # FRC values,         shape (n_rings,)
    n_pixels: np.ndarray  # pixels per ring,    shape (n_rings,) — needed for 1/2-bit
    pixel_size: float
    box_size: int
    resolutions: dict = field(default_factory=dict)

    @property
    def spatial_resolutions(self) -> np.ndarray:
        spatial_resolutions = np.zeros_like(self.freqs, dtype=float)
        spatial_resolutions[0] = np.inf
        spatial_resolutions[1:] = 1.0 / self.freqs[1:]
        return spatial_resolutions
