from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
from scipy import integrate, interpolate
from scipy.optimize import brentq


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


def _tukey_2d(shape: tuple[int, int], alpha: float = 0.15) -> np.ndarray:
    """
    2-D Tukey (tapered cosine) window.

    Multiplying by this before FFT suppresses edge discontinuities that would
    otherwise leak power across all Fourier rings.  alpha controls the fraction
    of the radius occupied by the cosine taper (0 = rectangular, 1 = Hann).
    alpha=0.15 is the conventional choice for cryoEM FRC.
    """

    def tukey_1d(n: int) -> np.ndarray:
        t = np.linspace(0, 1, n)
        w = np.ones(n)
        width = alpha / 2.0
        # Leading taper
        mask_left = t < width
        w[mask_left] = 0.5 * (1 - np.cos(np.pi * t[mask_left] / width))
        # Trailing taper
        mask_right = t > (1 - width)
        w[mask_right] = 0.5 * (1 - np.cos(np.pi * (1 - t[mask_right]) / width))
        return w

    wy = tukey_1d(shape[0])
    wx = tukey_1d(shape[1])
    return np.outer(wy, wx)


# Core FRC computation
def compute_frc(
    image1: np.ndarray,
    image2: np.ndarray,
    pixel_size: float = 1.0,
    apodize: bool = True,
    apodization_alpha: float = 0.15,
    nyquist_fraction: float = 0.95,
) -> FRCData:
    """
    Compute the Fourier Ring Correlation between two 2-D images.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Input images — must have identical shape.  Typically the averages of
        two random half-sets drawn from the same class.
    pixel_size : float
        Sampling rate in Å/pixel.  Determines the physical frequency axis.
    apodize : bool
        Apply a 2-D Tukey window before FFT.  Strongly recommended; disabling
        it produces artefactual oscillations from the sharp image boundary.
    apodization_alpha : float
        Taper fraction for the Tukey window (0.15 is the cryoEM convention).
    nyquist_fraction : float
        Discard rings beyond this fraction of the Nyquist radius.  The outermost
        rings are unreliable (few pixels, aliasing) — 0.95 is conservative but safe.

    Returns
    -------
    FRCData
        Contains the frequency axis, FRC curve, per-ring pixel counts, and metadata.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Images must have the same shape; got {image1.shape} vs {image2.shape}."
        )
    if image1.ndim != 2:
        raise ValueError("compute_frc expects 2-D images.")
    
    if image1.shape[0] != image1.shape[1]:
        raise ValueError("compute_frc requires square images to ensure isotropic frequency mapping.")

    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)

    # mean-subtract (removes DC offset; avoids a huge spike in ring 0)
    img1 -= img1.mean()
    img2 -= img2.mean()

    # apodization
    if apodize:
        window = _tukey_2d(img1.shape, alpha=apodization_alpha)
        img1 *= window
        img2 *= window

    # Compute FFT
    F1 = np.fft.fftshift(np.fft.fft2(img1))
    F2 = np.fft.fftshift(np.fft.fft2(img2))

    # radial coordinate map
    ny, nx = img1.shape
    cy, cx = ny // 2, nx // 2
    y_idx, x_idx = np.indices((ny, nx))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    # Max usable radius (inscribed circle — avoids corners where rings are incomplete)
    max_r_full = int(min(cy, cx))
    max_r = int(np.floor(max_r_full * nyquist_fraction))

    # Bin index per pixel (integer ring label)
    r_int = np.round(r).astype(np.int32)

    # per-ring sums via np.bincount
    mask_valid = r_int <= max_r  # only pixels within the usable radius

    r_flat = r_int[mask_valid]
    F1_flat = F1[mask_valid]
    F2_flat = F2[mask_valid]

    n_rings = max_r + 1

    # Real part of cross-correlation numerator: Re[sum F1 · conj(F2)]
    cross_re = np.real(F1_flat * np.conj(F2_flat))
    # Power in each image
    power1 = np.abs(F1_flat) ** 2
    power2 = np.abs(F2_flat) ** 2

    num = np.bincount(r_flat, weights=cross_re, minlength=n_rings)
    denom1 = np.bincount(r_flat, weights=power1, minlength=n_rings)
    denom2 = np.bincount(r_flat, weights=power2, minlength=n_rings)
    n_pixels = np.bincount(r_flat, minlength=n_rings).astype(float)

    denom = np.sqrt(denom1 * denom2)
    with np.errstate(invalid="ignore", divide="ignore"):
        frc = np.where(denom > 0, num / denom, 0.0)

    # clip to [-1, 1] (floating-point rounding can push slightly outside)
    frc = np.clip(frc, -1.0, 1.0)

    # frequency axis
    # k_i = i / (box_size * pixel_size)   [units: 1/Å]
    # Use the exact image dimension to define the frequency step
    box_size = image1.shape[0] 
    freqs = np.arange(n_rings) / (box_size * pixel_size)

    return FRCData(
        freqs=freqs,
        frc=frc,
        n_pixels=n_pixels,
        pixel_size=pixel_size,
        box_size=box_size,
    )


# Threshold functions


def _threshold_fixed(freqs: np.ndarray, value: float) -> np.ndarray:
    """Constant threshold (0.143 or 0.5)."""
    return np.full_like(freqs, value)


def _threshold_half_bit(freqs: np.ndarray, n_pixels: np.ndarray) -> np.ndarray:
    """
    1/2-bit information threshold (Rosenthal & Henderson 2003).

    At each ring the threshold is the FRC value at which half a bit of
    information per pixel is transferred.  It depends on the number of
    pixels in the ring, making it signal-content-aware.

    T(k) = (0.2071 + 1.9102 / sqrt(n(k))) / (1.2071 + 0.9102 / sqrt(n(k)))
    """
    sqrt_n = np.sqrt(np.maximum(n_pixels, 1.0))
    T = (0.2071 + 1.9102 / sqrt_n) / (1.2071 + 0.9102 / sqrt_n)
    return T


def get_threshold(
    frc_data: FRCData,
    threshold: FRCThreshold = FRCThreshold.ONE_OVER_SEVEN,
) -> np.ndarray:
    """
    Return the threshold curve as an array aligned with result.freqs.

    Parameters
    ----------
    threshold : FRCThreshold
        - "0.143"    Fixed criterion; appropriate for gold-standard half-sets.
        - "0.5"      Legacy fixed criterion (can be overly conservative).
        - "half-bit" Signal-dependent; theoretically preferred for cryoEM 2D.
    """
    if threshold == "0.143":
        return _threshold_fixed(frc_data.freqs, 0.143)
    elif threshold == "0.5":
        return _threshold_fixed(frc_data.freqs, 0.5)
    elif threshold == "half-bit":
        return _threshold_half_bit(frc_data.freqs, frc_data.n_pixels)
    else:
        raise ValueError(
            f"Unknown threshold '{threshold}'. Use '0.143', '0.5', or 'half-bit'."
        )


# Resolution from threshold crossing
def get_resolution(
    frc_data: FRCData,
    threshold: FRCThreshold = FRCThreshold.ONE_OVER_SEVEN,
) -> float:
    """
    Find the spatial resolution (Å) where FRC first drops below the threshold.

    Uses monotone cubic interpolation (PCHIP) in frequency space so the
    crossing point is smooth and not sensitive to the discrete ring spacing.
    Falls back to linear interpolation when fewer than 4 points are available.

    Parameters
    ----------
    result    : FRCResult from compute_frc()
    threshold : which threshold curve to use (see get_threshold())

    Returns
    -------
    float
        Resolution in Å.  Returns np.inf if FRC never drops below the threshold
        (all signal, unreliable input) or np.nan if something is degenerate.
    """
    freqs = frc_data.freqs
    frc = frc_data.frc
    T = get_threshold(frc_data, threshold)

    # difference curve: positive where FRC > threshold
    diff = frc - T

    # find first sign change from positive to negative (first crossing)
    sign_changes = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]

    if len(sign_changes) == 0:
        # FRC never crosses threshold — resolution is at/beyond Nyquist
        warnings.warn(
            f"FRC curve never crosses the {threshold} threshold. "
            "Returning Nyquist resolution.",
            stacklevel=2,
        )
        res = 2.0 * frc_data.pixel_size  # Nyquist
        return res

    idx = sign_changes[0]

    # Interpolate using PCHIP on a local window for smooth crossing
    lo = max(0, idx - 2)
    hi = min(len(freqs) - 1, idx + 3)
    diff_window = diff[lo : hi + 1]
    freqs_window = freqs[lo : hi + 1]

    if len(freqs_window) >= 4:
        try:
            interp_fn = interpolate.PchipInterpolator(freqs_window, diff_window)
            # Bracket the root
            f_lo, f_hi = freqs_window[0], freqs_window[-1]
            if interp_fn(f_lo) * interp_fn(f_hi) > 0:
                raise ValueError("No sign change in interpolation window.")
            freq_cross = brentq(interp_fn, freqs[idx], freqs[idx + 1])
        except Exception:
            # Fall back to linear interpolation between idx and idx+1
            f1, f2 = diff[idx], diff[idx + 1]
            q1, q2 = freqs[idx], freqs[idx + 1]
            freq_cross = q1 + (-f1) * (q2 - q1) / (f2 - f1) if f2 != f1 else q1
    else:
        # Linear fallback
        f1, f2 = diff[idx], diff[idx + 1]
        q1, q2 = freqs[idx], freqs[idx + 1]
        freq_cross = q1 + (-f1) * (q2 - q1) / (f2 - f1) if f2 != f1 else q1

    res = np.inf if freq_cross <= 0 else 1.0 / freq_cross

    return res


def area_under_frc(frc_data: FRCData, nyquist_fraction: float = 1.0) -> float:
    """
    Area Under the FRC curve (AUFRC).

    Integrates the FRC over [0, nyquist_fraction * f_Nyquist] and normalises
    by that frequency range, yielding a value in [0, 1].

    Higher AUFRC → better average correlation across all spatial frequencies
    → better class average quality.

    This is a threshold-free summary statistic.  It is particularly useful
    when comparing methods where different thresholds would give different
    resolution rankings — AUFRC captures the whole curve.

    Parameters
    ----------
    nyquist_fraction : float
        Fraction of the Nyquist frequency up to which to integrate.
        Default 1.0 (full range).  Use < 1 to exclude the noisy outermost rings.
    """
    freqs = frc_data.freqs
    frc_ = frc_data.frc

    f_nyq = freqs[-1]
    f_max = nyquist_fraction * f_nyq

    mask = freqs <= f_max
    f_sel = freqs[mask]
    c_sel = np.clip(frc_[mask], 0.0, 1.0)  # negative FRC has no information content

    if len(f_sel) < 2:
        return float("nan")

    integral = integrate.trapezoid(c_sel, f_sel)
    return float(integral / (f_sel[-1] - f_sel[0]))
