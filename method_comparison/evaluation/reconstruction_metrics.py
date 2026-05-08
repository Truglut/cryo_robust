import numpy as np
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

from method_comparison.domain.metrics import ReconstructionMetrics

### Fourier ring correlation ###


def compute_fsc(
    image1: np.ndarray, image2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the 2D Fourier Ring/Shell Correlation between two images.
    Returns the normalized frequencies and the FSC curve.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape to compute FSC.")

    # Compute 2D FFTs and shift zero frequency to center
    F1 = np.fft.fftshift(np.fft.fft2(image1))
    F2 = np.fft.fftshift(np.fft.fft2(image2))

    # Create radial distance map
    shape = image1.shape
    center = (shape[0] // 2, shape[1] // 2)
    y, x = np.indices(shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = np.round(r).astype(int)

    # Calculate Nyquist frequency (max radius)
    max_r = int(np.min([center[0], center[1]]))

    fsc = np.zeros(max_r)
    freqs = np.arange(max_r) / max_r  # Normalized frequency [0, 1] (1 = Nyquist)

    for i in range(max_r):
        mask = r == i
        if np.sum(mask) == 0:
            continue

        f1_shell = F1[mask]
        f2_shell = F2[mask]

        # Cross-correlation numerator
        num = np.real(np.sum(f1_shell * np.conj(f2_shell)))

        # Normalization denominator
        den = np.sqrt(np.sum(np.abs(f1_shell) ** 2) * np.sum(np.abs(f2_shell) ** 2))

        fsc[i] = num / den if den > 0 else 0.0

    return freqs, fsc


def get_resolution_from_fsc(
    freqs: np.ndarray, fsc: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Finds the spatial frequency where the FSC curve first drops below the threshold.
    Uses linear interpolation for sub-bin precision.
    """
    drop_idx = np.where(fsc < threshold)[0]

    if len(drop_idx) == 0:
        return freqs[-1]  # Never drops below threshold (perfect resolution)

    idx = drop_idx[0]
    if idx == 0:
        return freqs[0]

    # Linear interpolation
    f1, f2 = fsc[idx - 1], fsc[idx]
    q1, q2 = freqs[idx - 1], freqs[idx]

    # Solve for frequency crossing the threshold
    freq_thresh = q1 + (threshold - f1) * (q2 - q1) / (f2 - f1)
    return freq_thresh


### All reconstruction metrics


def compute_reconstruction_metrics(
    ground_truth_img: np.ndarray | None, estimated_img: np.ndarray, fsc_threshold: float
) -> tuple[ReconstructionMetrics, tuple[np.ndarray, np.ndarray]]:
    if ground_truth_img is None:
        return None
    rmse = root_mean_squared_error(ground_truth_img, estimated_img)
    corr, _ = pearsonr(ground_truth_img.flatten(), estimated_img.flatten())
    fsc_data = compute_fsc(estimated_img, ground_truth_img)
    resolution = get_resolution_from_fsc(
        freqs=fsc_data[0], fsc=fsc_data[1], threshold=fsc_threshold
    )

    metrics = ReconstructionMetrics(
        rmse=rmse, pearson_corr=corr, fsc_resolution=resolution
    )

    return metrics, fsc_data
