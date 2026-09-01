import argparse
from typing import Any, Iterable

PARAMETERS = (
    ("n_copies", "Number of good copies of reference"),
    ("n_copies_rotated", "Number of badly misaligned images"),
    ("n_misclassified", "Number of misclassified images"),
    ("max_rotation_reference", "Maximum rotation of good copies"),
    ("max_rotation_very_rotated", "Maximum rotation of misaligned images"),
    ("min_rotation_very_rotated", "Minimum rotation of misaligned images"),
)


def write_experiment_info(
    cfg: dict[str, Any],
    snr_list: Iterable[float],
    *,
    standardize: str,
    per_image_noise_std: bool,
    fourier_weight_mask: str,
) -> str:
    """
    Generate a LaTeX summary of the experiment configuration.

    Produces an itemized list of image generation parameters and the
    SNR levels used in the experiment.

    Parameters
    ----------
    cfg : dict[str, Any]
        Full experiment configuration dict. Must contain a ``"generation"``
        sub-dict with keys: ``n_copies``, ``n_copies_rotated``,
        ``n_misclassified``, ``max_rotation_reference``,
        ``min_rotation_very_rotated``, and ``max_rotation_very_rotated``.
    snr_list : Iterable[float]
        SNR values tested in the experiment.
    standardize : str
        Standardization strategy used in the experiments
    per_image_noise_std : bool
        Whether the image noise std was calculated independently for each image
        in the experiment
    fourier_weight_mask : str
        Type of mask used to aggregate Fourier-space weights.

    Returns
    -------
    str
        LaTeX-formatted experiment summary block.
    """
    gen = cfg["generation"]

    snr_text = ", ".join(f"{snr:.3f}" for snr in snr_list)

    lines = [
        r"\textbf{Image generation parameters:}",
        "",
        r"\begin{itemize}",
        *(f"\t\\item {label}: ${gen[key]}$." for key, label in PARAMETERS),
        r"\end{itemize}",
        "",
        rf"\noindent\textbf{{Signal-to-noise ratios tested:}} {snr_text}\\",
        rf"\textbf{{Per-image noise std:}} {per_image_noise_std}\\",
        rf"\textbf{{Standardization strategy:}} {standardize}\\",
        rf"\textbf{{Fourier weight mask:}} {fourier_weight_mask}\\",
        "",
    ]

    return "\n".join(lines)
