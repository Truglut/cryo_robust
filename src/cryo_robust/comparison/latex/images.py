from pathlib import Path
from typing import Any

import numpy as np

from cryo_robust.comparison.domain.reports import EvaluationReport, EvaluationStudy
from cryo_robust.comparison.visualization.plotting import generate_image_plots
from .figures import create_figure_block, create_figure_grid


def generate_images_section(
    results: dict[float, EvaluationReport] | dict[float, EvaluationStudy],
    ground_truth_image: np.ndarray,
    output_path: Path,
    figures_path: Path,
    plot_options: dict[str, Any],
) -> str:
    """
    Generate the LaTeX code for the images section of the report, containing the
    ground truth image and the various reconstructed averages for the methods at
    each SNR level.

    Parameters
    ----------
    results : dict[float, EvaluationReport] or dict[float, EvaluationStudy]
        Dict mapping each SNR level to its evaluation report or dict mapping each
        SNR level to its evaluation study that contains one report per simulation
        run.
    ground_truth_image: np.ndarray
        2-dimensional array holding the ground truth image used
        for generating the dataset.
    output_path : Path
        Path to the directory where the `report.tex` will be generated.
    figures_path : Path
        Path to the directory where the images will be saved.
    plot_options : dict[str, Any]
        Dict containing the following keyword arguments for figure generation:
            - max_subplots: int
            - density: bool
            - dpi: int

    Returns
    -------
    str
        LaTeX text for the images section ready to be written into the report document.
        This section contains:
            - A figure representing the ground truth image from which the dataset was
            generated.
            - One subsection per SNR level, which contains the reconstructed averages
            produced by each of the estimation methods at that SNR level.
    """
    # If results are EvaluationStudy, take the first report for each snr
    report = list(results.values())[0]
    if isinstance(report, EvaluationStudy):
        results = {snr: study.reports[0] for snr, study in results.items()}

    text = "\n\\section{Estimated images}\n"

    ground_truth_fig_path = figures_path / "ground_truth.png"
    ground_truth_fig_path = generate_image_plots(
        [ground_truth_image],
        [ground_truth_fig_path],
        link_contrast=False,
        dpi=plot_options["dpi"],
    )[0]
    text += "\n\\textbf{Ground truth}\n"

    text += create_figure_block(
        ground_truth_fig_path.relative_to(output_path),
        caption="Ground truth image",
        width="0.3\\textwidth",
    )

    images_dir = figures_path / "estimated_avgs"

    for snr, report in results.items():
        text += "\n\\newpage\n"
        text += f"\n\\subsection{{SNR {snr:.3f}}}\n"

        snr_str = f"{snr:.3f}".replace(".", "p")
        snr_images_dir = images_dir / snr_str
        snr_images_dir.mkdir(parents=True, exist_ok=True)

        method_names = [mr.name for mr in report.method_results]
        images = [mr.estimated_img for mr in report.method_results]
        save_paths = [
            snr_images_dir / (mr.name + ".png") for mr in report.method_results
        ]

        save_paths = generate_image_plots(
            images, save_paths=save_paths, link_contrast=True, dpi=plot_options["dpi"]
        )
        save_paths = [path.relative_to(output_path) for path in save_paths]
        # captions = [f"Estimation with {name}" for name in method_names]

        text += create_figure_grid(save_paths, captions=method_names, figures_per_row=3)

    return text
