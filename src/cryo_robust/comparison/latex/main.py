from pathlib import Path
from typing import Any, Collection

import numpy as np

from cryo_robust.comparison.domain.reports import EvaluationReport, EvaluationStudy
from .sections import ReportSection

# Report sections
from .preamble import generate_document_preamble
from .experiment_info import write_experiment_info
from .weights_and_frc import (
    generate_weight_and_frc_plots_section,
)
from .reconstruction import (
    generate_reconstruction_section,
)
from .classification import (
    generate_classification_section,
)
from .images import generate_images_section


def generate_latex_report(
    results: dict[float, EvaluationReport] | dict[float, EvaluationStudy],
    output_path: Path,
    cfg: dict[str, Any],
    ground_truth_image: np.ndarray,
    sections: Collection[ReportSection],
    plot_options: dict[str, Any],
    *,
    standardize: str,
    per_image_noise_std: bool,
    fourier_weight_mask: str,
) -> None:
    """
    Generate a LaTeX report from evaluation results.

    The report contains only the sections explicitly requested through
    ``sections``. Figures and tables required by those sections are generated
    and written alongside the LaTeX source in the output directory.

    Parameters
    ----------
    results : dict[float, EvaluationReport] or dict[float, EvaluationStudy]
        Evaluation results indexed by SNR.
    output_path : pathlib.Path
        Directory where the LaTeX report and its generated assets are written.
    cfg : dict[str, Any]
        Experiment configuration used to generate the results.
    ground_truth_image : np.ndarray
        Ground-truth reference image used in report figures when required.
    sections : Collection[ReportSection]
        Sections to include in the report.
    plot_options : dict[str, Any]
        Plotting options used when generating report figures.
    standardize : str
        Standardization strategy used in the experiment.
    per_image_noise_std : bool
        Whether noise standard deviations were sampled independently for each
        image.
    fourier_weight_mask : str
        Fourier weight mask configuration used during evaluation.

    Returns
    -------
    None
    """
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "report.tex"
    figures_path = output_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    # Preamble: document class, packages and setup
    document_preamble = generate_document_preamble()

    # Classification section: recall, precision, etc.
    class_section = (
        generate_classification_section(
            results=results,
            output_path=output_path,
            figures_path=figures_path,
            dpi=plot_options["dpi"],
        )
        if ReportSection.CLASSIFICATION in sections
        else ""
    )

    # Reconstruction section: rmse, correlation, resolution
    reconstruction_section = (
        generate_reconstruction_section(
            results=results,
            output_path=output_path,
            figures_path=figures_path,
            dpi=plot_options["dpi"],
        )
        if ReportSection.RECONSTRUCTION in sections
        else ""
    )

    # Save figures and generate the plots section
    plot_types = set()
    if ReportSection.WEIGHTS in sections:
        plot_types.add("weights")
    if ReportSection.FRC in sections:
        plot_types.add("frc")
    if ReportSection.FOURIER_RINGS in sections:
        plot_types.add("fourier-rings")
    plots_section = (
        generate_weight_and_frc_plots_section(
            results=results,
            output_path=output_path,
            figures_path=figures_path,
            plot_options=plot_options,
            plot_types=plot_types,
        )
        if plot_types
        else ""
    )

    # Images section with ground truth and estimation
    images_section = (
        generate_images_section(
            results=results,
            ground_truth_image=ground_truth_image,
            output_path=output_path,
            figures_path=figures_path,
            plot_options=plot_options,
        )
        if ReportSection.IMAGES in sections
        else ""
    )

    # Write all the contents to the file
    with report_path.open("w") as f:
        f.write(document_preamble)

        f.write("\n\n\\begin{document}\n\n")

        f.write(
            write_experiment_info(
                cfg=cfg,
                snr_list=results.keys(),
                standardize=standardize,
                per_image_noise_std=per_image_noise_std,
                fourier_weight_mask=fourier_weight_mask,
            )
        )

        f.write(class_section)

        f.write(reconstruction_section)

        f.write(plots_section)

        f.write(images_section)

        f.write("\n\\end{document}")
