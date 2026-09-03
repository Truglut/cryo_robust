from pathlib import Path
from typing import Any, Collection, Literal

import numpy as np

from cryo_robust.comparison.domain.reports import EvaluationReport, EvaluationStudy
from cryo_robust.comparison.visualization.plotting import save_snr_reports_figures

# Report sections
from .sections import ReportSection
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
from .gmm_fits import generate_gmm_fits_section


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
    gmm_initial_reference: bool,
    gmm_distance_plot_type: Literal[
        "stacked", "overlayed-full", "overlayed-proportional"
    ] = "overlayed-proportional",
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
    gmm_initial_reference : bool, optional
        Whether to add the initial reference to GMM distribution plots. Default is True.
        Will not do anything if ``ReportSection.GMM`` is not in ``sections``, since
        the GMM plots will not be generated in that case.

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
    if ReportSection.GMM in sections:
        plot_types.add("gmm")

    # If results are EvaluationStudy, take the first report for each snr for plotting
    report = list(results.values())[0]
    if isinstance(report, EvaluationStudy):
        results_for_plotting = {snr: study.reports[0] for snr, study in results.items()}
    else:
        results_for_plotting = results

    plots = save_snr_reports_figures(
        results_for_plotting,
        output_path=output_path,
        figures_path=figures_path,
        frc_x_axis_freqs=True,
        plot_types=plot_types,
        gmm_initial_reference=gmm_initial_reference,
        gmm_distance_plot_type=gmm_distance_plot_type,
        **plot_options,
    )

    plots_section = (
        generate_weight_and_frc_plots_section(
            plots=plots,
            output_path=output_path,
        )
        if (plot_types & {"weights", "frc", "fourier-rings"})
        else ""
    )

    gmm_section = (
        generate_gmm_fits_section(plots=plots, output_path=output_path)
        if "gmm" in plot_types
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

        f.write(gmm_section)

        f.write(images_section)

        f.write("\n\\end{document}")
