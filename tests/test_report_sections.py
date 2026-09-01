from cryo_robust.comparison.evaluation.report_building import (
    get_report_computation_options,
)
from cryo_robust.comparison.latex.sections import (
    ReportSection,
    ALL_REPORT_SECTIONS,
    resolve_report_sections,
)


def test_resolve_report_sections():
    assert resolve_report_sections(["classification", "images"]) == {
        ReportSection.CLASSIFICATION,
        ReportSection.IMAGES,
    }

    assert resolve_report_sections(["all"]) == ALL_REPORT_SECTIONS
    assert resolve_report_sections(None) == ALL_REPORT_SECTIONS


def test_report_computation_options_follow_requested_outputs():
    options = get_report_computation_options(
        report_sections={
            ReportSection.CLASSIFICATION,
            ReportSection.IMAGES,
        },
        plot_types={"frc"},
    )

    assert options.classification
    assert options.store_estimated_images
    assert options.reconstruction  # required by the requested FRC plot

    assert not options.scores
    assert not options.fourier_ring_metrics
