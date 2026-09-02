from typing import Iterable

from enum import Enum


class ReportSection(str, Enum):
    """Sections that can be included in a LaTeX report."""

    CLASSIFICATION = "classification"
    RECONSTRUCTION = "reconstruction"
    WEIGHTS = "weights"
    FRC = "frc"
    FOURIER_RINGS = "fourier-rings"
    IMAGES = "images"
    GMM = "gmm"


ALL_REPORT_SECTIONS = frozenset(ReportSection)


REPORT_SECTION_CHOICES = [
    *(section.value for section in ReportSection),
    "all",
]


def resolve_report_sections(
    values: Iterable[str] | None,
) -> frozenset[ReportSection]:
    """
    Resolve command-line report section values.

    If no explicit sections are provided, or ``"all"`` is present, all
    report sections are selected.

    Parameters
    ----------
    values : Iterable[str] or None
        Section names provided through the command line.

    Returns
    -------
    frozenset[ReportSection]
        Selected report sections.
    """
    if not values:
        return ALL_REPORT_SECTIONS

    values = set(values)

    if "all" in values:
        return ALL_REPORT_SECTIONS

    return frozenset(ReportSection(value) for value in values)