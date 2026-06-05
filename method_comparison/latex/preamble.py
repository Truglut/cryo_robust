from typing import Iterable

# Required LaTeX packages for rendering tables and formatting the document.
LATEX_PACKAGES = (
    "booktabs",
    "float",
    "caption",
    "graphicx",
    "subcaption",
)

DEFAULT_MARGIN = "3cm"


def generate_document_preamble(
    packages: Iterable[str] = LATEX_PACKAGES, margin: str = DEFAULT_MARGIN
) -> str:
    """
    Generate the LaTeX preamble required for the report.

    The generated preamble includes:
    - Document class declaration
    - Required LaTeX packages
    - Page geometry configuration
    - Table caption positioning

    Returns
    -------
    str
        LaTeX preamble string.
    """

    lines = [
        r"\documentclass{article}",
        "",
        r"% Packages",
        *(rf"\usepackage{{{pkg}}}" for pkg in packages),
        "",
        r"% Layout",
        rf"\usepackage[margin={margin}]{{geometry}}",
        "",
        r"% Caption configuration",
        r"\captionsetup[table]{position=bottom}",
    ]

    return "\n".join(lines)
