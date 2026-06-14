from pathlib import Path


def create_figure_block(
    figure_path: Path, caption: str = "", width: str = "\\textwidth"
):
    """
    Generate a LaTeX `figure` environment for a single image.

    Parameters
    ----------
    figure_path : Path
        Path to the figure file to include.
    caption : str, optional
        Caption text displayed below the figure. Defaults to an empty string.
    width : str, optional
        Desired width for the figure, formatted for the LaTeX
        ``\\includegraphics`` command, e.g. ``"\\textwidth"`` or ``"0.8\\textwidth"``.
        Defaults to ``"\\textwidth"``.

    Returns
    -------
    str
        LaTeX string containing a complete `figure` environment with the
        specified image and caption.
    """
    return (
        f"\n\\begin{{figure}}[H]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width={width}]{{{figure_path.as_posix()}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"\\end{{figure}}\n"
    )


def create_figure_grid(
    figure_paths: list[Path],
    captions: list[str],
    figures_per_row: int,
    width: str | None = None,
) -> str:
    """
    Generate a LaTeX block embedding a list of figures arranged in rows.

    Each row uses a ``subfigure``-based layout inside a single ``figure``
    environment.  The last row is centred automatically even when it contains
    fewer figures than the others.

    Parameters
    ----------
    figure_paths : list[Path]
        Paths to the figure files to include.
    captions : list[str]
        Captions for each of the figures in `figure_paths`.
        `captions[i]` will be the caption for `figure_paths[i]`.
    figures_per_row : int
        Number of figures to place side-by-side in each row.
    width : str, optional
        Desired width for every individual figure.  Must be in a format
        accepted by ``\\includegraphics``, e.g. ``"0.45\\textwidth"``.
        Defaults to ``1/<figures_per_row>\\textwidth`` with a small gutter
        subtracted so the subfigures fit on one line.

    Returns
    -------
    str
        LaTeX string containing one ``figure`` environment per row.
    """
    if figures_per_row < 1:
        raise ValueError("figures_per_row must be at least 1")
    if len(figure_paths) != len(captions):
        raise ValueError("figure_paths and captions must be the same length")

    if width is None:
        # Leave ~0.02 of \textwidth as gutter between each pair of neighbours.
        unit = round(1 / figures_per_row - 0.02, 4)
        width = f"{unit}\\textwidth"

    text = ""

    for i in range(0, len(figure_paths), figures_per_row):
        paths_row = figure_paths[i : i + figures_per_row]
        captions_row = captions[i : i + figures_per_row]

        text += "\n\\begin{figure}[H]\n  \\centering\n"

        for path, caption in zip(paths_row, captions_row):
            text += (
                f"  \\begin{{subfigure}}[t]{{{width}}}\n"
                f"    \\centering\n"
                f"    \\includegraphics[width=\\textwidth]{{{path.as_posix()}}}\n"
                f"    \\caption{{{caption}}}\n"
                f"  \\end{{subfigure}}\n"
            )

        text += "\\end{figure}\n"

    return text


def create_figure_section(
    figure_paths: list[Path], caption_prefix: str, width: str = "\\textwidth"
) -> str:
    """
    Generate a LaTeX block embedding a list of figures.

    Parameters
    ----------
    figure_paths : list[Path]
        Paths to the figure files to include.
    caption_prefix : str
        Prefix used in each figure caption, e.g. `"Weight distribution"`.
    width: str
        Desired width for the figures. Must be in an appropriate format for the
        Desired width for the figures. Must be in an appropriate format for the
        \\includegraphics LaTeX command, e.g. `"\\textwidth"`.

    Returns
    -------
    str
        LaTeX string containing one `figure` environment per path.
    """
    text = ""

    for i, path in enumerate(figure_paths, start=1):
        text += create_figure_block(
            figure_path=path, caption=f"{caption_prefix} {i}", width=width
        )

    return text
