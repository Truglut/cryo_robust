from pathlib import Path

from .figures import create_figure_grid


def generate_gmm_fits_section(
    plots: dict[float, dict[str, list[Path]]], output_path: Path
):

    text = "\n\\section{GMM plots}\n"

    for snr in plots:
        text += f"\n\\subsection{{SNR {snr:.3f}}}\n"

        gmm_figure_paths = [
            p.relative_to(output_path) for p in plots[snr].get("gmm", [])
        ]
        n = len(gmm_figure_paths)

        text += create_figure_grid(
            figure_paths=gmm_figure_paths,
            captions=[f"GMM Fit {i + 1}" for i in range(n)],
            figures_per_row=3,
        )

    return text
