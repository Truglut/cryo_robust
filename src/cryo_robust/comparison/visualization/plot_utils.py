import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pathlib import Path


def save_figure(fig: Figure, path: Path, dpi: int, **kwargs) -> Path:
    fig.savefig(path, dpi=dpi, **kwargs)
    plt.close(fig)
    return path


# Helper for consistent coloring in plots with different types of outliers
LABEL_MAP = {
    0: {"name": "Genuine", "color": "#083DB0"},
    1: {"name": "Misaligned", "color": "orange"},
    2: {"name": "Misclassified", "color": "#F9290D"},
    3: {"name": "Noise", "color": "darkorange"},
}

# For plots with only good vs. bad images
GOOD_BAD_PLOT_OPTIONS = {
    "good": {"label": "Inliers", "color": "#083DB0"},
    "bad": {"label": "Outliers", "color": "#F9290D"},
}

BASE_PLOT_OPTIONS = {
    "max_subplots": 3,
    "density": False,
    "dpi": 150,
}

HISTOGRAM_TYPE = "stepfilled"

ALL_PLOT_TYPES = frozenset(
    {
        "weights",
        "frc",
        "fourier-rings",
    }
)
