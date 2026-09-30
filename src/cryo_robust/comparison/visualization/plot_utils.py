import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pathlib import Path


def save_figure(fig: Figure, path: Path, dpi: int, **kwargs) -> Path:
    fig.savefig(path, dpi=dpi, **kwargs)
    plt.close(fig)
    return path


INLIER_BLUE = "#083DB0"
OUTLIER_RED = "#F9290D"

# Helper for consistent coloring in plots with different types of outliers
LABEL_MAP = {
    0: {"name": "Genuine", "color": INLIER_BLUE},
    1: {"name": "Misaligned", "color": "orange"},
    2: {"name": "Misclassified", "color": OUTLIER_RED},
    3: {"name": "Noise", "color": "darkorange"},
    4: {"name": "Shifted", "color": OUTLIER_RED}
}

# For plots with only good vs. bad images
GOOD_BAD_PLOT_OPTIONS = {
    "good": {"label": "Inliers", "color": INLIER_BLUE},
    "bad": {"label": "Outliers", "color": OUTLIER_RED},
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
