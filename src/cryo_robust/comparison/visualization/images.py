from cryo_robust.comparison.visualization.plot_utils import save_figure

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Iterable


def generate_image_plots(
    images: Iterable[np.ndarray],
    save_paths: Iterable[Path],
    link_contrast: bool = True,
    *,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 150,
) -> list[Path]:
    """
    Generate and save individual plots for a sequence of images.

    Parameters
    ----------
    images : Iterable[np.ndarray]
        An iterable of 2D arrays representing the images to plot.
    save_paths : Iterable[Path]
        An iterable of file paths where the corresponding images should be saved.
    link_contrast : bool, optional
        If True, applies a global minimum and maximum contrast across all images.
        Default is True.
    figsize : tuple[int, int], optional
        The dimensions of each generated figure in inches. Default is (6, 6).
    dpi : int, optional
        The resolution of the saved figures in dots per inch. Default is 150.

    Returns
    -------
    list[Path]
        A list of paths where the images were saved.

    Raises
    ------
    ValueError
        If the number of images and save paths do not match.
    """
    # Convert iterables to lists to safely calculate length and iterate multiple times
    images = list(images)
    save_paths = list(save_paths)

    if len(images) != len(save_paths):
        raise ValueError("images and save_paths must have the same length")

    # Determine global contrast limits if requested
    if link_contrast:
        vmin = min([image.min() for image in images])
        vmax = max([image.max() for image in images])
    else:
        vmin = None
        vmax = None

    for image, save_path in zip(images, save_paths):
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(image, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)

        # Remove axes and whitespace for a clean image output
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save cleanly
        save_figure(fig=fig, path=save_path, dpi=dpi, pad_inches=0)

    return save_paths
