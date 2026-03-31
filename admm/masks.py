import torch
import numpy as np
from typing import Tuple


def create_circular_mask(image_shape: Tuple[int, int], radius: float) -> np.ndarray:
    h, w = image_shape

    center = (w // 2, h // 2)
    Y, X = np.ogrid[:h, :w]
    dist = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    return dist <= radius**2