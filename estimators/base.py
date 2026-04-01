import torch
import numpy as np
from typing import Optional
from enum import Enum

class Space(Enum):
    REAL = 1
    FOURIER_REAL = 2
    FOURIER_IMAG = 3

class Estimator:
    def __init__(self, device: Optional[str] = None):
        self.avg = None
        self.final_weights = {
            space: None for space in Space
        }
        self.device = torch.device(device) if device is not None else None

    def _prepare_data(self, images) -> torch.Tensor:
        """Ensures input is a PyTorch tensor on the correct device."""
        device = self.device if self.device is not None else images.device
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        # Convert to float32 for stable gradient/division operations
        return images.to(dtype=torch.float32, device=device)

    def fit(self, images):
        raise NotImplementedError("Subclasses must implement the fit method.")
