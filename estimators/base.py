import torch
import numpy as np
from typing import Dict
from utils.space import Space


class Estimator:
    def __init__(self, device: str | None = None):
        self.avg = None
        self.final_weights = {space: None for space in Space}
        self.device = torch.device(device) if device is not None else None

    def _prepare_data(
        self, images: Dict[Space, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray
    ) -> Dict[Space, torch.Tensor]:
        """Ensures input is a PyTorch tensor on the correct device."""
        device = self.device if self.device is not None else images.device
        if isinstance(images, dict):
            for space, image in images.items():
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image)
                # Convert to float32 for stable gradient/division operations
                images[space] = image.to(dtype=torch.float32, device=device)
            return images
        
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        return images.to(dtype=torch.float32, device=device)

    def fit(self, images):
        raise NotImplementedError("Subclasses must implement the fit method.")
