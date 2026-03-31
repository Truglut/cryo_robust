import torch
import numpy as np

class Estimator:
    def __init__(self, device: str = "cpu"):
        self.avg = None
        self.final_weights = None
        self.device = torch.device(device)

    def _prepare_data(self, images) -> torch.Tensor:
        """Ensures input is a PyTorch tensor on the correct device."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        # Convert to float32 for stable gradient/division operations
        return images.to(dtype=torch.float32, device=self.device)

    def fit(self, images):
        raise NotImplementedError("Subclasses must implement the fit method.")
