import torch
import numpy as np

from method_comparison.domain.enums import ImageSpace


class Estimator:
    def __init__(self, device: str | None = None):
        self.avg = None
        self.final_weights = {space: None for space in ImageSpace}
        self.device = torch.device(device) if device is not None else None

    def _prepare_data(
        self,
        images: dict[ImageSpace, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
    ) -> dict[ImageSpace, torch.Tensor]:
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

        if torch.is_complex(images):
            return images.to(dtype=torch.complex64, device=device)

        return images.to(dtype=torch.float32, device=device)

    def fit_tensor(
        self, images: dict[ImageSpace, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[ImageSpace, torch.Tensor]]:
        raise NotImplementedError("Subclasses must implement the fit method.")

    def reconstruct_from_weights(
        self,
        images: dict[ImageSpace, torch.Tensor | None],
        weights: dict[ImageSpace, torch.Tensor | None],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement the reconstruct from weights method"
        )
