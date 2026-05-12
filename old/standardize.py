from typing import Callable

import numpy as np
import torch

STANDARDIZE_TYPES = ["global", "per_image"]


class ImageTransform:
    def fit(self, images: torch.Tensor) -> "ImageTransform":
        return self
    
    def transform(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplemented("ImageTransform subclasses must implement the transform method")
    
    def fit_transform(self, images: torch.Tensor) -> torch.Tensor:
        self.fit(images)
        return self.transform(images)        
    

class PerImageZScore(ImageTransform):
    def __init__(self, eps: float=1.0e-8):
        self.eps = eps

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        # Dynamically determine dimensions to reduce over to better handle 
        # different object dimension (e.g. volumes)
        reduce_dims = tuple(range(1, images.ndim))

        # Compute per-iamge mean and std
        means = images.mean(dim=reduce_dims, keepdim=True)
        stds = images.std(dim=reduce_dims, keepdim=True)

        return (images - means) / (stds + self.eps)



class DatasetScalarZScore(ImageTransform):
    def __init__(self, eps:float =1.0e-8):
        self.eps = eps

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.mean()
        std = images.std()
        return (images - mean) / (std + self.eps)
    

class PerPixelZScore(ImageTransform):
    def __init__(self, eps: float = 1.0e-8):
        self.eps = eps

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        pixel_means = images.mean(dim=0, keepdim=True)
        pixel_stds = images.std(dim=0, keepdim=True)

        return (images - pixel_means) / (pixel_stds + self.eps)



class ImageStandardizer:
    def fit_transform(
        self, images: torch.Tensor, transformation_type: str | None = None
    ):
        if transformation_type == "global":
            return (images - images.mean()) / (images.std() + 1.0e-8)

        if transformation_type == "per_image":
            means = images.mean(dim=(1, 2), keepdim=True)
            stds = images.std(dim=(1, 2), keepdim=True)
            return (images - means) / (stds + 1.0e-8)

        raise ValueError(
            f"Unrecognised transformation type for standardization: {transformation_type}." 
            f"Accepted values are {STANDARDIZE_TYPES}."
        )


def standardize(
    images: np.ndarray, transformation_type: str | None = None
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray] | None]:
    undo_transformation = None
    if transformation_type is None:
        return images, undo_transformation

    if transformation_type == "global":
        global_std = images.std()
        global_mean = images.mean()
        standardized_particles = (images - global_mean) / (global_std + 1.0e-8)

        def undo_transformation(particles: np.ndarray) -> np.ndarray:
            return particles * (global_std + 1.0e-8) + global_mean

        return standardized_particles, undo_transformation

    raise ValueError(
        f"Unrecognised transformation type for standardization: {transformation_type}. Accepted values are {STANDARDIZE_TYPES}"
    )
