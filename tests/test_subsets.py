import torch

from cryo_robust.estimators.data import ImageBatch
from cryo_robust.estimators.results import WeightSet


def test_weight_set_subset():
    weights = WeightSet(
        real=torch.arange(5.0).view(5, 1, 1),
        fourier_real=torch.arange(10.0).view(5, 1, 2),
    )

    subset = weights.subset(torch.tensor([1, 3]))

    torch.testing.assert_close(
        subset.real,
        weights.real[[1, 3]],
    )
    torch.testing.assert_close(
        subset.fourier_real,
        weights.fourier_real[[1, 3]],
    )
    assert subset.fourier_imag is None


def test_image_batch_subset_preserves_available_representations():
    images = torch.randn(6, 8, 8)
    batch = ImageBatch.from_real(images)

    fourier = batch.ensure_fourier()

    indices = torch.tensor([1, 4])
    subset = batch.subset(indices)

    torch.testing.assert_close(
        subset.real,
        images[indices],
    )
    torch.testing.assert_close(
        subset.fourier,
        fourier[indices],
    )

    assert subset.real_variance_value is None
    assert subset.fourier_variance_value is None
