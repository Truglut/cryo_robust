import torch

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