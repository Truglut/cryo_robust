# These tests use tiny synthetic image batches and deliberately simple weights.
# Their goal is to check that each estimator can run end-to-end, returns finite
# outputs with the expected shapes, and respects the simplest possible case:
# when every image has weight one, IRLS-style estimators should reduce to the
# ordinary sample mean.

import pytest
import torch

from cryo_robust.comparison.domain.enums import ImageSpace
from cryo_robust.estimators.admm import ADMMSolver
from cryo_robust.estimators.data import ImageBatch
from cryo_robust.estimators.distances import l2_norm
from cryo_robust.estimators.gmm import RecursiveGMMEstimator
from cryo_robust.estimators.irls import (
    IRLSSolver,
    IRLSFourier,
    JointIRLSFourier,
    FlatteningIRLSFourier,
)


# Constant weights make the robust update equivalent to a plain weighted mean.
# The shape is chosen to broadcast correctly for real, Fourier and flattened
# representations.
def unit_weights(images: torch.Tensor, reference: torch.Tensor, std: torch.Tensor | float):
    """Broadcastable unit weights for real, Fourier and flattened estimators."""
    trailing_singletons = (1,) * (images.ndim - 1)
    return torch.ones(
        (images.shape[0], *trailing_singletons),
        dtype=torch.float32,
        device=images.device,
    )


# A small deterministic batch with images that differ only by scalar offsets.
# This keeps the expected sample mean easy to compute and avoids stochastic
# failures.
def make_small_images() -> torch.Tensor:
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10.0 + 1.0
    offsets = torch.tensor([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=torch.float32).view(-1, 1, 1)
    return base.unsqueeze(0) + offsets


# A slightly more separated batch for the GMM estimator, where most images are
# close to a base pattern and a few are obvious outliers.  The test only checks
# that responsibilities are valid probabilities, not that clustering is perfect.
def make_clustered_images() -> torch.Tensor:
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10.0 + 1.0
    inliers = torch.stack([base + 0.01 * k for k in range(6)])
    outliers = torch.stack([2.5 * base + 1.0, -1.5 * base])
    return torch.cat([inliers, outliers], dim=0)


# Common output checks for estimators that are expected to reconstruct a real
# image.  Keeping this helper small avoids repeating shape/finiteness assertions.
def assert_valid_real_result(result, expected_shape: tuple[int, int]):
    assert result.average is not None
    assert result.average.shape == expected_shape
    assert torch.isfinite(result.average).all()
    assert result.estimate is not None
    assert torch.isfinite(result.estimate).all()
    assert result.n_iter is not None
    assert result.n_iter >= 1


# With unit weights, real-space IRLS should be indistinguishable from the sample
# mean.  The reconstruction-from-weights path is checked too, because it is used
# later by half-set reconstruction metrics.
def test_real_irls_with_unit_weights_reduces_to_sample_mean():
    images = make_small_images()
    batch = ImageBatch.from_real(images)
    expected = images.mean(dim=0)

    solver = IRLSSolver(
        weight_function=unit_weights,
        max_iter=3,
        tol=1.0e-12,
        space=ImageSpace.REAL,
    )
    result = solver.fit(batch)

    assert_valid_real_result(result, expected_shape=images.shape[1:])
    assert result.weights.real is not None
    assert result.weights.real.shape == (images.shape[0], 1, 1)
    torch.testing.assert_close(result.average, expected, atol=1.0e-5, rtol=1.0e-5)

    reconstructed = solver.reconstruct_from_weights(batch, result.weights)
    torch.testing.assert_close(reconstructed, expected, atol=1.0e-5, rtol=1.0e-5)


# Priors only make sense if both a mean and a variance are provided.  This test
# protects that validation logic from being accidentally removed.
def test_irls_rejects_incomplete_prior_information():
    images = make_small_images()
    batch = ImageBatch.from_real(images)
    solver = IRLSSolver(unit_weights, max_iter=1, space=ImageSpace.REAL)

    with pytest.raises(ValueError):
        solver.fit(batch, prior_mean=torch.zeros_like(images[0]))


# The three Fourier variants use different internal representations, but with
# unit weights they should all reconstruct the same real-space sample mean.
@pytest.mark.parametrize(
    "estimator",
    [
        IRLSFourier(
            IRLSSolver(unit_weights, max_iter=2, space=ImageSpace.FOURIER_REAL),
            IRLSSolver(unit_weights, max_iter=2, space=ImageSpace.FOURIER_IMAG),
        ),
        JointIRLSFourier(
            IRLSSolver(unit_weights, max_iter=2, space=ImageSpace.FOURIER_COMPLEX)
        ),
        FlatteningIRLSFourier(
            IRLSSolver(unit_weights, max_iter=2, space=ImageSpace.FOURIER_REAL)
        ),
    ],
)
def test_fourier_irls_variants_with_unit_weights_reduce_to_sample_mean(estimator):
    images = make_small_images()
    batch = ImageBatch.from_real(images)
    expected = images.mean(dim=0)

    result = estimator.fit(batch)

    assert_valid_real_result(result, expected_shape=images.shape[1:])
    torch.testing.assert_close(result.average, expected, atol=1.0e-5, rtol=1.0e-5)

    reconstructed = estimator.reconstruct_from_weights(batch, result.weights)
    torch.testing.assert_close(reconstructed, expected, atol=1.0e-5, rtol=1.0e-5)


# ADMM couples real-space and Fourier-space updates, so this is a smoke-style
# estimator test: one iteration should complete and return finite outputs and
# weights in all relevant spaces.
def test_admm_solver_runs_and_returns_finite_average():
    images = make_small_images()
    batch = ImageBatch.from_real(images)

    real_solver = IRLSSolver(unit_weights, max_iter=2, space=ImageSpace.REAL)
    fourier_solver = JointIRLSFourier(
        IRLSSolver(unit_weights, max_iter=2, space=ImageSpace.FOURIER_COMPLEX)
    )
    admm = ADMMSolver(
        irls_real=real_solver,
        irls_fourier=fourier_solver,
        max_iter=1,
        initial_mu=1.0,
        fourier_multiplier=1.0,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    result = admm.fit(batch, verbose=False)

    assert result.average is not None
    assert result.average.shape == images.shape[1:]
    assert torch.isfinite(result.average).all()
    assert result.weights.real is not None
    assert result.weights.fourier_real is not None
    assert result.weights.fourier_imag is not None


# The GMM estimator is probabilistic internally, so the test fixes the random
# state and checks robust invariants: it returns a real estimate and responsibility
# weights that are finite probabilities in [0, 1].
def test_recursive_gmm_estimator_returns_valid_responsibility_weights():
    images = make_clustered_images()
    batch = ImageBatch.from_real(images)

    estimator = RecursiveGMMEstimator(
        distance_function=l2_norm,
        max_iter=2,
        tol=0.0,
        random_state=0,
    )
    result = estimator.fit(batch)

    assert_valid_real_result(result, expected_shape=images.shape[1:])
    assert result.weights.real is not None
    assert result.weights.real.shape == (images.shape[0], 1, 1)
    assert torch.isfinite(result.weights.real).all()
    assert torch.all(result.weights.real >= 0.0)
    assert torch.all(result.weights.real <= 1.0)
