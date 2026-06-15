# These tests are intentionally small and deterministic.  They do not try to
# validate the full statistical behavior of every robust function; instead,
# they check basic invariants that should remain true after refactors: expected
# shapes, finite outputs, sensible ranges, and simple values that can be
# verified by hand.

import pytest
import torch

from cryo_robust.estimators.weights import (
    weighted_average,
    huber_weights,
    smooth_redescending_weights,
    tagare_weights,
    cosine_similarity,
    cross_correlation,
    cc_tagare_weights,
    cauchy_weights,
    student_weights,
    q_norm_weights,
    get_weight_function,
)
from cryo_robust.estimators.distances import (
    l1_norm,
    l2_norm,
    lp_norm,
    orthogonal_residual_norm,
    invert_similarity,
    get_distance_function,
)


# Shared tiny fixture: one image equal to the reference, one orthogonal image,
# and one scaled copy.  This lets us test alignment-based similarities without
# relying on random data or visual inspection.
def _reference_and_images():
    reference = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    orthogonal = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    images = torch.stack(
        [
            reference,
            orthogonal,
            2.0 * reference,
        ]
    )
    return reference, images


# Basic arithmetic sanity check for the weighted average implementation.
# With weights 1 and 3 over values 0 and 2, the result should be 1.5.
def test_weighted_average_matches_manual_result():
    images = torch.tensor([[[0.0]], [[2.0]]])
    weights = torch.tensor([[[1.0]], [[3.0]]])

    avg = weighted_average(images, weights, eps=0.0)

    torch.testing.assert_close(avg, torch.tensor([[1.5]]))


# Degenerate weights are almost always a sign that something upstream failed.
# The implementation is expected to reject this case instead of silently
# returning NaNs or an arbitrary image.
def test_weighted_average_rejects_degenerate_weights():
    images = torch.ones(2, 2, 2)
    weights = torch.zeros(2, 1, 1)

    with pytest.raises(ValueError):
        weighted_average(images, weights)


# Pixelwise robust weights should behave like masks over the image grid: same
# shape as the input images, finite values, and no negative weights.
@pytest.mark.parametrize(
    "weight_fn, kwargs",
    [
        (huber_weights, {"delta": 1.0}),
        (smooth_redescending_weights, {"delta": 1.0}),
        (cauchy_weights, {"c": 1.0}),
        (student_weights, {"df": 3.0}),
        (q_norm_weights, {"q": 1.5}),
    ],
)
def test_pixelwise_weight_functions_return_finite_nonnegative_weights(weight_fn, kwargs):
    reference, images = _reference_and_images()

    weights = weight_fn(images, reference, std=1.0, **kwargs)

    assert weights.shape == images.shape
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()


# Non-local weights/similarities return one scalar per image.  This test checks
# their broadcasting shape and verifies that a perfectly aligned image scores
# better than an orthogonal one for the Tagare-style similarities.
def test_nonlocal_weight_functions_have_expected_shapes_and_ranges():
    reference, images = _reference_and_images()

    tagare = tagare_weights(images, reference, beta=1.0e-6)
    cosine = cosine_similarity(images, reference)
    corr = cross_correlation(images, reference)
    cc_tagare = cc_tagare_weights(images, reference, beta=1.0e-6)

    for weights in [tagare, cosine, corr, cc_tagare]:
        assert weights.shape == (images.shape[0], 1, 1)
        assert torch.isfinite(weights).all()

    # Exact alignment should score higher than the orthogonal image for Tagare weights.
    assert tagare[0, 0, 0] > tagare[1, 0, 0]
    # Not necessarily for cross-correlation Tagare, because images are pre-centered
    assert cc_tagare[0, 0, 0] >= cc_tagare[1, 0, 0]

    # Cosine/correlation similarities should remain within their natural range.
    assert torch.all(cosine.abs() <= 1.0 + 1.0e-6)
    assert torch.all(corr.abs() <= 1.0 + 1.0e-6)


# For constant offsets, L1, L2 and Lp with p=2 have simple closed-form expected
# values.  This protects the normalization convention used by the distance code.
def test_l1_l2_and_lp_distances_match_simple_manual_values():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    images = torch.stack(
        [
            reference,
            reference + 1.0,
            reference - 2.0,
        ]
    )

    expected_l1 = torch.tensor([0.0, 1.0, 2.0])
    expected_l2 = torch.tensor([0.0, 1.0, 2.0])

    torch.testing.assert_close(l1_norm(images, reference), expected_l1)
    torch.testing.assert_close(l2_norm(images, reference), expected_l2)
    torch.testing.assert_close(lp_norm(images, reference, p=2.0), expected_l2)


# Any scaled copy of the reference lies in the one-dimensional subspace spanned
# by the reference, so its orthogonal residual should be zero.
def test_orthogonal_residual_norm_is_zero_for_scaled_reference():
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    images = torch.stack([reference, 2.0 * reference, -3.0 * reference])

    residual = orthogonal_residual_norm(images, reference)

    torch.testing.assert_close(residual, torch.zeros(3), atol=1.0e-6, rtol=1.0e-6)


# The distance module sometimes converts similarities into dissimilarities.  The
# non-inplace path is tested explicitly so callers can safely reuse the original
# similarity tensor after inversion.
def test_invert_similarity_does_not_modify_input_when_inplace_false():
    similarity = torch.tensor([2.0, 4.0])

    reciprocal = invert_similarity(similarity, inv_type="reciprocal", inplace=False)
    negative = invert_similarity(similarity, inv_type="neg", inplace=False)

    torch.testing.assert_close(reciprocal, torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(negative, torch.tensor([-2.0, -4.0]))
    torch.testing.assert_close(similarity, torch.tensor([2.0, 4.0]))


# Registry tests check that configuration names used in experiment files still
# resolve to callable functions, including automatic beta scaling for Tagare
# weights.
def test_weight_and_distance_registries_return_configured_functions():
    reference, images = _reference_and_images()

    weight_fn = get_weight_function("global", {"beta": "auto"}, imgs=images)
    distance_fn = get_distance_function("l2", {})

    weights = weight_fn(images, reference, std=1.0)
    distances = distance_fn(images, reference)

    assert weights.shape == (images.shape[0], 1, 1)
    assert distances.shape == (images.shape[0],)
    assert torch.isfinite(weights).all()
    assert torch.isfinite(distances).all()


# Unknown registry entries should fail clearly.  This is useful because typos in
# YAML/TOML experiment configs would otherwise be hard to diagnose.
def test_unknown_registry_entries_raise_clear_errors():
    with pytest.raises(ValueError):
        get_weight_function("not_a_weight", {})

    with pytest.raises(ValueError):
        get_distance_function("not_a_distance", {})