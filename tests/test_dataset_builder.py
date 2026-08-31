import numpy as np
import scipy

from cryo_robust.comparison.dataset_builder import generate_rotated_copies


def _generate_rotated_copies_reference(
    image: np.ndarray,
    n_copies: int,
    min_angle: float,
    max_angle: float,
    rng: np.random.Generator,
    interpolation_order: int = 3,
) -> np.ndarray:
    """Reference implementation using SciPy's automatic prefiltering."""
    output_images = np.zeros(
        (n_copies, image.shape[0], image.shape[1]),
        dtype=image.dtype,
    )

    angles = rng.uniform(min_angle, max_angle, size=n_copies)

    for i, angle in enumerate(angles):
        scipy.ndimage.rotate(
            image,
            angle,
            order=interpolation_order,
            reshape=False,
            output=output_images[i],
        )

    return output_images


def test_generate_rotated_copies_matches_reference_implementation():
    image_rng = np.random.default_rng(0)
    image = image_rng.normal(size=(32, 32)).astype(np.float32)

    reference_rng = np.random.default_rng(42)
    optimized_rng = np.random.default_rng(42)

    expected = _generate_rotated_copies_reference(
        image,
        n_copies=8,
        min_angle=-30.0,
        max_angle=30.0,
        rng=reference_rng,
    )
    actual = generate_rotated_copies(
        image,
        n_copies=8,
        min_angle=-30.0,
        max_angle=30.0,
        rng=optimized_rng,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == image.dtype
