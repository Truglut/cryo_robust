# These tests focus on the basic scalar metrics used in the reports.  They avoid
# Fourier ring/FRC validation and instead use small cases where precision,
# recall, AP, ROC-AUC, RMSE and Pearson correlation have obvious expected values.

import numpy as np
import pytest
import torch

from cryo_robust.comparison.domain.enums import AggregationStrategy, ImageSpace
from cryo_robust.comparison.evaluation.classification_metrics import (
    get_precision,
    get_recall,
    compute_soft_metrics,
    compute_classification_metrics,
)
from cryo_robust.comparison.evaluation.reconstruction_metrics import (
    compute_reconstruction_metrics,
    get_half_set_indices,
)
from cryo_robust.estimators.data import ImageBatch


# Minimal estimator used to exercise compute_reconstruction_metrics without
# depending on any robust-estimator internals.  It implements only the methods
# needed by that metric function.
class MeanEstimatorForMetricTest:
    """Minimal estimator used only to exercise reconstruction metric formulas."""

    def __init__(self):
        self.avg = None

    def fit_tensor(self, images):
        self.avg = images[ImageSpace.REAL].mean(dim=0)
        return self.avg

    def reconstruct_from_weights(self, images, weights):
        return images[ImageSpace.REAL].mean(dim=0)


# Soft precision is the fraction of total score assigned to inliers.  The recall
# variants use different normalizations, so this simple vector checks the two
# most important conventions explicitly.
def test_get_precision_and_recall_on_simple_soft_scores():
    scores = np.array([1.0, 0.9, 0.1, 0.0])
    idx_good = np.array([True, True, False, False])

    assert get_precision(scores, idx_good) == pytest.approx(0.95)
    assert get_recall(scores, idx_good, "huang_tagare") == pytest.approx(0.95)
    assert get_recall(scores, idx_good, "global_avg") == pytest.approx(1.0)


# In this ranking, both inliers receive higher scores than both outliers.  AP and
# ROC-AUC should therefore be exactly one, while the soft metrics keep their
# score-weighted values.
def test_compute_soft_metrics_for_perfect_ranking():
    scores = np.array([1.0, 0.9, 0.1, 0.0])
    idx_good = np.array([True, True, False, False])

    metrics = compute_soft_metrics(
        scores=scores,
        idx_good=idx_good,
        recall_methods=["huang_tagare", "global_avg"],
    )

    assert metrics.ap == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.soft_precision == pytest.approx(0.95)
    assert metrics.soft_recall["huang_tagare"] == pytest.approx(0.95)
    assert metrics.soft_recall["global_avg"] == pytest.approx(1.0)


# The higher-level classification function should not only compute correct
# values, but also preserve the nested organization by image space and
# aggregation strategy used later by the report code.
def test_compute_classification_metrics_preserves_space_and_strategy_keys():
    scores = np.array([1.0, 0.9, 0.1, 0.0])
    labels = np.array([0, 0, 1, 1])
    aggregated_weights = {
        ImageSpace.REAL: {
            AggregationStrategy.MEAN: scores,
        }
    }

    result = compute_classification_metrics(
        agg_weights=aggregated_weights,
        labels=labels,
        recall_methods=["huang_tagare"],
    )

    metrics = result[ImageSpace.REAL][AggregationStrategy.MEAN]
    assert metrics.ap == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.soft_precision == pytest.approx(0.95)


# Half-set splits are used for reproducible reconstruction comparisons.  The
# same seed should give the same split, and both halves together should contain
# every index exactly once.
def test_half_set_indices_are_reproducible_and_form_a_partition():
    idx_a_1, idx_b_1 = get_half_set_indices(8, seed=123, device="cpu")
    idx_a_2, idx_b_2 = get_half_set_indices(8, seed=123, device="cpu")

    torch.testing.assert_close(idx_a_1, idx_a_2)
    torch.testing.assert_close(idx_b_1, idx_b_2)

    all_indices = torch.cat([idx_a_1, idx_b_1]).sort().values
    torch.testing.assert_close(all_indices, torch.arange(8))
    assert set(idx_a_1.tolist()).isdisjoint(set(idx_b_1.tolist()))


# The estimated image is ground truth plus a constant offset of one, so RMSE is
# one.  Adding a constant does not change Pearson correlation, so correlation is
# one as well.  The FRC objects are only checked for existence/finite summaries;
# detailed FRC behavior is intentionally outside this basic test file.
def test_reconstruction_metrics_compute_basic_rmse_and_correlation():
    ground_truth = (np.arange(16, dtype=np.float32).reshape(4, 4) / 10.0) + 1.0
    estimated = ground_truth + 1.0

    images = torch.stack(
        [
            torch.from_numpy(ground_truth - 0.1),
            torch.from_numpy(ground_truth + 0.1),
            torch.from_numpy(ground_truth - 0.2),
            torch.from_numpy(ground_truth + 0.2),
        ]
    ).float()
    image_batch = ImageBatch.from_real(images)
    images_dict = image_batch.as_space_dict()
    split_indices = (torch.tensor([0, 1]), torch.tensor([2, 3]))

    metrics, gt_frc_data, hs_frc_data = compute_reconstruction_metrics(
        ground_truth_img=ground_truth,
        estimated_img=estimated,
        frc_thresholds=[],
        images_dict=images_dict,
        estimator=MeanEstimatorForMetricTest(),
        weights={space: None for space in ImageSpace},
        split_indices=split_indices,
        pixel_size=1.0,
        reapply_mask=False,
        independent_half_sets=True,
    )

    assert metrics.rmse == pytest.approx(1.0)
    assert metrics.pearson_corr == pytest.approx(1.0)
    assert gt_frc_data is not None
    assert hs_frc_data is not None
    assert np.isfinite(metrics.gt_aufrc)
    assert np.isfinite(metrics.hs_aufrc)