"""Comprehensive tests for metrics module."""

import numpy as np
import pytest

from ipt.metrics import (
    centroid_accuracy,
    deg_auprc,
    exclude_target_mask,
    paired_bootstrap,
    pcc_delta,
    pcc_delta_centered,
    pearson_rows,
    precision_at_k,
    rmse_delta,
    top_k_pcc,
)


class TestPearsonRows:
    """Test Pearson correlation per row."""

    def test_perfect_correlation(self):
        """Perfect correlation returns 1.0."""
        P, G = 5, 10
        a = np.random.RandomState(0).randn(P, G)
        b = a.copy()  # Perfect match
        result = pearson_rows(a, b)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_negative_perfect_correlation(self):
        """Perfect negative correlation returns -1.0."""
        P, G = 5, 10
        a = np.random.RandomState(0).randn(P, G)
        b = -a  # Perfect negative match
        result = pearson_rows(a, b)
        np.testing.assert_allclose(result, -1.0, rtol=1e-10)

    def test_constant_prediction_returns_zero(self):
        """Constant prediction (zero variance) returns 0.0, not NaN."""
        P, G = 5, 10
        a = np.ones((P, G))  # Constant
        b = np.random.RandomState(0).randn(P, G)
        result = pearson_rows(a, b)
        assert np.all(result == 0.0)
        assert not np.any(np.isnan(result))

    def test_both_constant_returns_zero(self):
        """When both arrays are constant, return 0.0."""
        P, G = 5, 10
        a = np.ones((P, G))
        b = np.ones((P, G)) * 2
        result = pearson_rows(a, b)
        assert np.all(result == 0.0)

    def test_mask_applied(self):
        """Mask restricts correlation computation to masked columns."""
        P, G = 2, 5
        a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])
        b = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])

        # Mask: use only first 3 columns for each row
        mask = np.array([[True, True, True, False, False], [True, True, True, False, False]])

        result = pearson_rows(a, b, mask)
        # Correlation on masked columns should be perfect
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)


class TestExcludeTargetMask:
    """Test target exclusion mask."""

    def test_single_target_excluded(self):
        """Target column marked as False."""
        target_idx = np.array([0, 2, 4])
        G = 5
        mask = exclude_target_mask(target_idx, G)
        assert mask.shape == (3, 5)
        assert mask[0, 0] == False and np.all(mask[0, 1:])
        assert mask[1, 2] == False and np.all(mask[1, [0, 1, 3, 4]])
        assert mask[2, 4] == False and np.all(mask[2, :4])

    def test_off_axis_target(self):
        """Off-axis target (-1) keeps all columns."""
        target_idx = np.array([0, -1, 2])
        G = 5
        mask = exclude_target_mask(target_idx, G)
        assert np.all(mask[1, :])  # All True when target_idx=-1
        assert mask[0, 0] == False
        assert mask[2, 2] == False


class TestPccDelta:
    """Test Pearson on deltas excluding target."""

    def test_perfect_prediction_excluding_target(self):
        """Perfect match excluding target gives pcc=1."""
        P, G = 5, 10
        pred = np.random.RandomState(0).randn(P, G)
        obs = pred.copy()
        target_idx = np.array([0, 1, 2, 3, 4])
        result = pcc_delta(pred, obs, target_idx)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_target_column_ignored(self):
        """Target column with huge values does not affect pcc."""
        P, G = 3, 8
        pred = np.random.RandomState(0).randn(P, G)
        obs = pred.copy()
        target_idx = np.array([0, 1, 2])

        # Add huge values to target columns in both pred and obs
        pred_modified = pred.copy()
        obs_modified = obs.copy()
        for p in range(P):
            pred_modified[p, target_idx[p]] = 1e6
            obs_modified[p, target_idx[p]] = 1e6

        # Should still get perfect correlation on non-target columns
        result = pcc_delta(pred_modified, obs_modified, target_idx)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_off_axis_targets_use_all_columns(self):
        """Off-axis targets use all columns."""
        P, G = 2, 5
        pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])
        obs = pred.copy()
        target_idx = np.array([-1, -1])
        result = pcc_delta(pred, obs, target_idx)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)


class TestPccDeltaCentered:
    """Test centered Pearson."""

    def test_train_mean_baseline_gives_zero_centered_pcc(self):
        """If pred == train_mean, centered pcc should be 0."""
        P, G = 5, 10
        obs = np.random.RandomState(0).randn(P, G)
        train_mean = np.mean(obs, axis=0)
        pred = np.tile(train_mean, (P, 1))  # pred is the mean
        target_idx = np.array([0, 1, 2, 3, 4])

        result = pcc_delta_centered(pred, obs, train_mean, target_idx)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_centering_removes_shared_component(self):
        """Centering removes shared systematic response."""
        P, G = 3, 6
        train_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        obs = np.array([
            [1.0, 1.5, 0.3, 0.7, 0.9, 1.1],
            [1.2, 1.3, 0.4, 0.6, 0.8, 1.2],
            [0.9, 1.7, 0.2, 0.8, 1.0, 1.0],
        ])
        pred = obs + train_mean[np.newaxis, :]  # pred has shared component
        target_idx = np.array([0, 1, 2])

        centered = pcc_delta_centered(pred, obs, train_mean, target_idx)
        # After centering, pred - train_mean aligns with obs - train_mean
        np.testing.assert_allclose(centered, 1.0, rtol=1e-10)


class TestTopKPcc:
    """Test top-k Pearson."""

    def test_top_k_selects_largest_absolute_obs(self):
        """Top-k should use genes with largest |obs|."""
        P, G = 2, 10
        # Row 0: obs varies, pred is random
        obs = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [10.0, 9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        pred = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [10.0, 9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        target_idx = np.array([-1, -1])
        k = 3

        result = top_k_pcc(pred, obs, target_idx, k=k)
        # For row 0: top 3 by |obs| are columns 5,6,7 -> perfect correlation
        # For row 1: top 3 by |obs| are columns 0,1,2 -> perfect correlation
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_top_k_with_centering(self):
        """Top-k with centering subtracts mean first."""
        P, G = 2, 8
        centered_mean = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5])
        obs = np.array([
            [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0],
        ])
        pred = obs.copy()
        target_idx = np.array([-1, -1])
        k = 4

        result = top_k_pcc(pred, obs, target_idx, k=k, centered_mean=centered_mean)
        # After centering both with varying means, pred==obs so correlation is perfect
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)


class TestRmseDelta:
    """Test RMSE excluding target."""

    def test_perfect_prediction_gives_zero_rmse(self):
        """Perfect prediction gives RMSE=0."""
        P, G = 5, 10
        pred = np.random.RandomState(0).randn(P, G)
        obs = pred.copy()
        target_idx = np.array([0, 1, 2, 3, 4])
        result = rmse_delta(pred, obs, target_idx)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_rmse_constant_offset(self):
        """RMSE with constant offset."""
        P, G = 2, 5
        pred = np.zeros((P, G))
        obs = np.ones((P, G))
        target_idx = np.array([0, 1])
        result = rmse_delta(pred, obs, target_idx)
        # RMSE should be 1.0 on non-target columns
        expected = np.array([1.0, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestDegAuprc:
    """Test DEG average precision."""

    def test_perfect_score_gives_ap_one(self):
        """Perfect ranking gives AP=1."""
        P, G = 2, 10
        deg_labels = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ])
        # Score: |pred| = absolute value; perfect order: DE genes first
        pred = np.array([
            [10.0, 9.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 10.0, 9.0, 8.0, 0.1, 0.1, 0.1],
        ])
        target_idx = np.array([-1, -1])

        result = deg_auprc(pred, deg_labels, target_idx)
        np.testing.assert_allclose(result["ap"], 1.0, rtol=1e-10)
        assert result["n_skipped"] == 0

    def test_all_negative_row_skipped(self):
        """Row with all zeros in labels is skipped."""
        P, G = 2, 8
        deg_labels = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],  # All negative
        ])
        pred = np.random.RandomState(0).randn(P, G)
        target_idx = np.array([-1, -1])

        result = deg_auprc(pred, deg_labels, target_idx)
        assert np.isnan(result["ap"][1])
        assert result["n_skipped"] == 1

    def test_all_positive_row_skipped(self):
        """Row with all ones in labels is skipped."""
        P, G = 2, 8
        deg_labels = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],  # All positive
        ])
        pred = np.random.RandomState(0).randn(P, G)
        target_idx = np.array([-1, -1])

        result = deg_auprc(pred, deg_labels, target_idx)
        assert np.isnan(result["ap"][1])
        assert result["n_skipped"] == 1

    def test_prevalence_computed(self):
        """Prevalence is positive fraction."""
        P, G = 2, 10
        deg_labels = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        pred = np.random.RandomState(0).randn(P, G)
        target_idx = np.array([-1, -1])

        result = deg_auprc(pred, deg_labels, target_idx)
        # Row 0: 3/10 = 0.3, Row 1: 1/10 = 0.1
        np.testing.assert_allclose(result["prevalence"], [0.3, 0.1], rtol=1e-10)

    def test_target_excluded(self):
        """Target column is excluded."""
        P, G = 2, 5
        deg_labels = np.array([
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
        ])
        pred = np.array([
            [10.0, 9.0, 0.1, 0.1, 0.1],
            [0.1, 8.0, 0.1, 0.1, 10.0],
        ])
        target_idx = np.array([0, 4])  # Exclude columns 0 and 4

        result = deg_auprc(pred, deg_labels, target_idx)
        # Target columns should be excluded from AP computation
        # Row 0: target 0 excluded, has 1 positive in columns [1,2,3,4]
        # Row 1: target 4 excluded, has 1 positive in columns [0,1,2,3]
        # Both should still compute AP on remaining genes
        assert not np.isnan(result["ap"][0])
        assert not np.isnan(result["ap"][1])


class TestPrecisionAtK:
    """Test precision@k."""

    def test_precision_at_k_perfect_ranking(self):
        """Perfect ranking gives precision@k=1."""
        P, G = 2, 10
        deg_labels = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ])
        pred = np.array([
            [10.0, 9.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 10.0, 9.0, 8.0, 0.1, 0.1, 0.1],
        ])
        target_idx = np.array([-1, -1])
        k = 2

        result = precision_at_k(pred, deg_labels, target_idx, k=k)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_precision_at_k_no_positives(self):
        """Row with no positives returns NaN."""
        P, G = 2, 8
        deg_labels = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        pred = np.random.RandomState(0).randn(P, G)
        target_idx = np.array([-1, -1])
        k = 3

        result = precision_at_k(pred, deg_labels, target_idx, k=k)
        assert not np.isnan(result[0])
        assert np.isnan(result[1])


class TestCentroidAccuracy:
    """Test centroid accuracy (cross-perturbation)."""

    def test_perfect_prediction_gives_top1_and_rank_one(self):
        """Perfect pred==obs gives top1=1, norm_rank=1."""
        P, G = 10, 20
        np.random.seed(42)
        obs = np.random.randn(P, G)
        pred = obs.copy()
        train_mean = np.zeros(G)
        target_idx = np.full(P, -1)

        result = centroid_accuracy(pred, obs, train_mean, target_idx)
        assert result["top1"] == 1.0
        assert np.isclose(result["norm_rank"], 1.0, rtol=1e-10)
        np.testing.assert_allclose(result["norm_rank_each"], 1.0, rtol=1e-10)

    def test_shuffled_rows_gives_chance_level_norm_rank(self):
        """Shuffled rows should give norm_rank near 0.5."""
        P, G = 200, 50
        np.random.seed(42)
        obs = np.random.randn(P, G)
        shuffled_idx = np.random.permutation(P)
        pred = obs[shuffled_idx, :]  # Shuffled version
        train_mean = np.zeros(G)
        target_idx = np.full(P, -1)

        result = centroid_accuracy(pred, obs, train_mean, target_idx)
        # Should be near chance level (0.5) with some tolerance
        assert 0.42 < result["norm_rank"] < 0.58, f"norm_rank={result['norm_rank']}"

    def test_excludes_all_target_genes(self):
        """All target genes excluded from columns used."""
        P, G = 5, 10
        np.random.seed(42)
        obs = np.random.randn(P, G)
        pred = obs.copy()
        train_mean = np.zeros(G)
        target_idx = np.array([0, 1, 2, 3, 4])  # Each perturb different target

        result = centroid_accuracy(pred, obs, train_mean, target_idx)
        # Even though pred==obs, target genes excluded means correlation computed only on 5/10 genes
        # but should still be perfect because pred and obs are identical
        assert result["top1"] == 1.0
        assert np.isclose(result["norm_rank"], 1.0, rtol=1e-10)


class TestPairedBootstrap:
    """Test bootstrap confidence intervals and p-values."""

    def test_identical_arrays_gives_zero_diff(self):
        """Identical a and b give mean_diff=0, CI contains 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a.copy()
        result = paired_bootstrap(a, b, n_boot=5000, seed=0)

        assert np.isclose(result["mean_diff"], 0.0, atol=1e-10)
        assert result["ci_low"] <= 0.0 <= result["ci_high"]
        assert result["win_frac"] == 0.0  # a is never > b when a == b

    def test_constant_difference_excludes_zero(self):
        """a = b + 0.5 gives CI excluding 0, p < 0.01."""
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = b + 0.5
        result = paired_bootstrap(a, b, n_boot=5000, seed=0)

        assert np.isclose(result["mean_diff"], 0.5, atol=1e-10)
        assert result["ci_low"] > 0.0  # CI doesn't contain 0
        assert result["p_two_sided"] < 0.01

    def test_nan_pairs_ignored(self):
        """NaN pairs are ignored in bootstrap."""
        a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        result = paired_bootstrap(a, b, n_boot=1000, seed=0)

        # Only 3 valid pairs: (1,1), (2,2), (5,5)
        assert result["n"] == 3
        assert np.isclose(result["mean_diff"], 0.0, atol=1e-10)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_workflow_realistic_scenario(self):
        """Test typical workflow: predict, mask, compute metrics."""
        P, G = 10, 100
        np.random.seed(42)

        # Simulate training and test data
        train_obs = np.random.randn(P, G) * 0.3
        train_mean = np.mean(train_obs, axis=0)

        # Test data: predictions with some systematic bias
        test_obs = np.random.randn(P, G) * 0.5 + 0.1
        test_pred = test_obs + 0.2  # Small systematic bias

        target_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Compute metrics
        pcc = pcc_delta(test_pred, test_obs, target_idx)
        pcc_c = pcc_delta_centered(test_pred, test_obs, train_mean, target_idx)
        rmse = rmse_delta(test_pred, test_obs, target_idx)

        # Check shapes
        assert pcc.shape == (P,)
        assert pcc_c.shape == (P,)
        assert rmse.shape == (P,)

        # Check ranges
        assert np.all(np.abs(pcc) <= 1.0)
        assert np.all(np.abs(pcc_c) <= 1.0)
        assert np.all(rmse >= 0)


def test_bootstrap_p_matches_ci_consistency():
    from ipt.metrics import paired_bootstrap
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 60); b = rng.normal(0.0, 1.0, 60)
    r = paired_bootstrap(a, b, n_boot=4000, seed=0)
    assert (r["ci_low"] <= 0 <= r["ci_high"]) == (r["p_two_sided"] > 0.05) or abs(r["p_two_sided"] - 0.05) < 0.03
    r2 = paired_bootstrap(a + 1.0, b, n_boot=4000, seed=0)
    assert r2["p_two_sided"] < 0.01


def test_pcc_resid_removes_magnitude_only_signal():
    from ipt.metrics import pcc_delta_resid
    rng = np.random.default_rng(0)
    G = 300; tm = rng.normal(size=G); spec = rng.normal(size=(40, G))
    obs = 0.3 * tm[None, :] + spec; tgt = np.full(40, -1)
    pred_mag = 1.0 * tm[None, :].repeat(40, 0)           # only the shared direction
    pred_spec = spec + 0.0 * tm
    assert abs(pcc_delta_resid(pred_mag, obs, tm, tgt).mean()) < 0.05
    assert pcc_delta_resid(pred_spec, obs, tm, tgt).mean() > 0.9
