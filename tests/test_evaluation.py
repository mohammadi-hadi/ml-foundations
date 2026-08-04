"""Verification for lesson 6.

Splitters cannot be compared to a reference by equality — two implementations shuffle
differently and there is no canonical partition. What can be checked is every property a
correct partition must have, plus agreement with scikit-learn on the part that is not
arbitrary: given the same folds, the aggregation must produce the same scores.

The calibration functions have no reference implementation in scikit-learn at all, so they are
checked against cases computed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as sklearn_cross_val_score

from ml_foundations import datasets as ds
from ml_foundations.evaluation import (
    cross_val_scores,
    expected_calibration_error,
    k_fold,
    reliability_curve,
    repeated_splits,
    select_k_best,
)
from ml_foundations.metrics import r2, roc_auc


def test_folds_partition_the_data_exactly_once() -> None:
    folds = k_fold(103, n_splits=5, seed=120)
    seen = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(np.sort(seen), np.arange(103))
    for train, test in folds:
        assert train.size + test.size == 103
        assert not set(train.tolist()) & set(test.tolist())


def test_fold_sizes_differ_by_at_most_one() -> None:
    for n_samples in (100, 101, 103, 7):
        sizes = [test.size for _, test in k_fold(n_samples, n_splits=5, seed=121)]
        assert max(sizes) - min(sizes) <= 1


def test_stratified_folds_keep_the_base_rate() -> None:
    data = ds.make_logistic(n_samples=1000, positive_rate=0.05, seed=122)
    for _, test in k_fold(data.n_samples, n_splits=5, seed=122, y=data.y):
        assert abs(float(data.y[test].mean()) - 0.05) < 0.02


def test_an_unstratified_fold_can_miss_a_rare_class_entirely() -> None:
    """Why the ``y`` argument exists. On a rare class this is not a corner case."""
    y = np.zeros(60)
    y[:2] = 1.0
    empty = 0
    for seed in range(30):
        for _, test in k_fold(60, n_splits=10, seed=seed):
            empty += int(y[test].sum() == 0)
    assert empty > 0
    # Stratification cannot conjure positives that do not exist — with two of them and ten
    # folds, eight folds still have none. What it guarantees is that they are spread as evenly
    # as they can be, which with two folds means exactly one each, every time.
    for seed in range(30):
        counts = [float(y[test].sum()) for _, test in k_fold(60, n_splits=2, seed=seed, y=y)]
        assert counts == [1.0, 1.0]


def test_folds_are_reproducible_and_seed_dependent() -> None:
    a = k_fold(50, seed=1)
    b = k_fold(50, seed=1)
    c = k_fold(50, seed=2)
    for (_, first), (_, second) in zip(a, b, strict=True):
        np.testing.assert_array_equal(first, second)
    assert any(
        first.tolist() != third.tolist() for (_, first), (_, third) in zip(a, c, strict=True)
    )


def test_cross_validation_aggregates_the_same_way_sklearn_does() -> None:
    """Given identical folds, the per-fold scores must match exactly."""
    data = ds.make_linear(n_samples=200, n_features=5, seed=124)
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    folds = [
        (train.astype(np.intp), test.astype(np.intp)) for train, test in splitter.split(data.X)
    ]

    def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        return SklearnRidge(alpha=1.0).fit(X_train, y_train).predict(X_test)

    mine = cross_val_scores(fit_predict, data.X, data.y, r2, folds)
    reference = sklearn_cross_val_score(SklearnRidge(alpha=1.0), data.X, data.y, cv=splitter)
    np.testing.assert_allclose(mine, reference, rtol=1e-10)


def test_select_k_best_picks_the_features_it_should() -> None:
    data = ds.make_sparse_linear(n_samples=400, n_features=30, n_informative=4, noise=0.5, seed=125)
    chosen = select_k_best(data.X, data.y, k=4)
    assert set(chosen.tolist()) == {0, 1, 2, 3}


def test_calibration_error_of_a_perfectly_calibrated_model_is_zero() -> None:
    """Twenty groups; in each, the observed frequency is exactly the claimed probability."""
    probability = np.repeat(np.linspace(0.025, 0.975, 20), 40)
    y = np.concatenate(
        [
            np.concatenate([np.ones(round(p * 40)), np.zeros(40 - round(p * 40))])
            for p in np.linspace(0.025, 0.975, 20)
        ]
    )
    assert expected_calibration_error(y, probability, n_bins=20) < 0.02


def test_calibration_error_matches_a_hand_computed_case() -> None:
    """Four predictions, two bins, everything checkable on paper.

    Bin [0, 0.5): claims 0.1 and 0.3 (mean 0.2), observes 0 and 1 (mean 0.5) — gap 0.3.
    Bin [0.5, 1]: claims 0.6 and 0.8 (mean 0.7), observes 1 and 1 (mean 1.0) — gap 0.3.
    Both bins hold half the data, so the weighted average is 0.3.
    """
    y = np.array([0.0, 1.0, 1.0, 1.0])
    probability = np.array([0.1, 0.3, 0.6, 0.8])
    assert expected_calibration_error(y, probability, n_bins=2) == pytest.approx(0.3)


def test_a_confident_liar_has_a_large_calibration_error_and_a_perfect_auc() -> None:
    """The pair of facts lesson 6 turns on: ranking and calibration are different questions."""
    rng = np.random.default_rng(126)
    y = (rng.random(2000) < 0.5).astype(np.float64)
    # Ranks perfectly, and claims near-certainty on every single case.
    probability = np.where(y == 1, 0.999, 0.001)
    assert roc_auc(y, probability) == pytest.approx(1.0)
    assert expected_calibration_error(y, probability) < 0.01

    # Same ranking, but every probability squashed towards one — still a perfect AUC.
    squashed = np.where(y == 1, 0.999, 0.95)
    assert roc_auc(y, squashed) == pytest.approx(1.0)
    assert expected_calibration_error(y, squashed) > 0.4


def test_reliability_curve_reports_one_entry_per_non_empty_bin() -> None:
    y = np.array([0.0, 1.0, 1.0, 1.0])
    probability = np.array([0.1, 0.3, 0.6, 0.8])
    curve = reliability_curve(y, probability, n_bins=2)
    assert curve == [(0.2, 0.5, 2), (pytest.approx(0.7), 1.0, 2)]


def test_reliability_bins_ascend() -> None:
    rng = np.random.default_rng(127)
    probability = rng.random(1000)
    y = (rng.random(1000) < probability).astype(np.float64)
    curve = reliability_curve(y, probability, n_bins=10)
    claimed = [point[0] for point in curve]
    assert claimed == sorted(claimed)
    assert sum(point[2] for point in curve) == 1000


def test_repeated_splits_are_disjoint_and_the_right_size() -> None:
    for train, test in repeated_splits(100, test_size=0.25, repeats=5, seed=128):
        assert test.size == 25
        assert train.size == 75
        assert not set(train.tolist()) & set(test.tolist())
