"""Verification tier A for the scoring functions: every one against scikit-learn's.

A metric that is subtly wrong is worse than a model that is subtly wrong, because it is what
you would use to find out.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as sk

from ml_foundations import metrics as mf
from ml_foundations.functions import sigmoid


@pytest.fixture
def binary() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(40)
    scores = rng.standard_normal(500) * 2.0
    y = (rng.random(500) < sigmoid(scores)).astype(np.float64)
    return y, scores


def test_regression_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(41)
    y_true = rng.standard_normal(300)
    y_pred = y_true + rng.standard_normal(300) * 0.5
    assert mf.rmse(y_true, y_pred) == pytest.approx(sk.root_mean_squared_error(y_true, y_pred))
    assert mf.mae(y_true, y_pred) == pytest.approx(sk.mean_absolute_error(y_true, y_pred))
    assert mf.r2(y_true, y_pred) == pytest.approx(sk.r2_score(y_true, y_pred))


def test_r2_is_zero_for_the_mean_and_negative_for_worse(binary: object) -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mf.r2(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert mf.r2(y, np.full(4, 100.0)) < 0.0


def test_classification_metrics_match_sklearn(binary: tuple[np.ndarray, np.ndarray]) -> None:
    y, scores = binary
    y_pred = (scores > 0).astype(np.float64)
    assert mf.accuracy(y, y_pred) == pytest.approx(sk.accuracy_score(y, y_pred))
    assert mf.precision(y, y_pred) == pytest.approx(sk.precision_score(y, y_pred))
    assert mf.recall(y, y_pred) == pytest.approx(sk.recall_score(y, y_pred))
    assert mf.f1(y, y_pred) == pytest.approx(sk.f1_score(y, y_pred))
    np.testing.assert_array_equal(
        np.array(mf.confusion(y, y_pred)), sk.confusion_matrix(y, y_pred).ravel()
    )


def test_ranking_metrics_match_sklearn(binary: tuple[np.ndarray, np.ndarray]) -> None:
    y, scores = binary
    assert mf.roc_auc(y, scores) == pytest.approx(sk.roc_auc_score(y, scores))
    assert mf.average_precision(y, scores) == pytest.approx(sk.average_precision_score(y, scores))


def test_probabilistic_metrics_match_sklearn(binary: tuple[np.ndarray, np.ndarray]) -> None:
    y, scores = binary
    probability = sigmoid(scores)
    assert mf.log_loss(y, scores) == pytest.approx(sk.log_loss(y, probability))
    assert mf.brier(y, probability) == pytest.approx(sk.brier_score_loss(y, probability))


def test_log_loss_from_logits_stays_finite_where_the_probability_form_does_not() -> None:
    """A confidently wrong prediction has a large loss, not an infinite one.

    Passing ``sigmoid(-900)`` to a probability-based implementation gives exactly zero, and
    its logarithm is ``-inf``; every library papers over this by clipping. Staying in log
    space needs no clip, and the gradient that comes out is the true one.
    """
    y = np.array([1.0])
    assert mf.log_loss(y, np.array([-900.0])) == pytest.approx(900.0)
    assert np.isfinite(mf.log_loss(y, np.array([-1e6])))


def test_auc_of_constant_scores_is_exactly_one_half() -> None:
    """Everything tied means no ranking information, whatever order the sort happened to use."""
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    assert mf.roc_auc(y, np.zeros(5)) == pytest.approx(0.5)


def test_auc_handles_partial_ties_the_way_sklearn_does() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    scores = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 1.0])
    assert mf.roc_auc(y, scores) == pytest.approx(sk.roc_auc_score(y, scores))


def test_auc_is_invariant_to_any_monotone_rescoring(
    binary: tuple[np.ndarray, np.ndarray],
) -> None:
    """It is a rank statistic. Calibrating the scores cannot change it — which is the reason
    lesson 4 needs a second metric to say anything about the probabilities themselves."""
    y, scores = binary
    assert mf.roc_auc(y, scores) == pytest.approx(mf.roc_auc(y, sigmoid(scores)))
    assert mf.roc_auc(y, scores) == pytest.approx(mf.roc_auc(y, scores * 7.0 - 3.0))


def test_ranking_metrics_refuse_a_single_class() -> None:
    with pytest.raises(ValueError):
        mf.roc_auc(np.zeros(5), np.arange(5.0))
    with pytest.raises(ValueError):
        mf.average_precision(np.zeros(5), np.arange(5.0))


def test_empty_denominators_do_not_divide_by_zero() -> None:
    y = np.array([1.0, 1.0])
    assert mf.precision(y, np.zeros(2)) == 0.0
    assert mf.f1(y, np.zeros(2)) == 0.0
