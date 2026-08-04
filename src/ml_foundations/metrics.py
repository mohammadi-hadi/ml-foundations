"""Scoring functions, written out rather than imported, because the definition is the lesson.

Most of these are three lines. That is the point: a metric you cannot write from memory is a
metric whose failure modes you will not recognise, and every argument about model quality is
downstream of one of these definitions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ml_foundations.functions import log_sigmoid

Array = NDArray[np.float64]


def rmse(y_true: Array, y_pred: Array) -> float:
    """Root mean squared error, in the units of the target.

    Reported in preference to MSE throughout, for the sole reason that it is comparable to
    the noise level of the data — and knowing that a model has reached the noise floor is
    the difference between "improve it" and "stop".
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Array, y_pred: Array) -> float:
    """Mean absolute error: the same idea without the squaring, so outliers count once."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: Array, y_pred: Array) -> float:
    """Fraction of variance explained, relative to predicting the mean of ``y_true``.

    Zero means the model matched the intercept-only baseline. Negative means it did worse,
    which is possible on held-out data and is worth knowing about — a metric floored at zero
    would hide it.
    """
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total == 0.0:
        return 0.0
    return 1.0 - residual / total


def accuracy(y_true: Array, y_pred: Array) -> float:
    return float(np.mean(y_true == y_pred))


def confusion(y_true: Array, y_pred: Array) -> tuple[int, int, int, int]:
    """``(true negatives, false positives, false negatives, true positives)``."""
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    return true_negative, false_positive, false_negative, true_positive


def precision(y_true: Array, y_pred: Array) -> float:
    _, false_positive, _, true_positive = confusion(y_true, y_pred)
    denominator = true_positive + false_positive
    return float(true_positive / denominator) if denominator else 0.0


def recall(y_true: Array, y_pred: Array) -> float:
    _, _, false_negative, true_positive = confusion(y_true, y_pred)
    denominator = true_positive + false_negative
    return float(true_positive / denominator) if denominator else 0.0


def f1(y_true: Array, y_pred: Array) -> float:
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def log_loss(y_true: Array, scores: Array) -> float:
    """Mean negative log likelihood, taken from *logits* rather than from probabilities.

    Every other implementation you will meet accepts probabilities and clips them away from
    zero and one to avoid an infinite loss. The clip is a patch over an avoidable problem:
    the loss of a confidently wrong prediction is large but perfectly finite, and it is only
    the round trip through a probability that destroys it. Staying in log space keeps the
    gradient honest, which matters because this is the thing being optimised.
    """
    return float(-np.mean(y_true * log_sigmoid(scores) + (1 - y_true) * log_sigmoid(-scores)))


def brier(y_true: Array, probability: Array) -> float:
    """Mean squared error of a probability. Unlike log loss it stays finite when you are sure
    and wrong, which makes it the gentler of the two and the one that punishes overconfidence
    least — a property to remember before choosing it as the thing to optimise."""
    return float(np.mean((probability - y_true) ** 2))


def roc_auc(y_true: Array, scores: Array) -> float:
    """Area under the ROC curve, computed as the rank statistic it actually is.

    AUC is the probability that a randomly chosen positive outranks a randomly chosen
    negative. Written that way — average rank of the positives, shifted and scaled — it needs
    no curve, no thresholds and no trapezoids, and ties are handled by the average ranking
    rather than by a special case.
    """
    positive = y_true == 1
    n_positive = int(np.sum(positive))
    n_negative = int(y_true.shape[0] - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUC is undefined when one class is absent")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    ranks = _average_ties(scores, ranks)
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def _average_ties(scores: Array, ranks: Array) -> Array:
    """Give every member of a tied group the group's mean rank.

    Without this a tied block is ranked by whatever order the sort happened to produce, and
    a model that outputs a constant score would get an AUC anywhere between 0 and 1 depending
    on how the labels were shuffled. With it, that model scores exactly 0.5, which is true.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    start = 0
    for index in range(1, sorted_scores.shape[0] + 1):
        if index == sorted_scores.shape[0] or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def average_precision(y_true: Array, scores: Array) -> float:
    """Area under the precision-recall curve, as the step-wise sum used by every library.

    Reported alongside AUC on imbalanced problems in lesson 4, where the two disagree loudly
    about the same model.
    """
    order = np.argsort(-scores, kind="mergesort")
    labels = y_true[order]
    cumulative_true = np.cumsum(labels)
    precision_at_k = cumulative_true / np.arange(1, labels.shape[0] + 1)
    n_positive = float(np.sum(y_true))
    if n_positive == 0.0:
        raise ValueError("average precision is undefined with no positive examples")
    return float(np.sum(precision_at_k * labels) / n_positive)
