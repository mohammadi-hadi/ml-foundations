"""Splitting, scoring, and measuring whether a probability means anything.

Every lesson before this one reported a held-out number and asked you to believe it. This
module is the machinery for deciding whether such a number deserves belief, and it is short —
which is the point. Nothing here is difficult. The mistakes it guards against are mistakes of
*procedure*, not of arithmetic, and they are invisible in the output: a leaky pipeline does
not raise an exception, it reports an excellent score.

The single rule everything else follows from: **anything fitted to data must be fitted inside
the fold.** Scalers, feature selection, imputation, the choice of hyperparameter, the choice of
threshold — all of it. A step that looks at the held-out rows, even without looking at their
labels, has put information into the model that will not be there at prediction time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IndexArray = NDArray[np.intp]
Fold = tuple[IndexArray, IndexArray]


def k_fold(
    n_samples: int,
    *,
    n_splits: int = 5,
    seed: int = 0,
    y: Array | None = None,
) -> list[Fold]:
    """Partition the rows into ``n_splits`` folds, returning ``(train, test)`` index pairs.

    Passing ``y`` stratifies: each fold then holds roughly the same proportion of each class
    as the whole. On a problem with a 2% positive rate and five folds, an unstratified split
    can hand a fold no positives at all, at which point every metric computed on it is either
    undefined or a lie.
    """
    rng = np.random.default_rng(seed)
    assignment = np.empty(n_samples, dtype=np.intp)
    if y is None:
        order = rng.permutation(n_samples)
        assignment[order] = np.arange(n_samples) % n_splits
    else:
        for value in np.unique(y):
            members = np.flatnonzero(y == value)
            rng.shuffle(members)
            assignment[members] = np.arange(members.size) % n_splits

    folds: list[Fold] = []
    for split in range(n_splits):
        test = np.flatnonzero(assignment == split).astype(np.intp)
        train = np.flatnonzero(assignment != split).astype(np.intp)
        folds.append((train, test))
    return folds


def cross_val_scores(
    fit_predict: Callable[[Array, Array, Array], Array],
    X: Array,
    y: Array,
    score: Callable[[Array, Array], float],
    folds: list[Fold],
) -> list[float]:
    """Score each fold, given a function that trains on one part and predicts on another.

    ``fit_predict`` takes ``(X_train, y_train, X_test)`` and returns predictions, rather than
    taking a fitted model. That signature is deliberate: it makes it impossible to pass in
    something already fitted on all the data, which is the most common way this goes wrong.
    Everything a pipeline does has to happen inside the callback, where it only sees the
    training rows.
    """
    return [score(y[test], fit_predict(X[train], y[train], X[test])) for train, test in folds]


def nested_scores(
    fit_predict: Callable[[Array, Array, Array, list[Fold]], Array],
    X: Array,
    y: Array,
    score: Callable[[Array, Array], float],
    *,
    n_outer: int = 5,
    n_inner: int = 4,
    seed: int = 0,
    stratify: bool = False,
) -> list[float]:
    """Cross-validation with the model choice made inside each outer fold.

    ``fit_predict`` receives the outer training data *and* a set of inner folds carved out of
    it, and is expected to choose whatever it needs to choose using only those. The outer test
    rows are then untouched by every decision, including the decision of which model to use —
    which is the whole difference between an estimate of a model and an estimate of a
    *procedure*.
    """
    outer = k_fold(X.shape[0], n_splits=n_outer, seed=seed, y=y if stratify else None)
    results = []
    for train, test in outer:
        inner = k_fold(
            train.size, n_splits=n_inner, seed=seed + 1, y=y[train] if stratify else None
        )
        results.append(score(y[test], fit_predict(X[train], y[train], X[test], inner)))
    return results


def expected_calibration_error(y_true: Array, probability: Array, *, n_bins: int = 10) -> float:
    """Average gap between claimed confidence and observed frequency, weighted by bin size.

    A model that says 0.8 should be right about 80% of the time. This bins the predictions by
    claimed probability and measures how far each bin's actual frequency sits from the
    probability it claimed, which is the only way to catch a model that ranks perfectly and
    lies about how sure it is.

    It is a coarse instrument and the bin count changes the answer, so it is reported
    alongside the reliability curve it summarises rather than on its own.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so that a prediction of exactly 1.0 lands in the last one rather than
    # in a bin of its own past the end.
    index = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        members = index == b
        if not members.any():
            continue
        gap = abs(float(y_true[members].mean()) - float(probability[members].mean()))
        total += gap * float(members.sum())
    return total / float(y_true.shape[0])


def reliability_curve(
    y_true: Array, probability: Array, *, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """``(mean claimed probability, observed frequency, count)`` for each non-empty bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1)
    curve = []
    for b in range(n_bins):
        members = index == b
        if not members.any():
            continue
        curve.append(
            (
                float(probability[members].mean()),
                float(y_true[members].mean()),
                int(members.sum()),
            )
        )
    return curve


def select_k_best(X: Array, y: Array, *, k: int) -> IndexArray:
    """The ``k`` features most correlated with the target, by absolute correlation.

    Harmless in itself. The trouble is entirely in *what data you call it on* — lesson 6 calls
    it once on everything and once inside the fold, and the difference between the two answers
    is the largest number in this repository.
    """
    centred = X - X.mean(axis=0)
    target = y - y.mean()
    scale = np.sqrt(np.einsum("ij,ij->j", centred, centred)) * float(np.sqrt(target @ target))
    correlation = np.abs(centred.T @ target) / np.where(scale == 0.0, 1.0, scale)
    return np.argsort(-correlation, kind="mergesort")[:k].astype(np.intp)


def repeated_splits(
    n_samples: int, *, test_size: float = 0.3, repeats: int = 50, seed: int = 0
) -> Iterator[Fold]:
    """Independent random train/test splits, for measuring how much one split is worth."""
    rng = np.random.default_rng(seed)
    n_test = round(n_samples * test_size)
    for _ in range(repeats):
        order = rng.permutation(n_samples)
        yield order[n_test:].astype(np.intp), order[:n_test].astype(np.intp)
