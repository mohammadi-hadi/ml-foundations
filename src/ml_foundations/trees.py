"""Decision trees, grown greedily one split at a time.

A tree asks a sequence of yes-or-no questions about a single feature at a time and predicts a
constant in each region it carves out. Two properties follow immediately and are worth naming
before any code: it can represent interactions and nonlinearity that a linear model cannot,
and it is invariant to any monotone transformation of a feature, because only the *order* of
the values affects which splits are possible. No standardising, no log transforms.

The cost is that a deep tree memorises. Lesson 5 measures both.

**One implementation, two criteria.** Regression by squared error and binary classification by
the Gini index need the same three numbers from each candidate side of a split: how many rows,
their sum, and their sum of squares. Because the labels are 0 and 1 in the classification case,
``sum(y²) = sum(y)``, and the Gini index falls out of the same prefix sums. So the split search
below is written once and the criterion only changes one line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Criterion = Literal["mse", "gini"]

#: Candidate splits closer than this — relative to the impurity being reduced — count as tied.
#: See :meth:`DecisionTree._best_split` for why a greedy tree needs this to be deterministic.
TIE_RESOLUTION = 1e-9


def _quantise(value: Array | np.float64, quantum: float) -> Array | np.float64:
    """Round to a multiple of ``quantum``, so that near-ties become exact ties."""
    if quantum <= 0.0:
        return value
    return np.round(value / quantum) * quantum


@dataclass
class Node:
    """A leaf if ``feature`` is None, otherwise an internal split."""

    value: float
    n_samples: int
    feature: int | None = None
    threshold: float = 0.0
    left: Node | None = None
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _impurity_total(criterion: Criterion, count: Array, total: Array, total_sq: Array) -> Array:
    """Impurity times the number of rows, which is the quantity that adds across a split.

    Working with the total rather than the average is what makes a split's score a plain sum
    of its two sides. Squared error is ``Σy² - (Σy)²/n``; the Gini index of a binary node is
    ``2p(1-p)``, and multiplied by ``n`` that is ``2·Σy·(n - Σy)/n``.
    """
    safe = np.where(count == 0, 1.0, count)
    if criterion == "mse":
        return total_sq - total**2 / safe
    return 2.0 * total * (count - total) / safe


class DecisionTree:
    """CART, grown greedily, split by exhaustive search over midpoints.

    Every candidate split of every feature is scored. That is O(n log n) per feature per node
    rather than the cleverer incremental schemes a production implementation uses, and it is
    written this way because the search is the algorithm — a tree is nothing but this loop,
    repeated.
    """

    def __init__(
        self,
        *,
        criterion: Criterion = "mse",
        max_depth: int = 8,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_features: int | None = None,
        seed: int = 0,
    ) -> None:
        self.criterion: Criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.root_: Node = Node(value=0.0, n_samples=0)

    def fit(self, X: Array, y: Array) -> Self:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: Array, y: Array, *, depth: int) -> Node:
        node = Node(value=float(y.mean()), n_samples=y.shape[0])
        if depth >= self.max_depth or y.shape[0] < self.min_samples_split or y.std() == 0.0:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold = split
        mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[mask], y[mask], depth=depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth=depth + 1)
        return node

    def _candidate_features(self, n_features: int) -> NDArray[np.intp]:
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features, dtype=np.intp)
        chosen = self._rng.choice(n_features, size=self.max_features, replace=False)
        return np.asarray(chosen, dtype=np.intp)

    def _best_split(self, X: Array, y: Array) -> tuple[int, float] | None:
        n_samples = y.shape[0]
        best_score = _impurity_total(
            self.criterion,
            np.array([float(n_samples)]),
            np.array([float(y.sum())]),
            np.array([float((y**2).sum())]),
        )[0]
        # Candidate scores are quantised before they are compared. Two splits are often equally
        # good to within the last bits of a floating-point sum, and which one then wins depends
        # on how numpy blocked the addition — which depends on the machine. A greedy tree
        # amplifies that: one flipped split at the root gives a completely different subtree, so
        # a model that should be a deterministic function of the data becomes a function of the
        # hardware. Rounding to a part in a billion of the parent's impurity is far below any
        # difference that means anything and far above any that does not, and ties then resolve
        # by position: the first feature and the lowest threshold win, on every machine.
        quantum = abs(float(best_score)) * TIE_RESOLUTION
        best_score = _quantise(best_score, quantum)
        best: tuple[int, float] | None = None

        for feature in self._candidate_features(X.shape[1]):
            order = np.argsort(X[:, feature], kind="mergesort")
            values = X[order, feature]
            targets = y[order]

            left_count = np.arange(1, n_samples, dtype=np.float64)
            left_total = np.cumsum(targets)[:-1]
            left_total_sq = np.cumsum(targets**2)[:-1]
            right_count = n_samples - left_count
            right_total = float(targets.sum()) - left_total
            right_total_sq = float((targets**2).sum()) - left_total_sq

            score = _impurity_total(
                self.criterion, left_count, left_total, left_total_sq
            ) + _impurity_total(self.criterion, right_count, right_total, right_total_sq)

            # A split is only allowed between two *different* values, and only if both sides
            # keep enough rows. Everything else is masked out rather than skipped, so the
            # whole feature is still scored in one vectorised pass.
            allowed = (values[:-1] < values[1:]) & (
                (left_count >= self.min_samples_leaf) & (right_count >= self.min_samples_leaf)
            )
            score = np.where(allowed, _quantise(score, quantum), np.inf)
            # argmin returns the *first* minimum, so among quantised ties the lowest threshold
            # wins; the strict comparison below then keeps the earliest feature.
            position = int(np.argmin(score))
            if score[position] < best_score:
                best_score = float(score[position])
                best = (int(feature), float((values[position] + values[position + 1]) / 2.0))
        return best

    def predict(self, X: Array) -> Array:
        """Send every row down the tree at once, rather than one row at a time.

        The obvious implementation walks each row from the root to its leaf, and it is a
        Python loop over rows — which turns a millisecond of arithmetic into seconds of
        interpreter. This carries a *set* of rows down each branch instead: at each node the
        set splits in two, and each leaf assigns its value to whichever rows arrived. The
        same comparisons happen, in one vectorised operation per node instead of one per row.
        """
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.float64)
        pending: list[tuple[Node, NDArray[np.intp]]] = [(self.root_, np.arange(X.shape[0]))]
        while pending:
            node, rows = pending.pop()
            if rows.size == 0:
                continue
            if node.is_leaf:
                out[rows] = node.value
                continue
            assert node.feature is not None and node.left is not None and node.right is not None
            goes_left = X[rows, node.feature] <= node.threshold
            pending.append((node.left, rows[goes_left]))
            pending.append((node.right, rows[~goes_left]))
        return out

    def predict_proba(self, X: Array) -> Array:
        """Only meaningful for ``criterion='gini'``: the class-one proportion in each leaf."""
        return self.predict(X)

    @property
    def n_leaves(self) -> int:
        def count(node: Node) -> int:
            if node.is_leaf:
                return 1
            assert node.left is not None and node.right is not None
            return count(node.left) + count(node.right)

        return count(self.root_)

    @property
    def depth(self) -> int:
        def measure(node: Node) -> int:
            if node.is_leaf:
                return 0
            assert node.left is not None and node.right is not None
            return 1 + max(measure(node.left), measure(node.right))

        return measure(self.root_)
