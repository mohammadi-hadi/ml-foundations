"""Many trees instead of one, in the two ways that are not the same thing.

**Bagging and random forests average away variance.** Each tree is grown on a bootstrap
resample, so each is wrong in its own direction, and the average of many independent errors is
smaller than any of them. The bias is unchanged — the average of unbiased-ish deep trees is
still unbiased-ish — which means this only helps a model whose problem is variance. Lesson 5
measures the bias and the variance separately to show which one moved.

Averaging ``B`` estimators with variance ``σ²`` and pairwise correlation ``ρ`` leaves
``ρσ² + (1-ρ)σ²/B``. The second term vanishes with enough trees; the first does not. That is
the entire argument for a random forest over plain bagging: restricting each split to a random
subset of features makes the trees less alike, which lowers ``ρ``, which lowers the floor.

**Boosting reduces bias.** Each tree is fitted to what the ensemble so far got wrong, so the
ensemble is a sum rather than an average and the trees are deliberately weak. It does not
converge to a limit as more trees are added — it keeps fitting, eventually to the noise, so
where to stop is a decision and lesson 5 measures what it is worth.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from ml_foundations.trees import DecisionTree

Array = NDArray[np.float64]


class BaggedTrees:
    """Average of trees grown on bootstrap resamples.

    ``max_features`` is what turns this into a random forest: with it set, each split may only
    consider a random subset of the features, so two trees that would otherwise both split on
    the single strongest feature at the root are forced apart.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 50,
        max_depth: int = 12,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        seed: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees_: list[DecisionTree] = []

    def fit(self, X: Array, y: Array) -> Self:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        n_samples = X.shape[0]
        self.trees_ = []
        for index in range(self.n_estimators):
            rows = rng.integers(0, n_samples, size=n_samples)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                seed=self.seed + index,
            )
            self.trees_.append(tree.fit(X[rows], y[rows]))
        return self

    def predict(self, X: Array) -> Array:
        return np.asarray(
            np.mean([tree.predict(X) for tree in self.trees_], axis=0), dtype=np.float64
        )


class GradientBoostedTrees:
    """Additive trees fitted to the residual, for squared error.

    With squared loss the negative gradient of the loss at each point is just the residual, so
    "gradient boosting" here reads as: fit a shallow tree to what is left over, add a small
    multiple of it, repeat. The small multiple is the learning rate, and it is the difference
    between a method that converges usefully and one that has memorised the training set by
    round twenty.

    Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine*, Annals of
    Statistics 29(5).
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        seed: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.initial_ = 0.0
        self.trees_: list[DecisionTree] = []

    def fit(self, X: Array, y: Array) -> Self:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # The constant that minimises squared error is the mean; every tree after this one is
        # a correction to it, which is why the first prediction is already sensible.
        self.initial_ = float(y.mean())
        prediction = np.full(y.shape[0], self.initial_)
        self.trees_ = []
        for index in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                seed=self.seed + index,
            ).fit(X, y - prediction)
            prediction = prediction + self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def staged_predict(self, X: Array) -> Array:
        """Predictions after every round, as an ``(n_estimators + 1, n_samples)`` array.

        Returned in one pass because the interesting question about boosting is not what the
        finished ensemble does — it is where along the way it was at its best.
        """
        X = np.asarray(X, dtype=np.float64)
        stages: Array = np.empty((len(self.trees_) + 1, X.shape[0]), dtype=np.float64)
        stages[0] = self.initial_
        for index, tree in enumerate(self.trees_):
            stages[index + 1] = stages[index] + self.learning_rate * tree.predict(X)
        return stages

    def predict(self, X: Array) -> Array:
        final: Array = self.staged_predict(X)[-1]
        return final
