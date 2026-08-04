"""Verification for lesson 5.

A tree is harder to check against a reference than a linear model: two implementations that
both split optimally can still disagree when two candidate splits score identically, and which
one wins then depends on iteration order. So the checks here are layered — exact agreement on
data with no ties, brute-force verification that the chosen split really is the best available,
and exact recovery of a function the tree is capable of representing perfectly.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_foundations import datasets as ds
from ml_foundations import trees
from ml_foundations.ensembles import BaggedTrees, GradientBoostedTrees
from ml_foundations.metrics import rmse, roc_auc
from ml_foundations.trees import DecisionTree


@pytest.mark.parametrize("max_depth", [1, 2, 3, 5])
def test_regression_tree_predicts_what_sklearn_predicts(max_depth: int) -> None:
    """Continuous features from a normal distribution have no exact ties, so both
    implementations face the same unique best split at every node."""
    data = ds.make_friedman1(n_samples=300, seed=100)
    mine = DecisionTree(max_depth=max_depth).fit(data.X, data.y)
    reference = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(data.X, data.y)
    np.testing.assert_allclose(mine.predict(data.X), reference.predict(data.X), rtol=1e-9)


@pytest.mark.parametrize("max_depth", [1, 2, 3, 4])
def test_classification_tree_predicts_what_sklearn_predicts(max_depth: int) -> None:
    data = ds.make_logistic(n_samples=400, n_features=5, seed=101)
    mine = DecisionTree(criterion="gini", max_depth=max_depth).fit(data.X, data.y)
    reference = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=0).fit(
        data.X, data.y
    )
    np.testing.assert_allclose(
        mine.predict_proba(data.X), reference.predict_proba(data.X)[:, 1], rtol=1e-9
    )


@pytest.mark.parametrize("max_depth", [5, 6, 8])
def test_deep_trees_split_at_least_as_well_as_sklearns(max_depth: int) -> None:
    """Past depth four the two implementations stop agreeing exactly, and that is expected.

    Several candidate splits can achieve *identical* impurity, and which one wins then depends
    on iteration order. Demanding equality there would be testing a coincidence. What can be
    demanded is that the partition this tree produces is no worse by the objective both are
    optimising — at depth five the two are exactly equal, and at depth six this one is better.
    """
    data = ds.make_logistic(n_samples=400, n_features=5, seed=101)
    mine = DecisionTree(criterion="gini", max_depth=max_depth).fit(data.X, data.y)
    reference = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=0).fit(
        data.X, data.y
    )

    def total_gini(leaf_values: np.ndarray) -> float:
        """Impurity of the partition those predictions imply, summed over leaves."""
        total = 0.0
        for value in np.unique(leaf_values):
            members = leaf_values == value
            count, positives = float(members.sum()), float(data.y[members].sum())
            total += 2.0 * positives * (count - positives) / count
        return total

    assert (
        total_gini(mine.predict_proba(data.X))
        <= total_gini(reference.predict_proba(data.X)[:, 1]) + 1e-9
    )


def test_the_chosen_split_is_the_best_one_available() -> None:
    """Brute force over every threshold, with no reference implementation involved.

    The tree's own search is vectorised and uses prefix sums; this recomputes the objective
    from its definition for every candidate and checks the winner agrees.
    """
    rng = np.random.default_rng(102)
    X = rng.standard_normal((80, 3))
    y = rng.standard_normal(80)
    stump = DecisionTree(max_depth=1).fit(X, y)
    assert stump.root_.feature is not None

    def total_error(feature: int, threshold: float) -> float:
        mask = X[:, feature] <= threshold
        if mask.all() or (~mask).all():
            return float("inf")
        return float(
            np.sum((y[mask] - y[mask].mean()) ** 2) + np.sum((y[~mask] - y[~mask].mean()) ** 2)
        )

    best = min(
        (total_error(f, t), f, t)
        for f in range(3)
        for t in (np.sort(X[:, f])[:-1] + np.sort(X[:, f])[1:]) / 2.0
    )
    assert total_error(stump.root_.feature, stump.root_.threshold) == pytest.approx(best[0])


def test_a_tree_recovers_a_function_it_can_represent_exactly() -> None:
    """Piecewise constant on a grid, no noise. The right answer is zero error."""
    rng = np.random.default_rng(103)
    X = rng.uniform(0, 1, size=(600, 2))
    y = np.where(X[:, 0] < 0.5, np.where(X[:, 1] < 0.3, 1.0, 2.0), 3.0)
    fitted = DecisionTree(max_depth=3).fit(X, y)
    np.testing.assert_allclose(fitted.predict(X), y, atol=1e-12)
    assert fitted.n_leaves == 3


def test_depth_and_leaf_limits_are_respected() -> None:
    data = ds.make_friedman1(n_samples=400, seed=104)
    for depth in (1, 2, 4):
        fitted = DecisionTree(max_depth=depth).fit(data.X, data.y)
        assert fitted.depth <= depth
        assert fitted.n_leaves <= 2**depth
    big_leaves = DecisionTree(max_depth=20, min_samples_leaf=50).fit(data.X, data.y)

    def leaf_sizes(node: object) -> list[int]:
        assert isinstance(node, type(big_leaves.root_))
        if node.is_leaf:
            return [node.n_samples]
        assert node.left is not None and node.right is not None
        return leaf_sizes(node.left) + leaf_sizes(node.right)

    assert min(leaf_sizes(big_leaves.root_)) >= 50


def test_a_tree_ignores_any_monotone_rescaling_of_a_feature() -> None:
    """Only the order of the values matters, so no standardising is needed — ever."""
    data = ds.make_friedman1(n_samples=300, seed=105)
    base = DecisionTree(max_depth=4).fit(data.X, data.y)
    X = data.X.copy()
    X[:, 0] = np.exp(5.0 * X[:, 0])
    X[:, 1] = X[:, 1] * 1e6 - 400.0
    rescaled = DecisionTree(max_depth=4).fit(X, data.y)
    np.testing.assert_allclose(base.predict(data.X), rescaled.predict(X), rtol=1e-12)


def test_a_deep_tree_memorises_the_training_set() -> None:
    """The claim lesson 5 opens with, from both directions."""
    data = ds.make_friedman1(n_samples=300, noise=1.0, seed=106)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=106)
    shallow = DecisionTree(max_depth=3).fit(X_train, y_train)
    deep = DecisionTree(max_depth=30).fit(X_train, y_train)
    assert rmse(y_train, deep.predict(X_train)) < 1e-9
    assert rmse(y_train, shallow.predict(X_train)) > 1.0
    assert rmse(y_test, deep.predict(X_test)) > rmse(y_test, shallow.predict(X_test))


def test_bagging_beats_the_tree_it_is_made_of() -> None:
    data = ds.make_friedman1(n_samples=400, seed=107)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=107)
    single = DecisionTree(max_depth=12).fit(X_train, y_train)
    bagged = BaggedTrees(n_estimators=30, max_depth=12, seed=107).fit(X_train, y_train)
    assert rmse(y_test, bagged.predict(X_test)) < rmse(y_test, single.predict(X_test))


def test_a_random_forest_is_within_reach_of_sklearns() -> None:
    """Not exact — the bootstrap draws and feature subsets differ — but the same model.

    An implementation with a broken split rule or a mis-scaled average would not land within
    a few per cent of a mature one, so this is a real check even though it is not an equality.
    """
    data = ds.make_friedman1(n_samples=500, seed=108)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=108)
    mine = BaggedTrees(n_estimators=40, max_depth=12, max_features=3, seed=108).fit(
        X_train, y_train
    )
    reference = RandomForestRegressor(
        n_estimators=40, max_depth=12, max_features=3, random_state=108
    ).fit(X_train, y_train)
    assert rmse(y_test, mine.predict(X_test)) < 1.15 * rmse(y_test, reference.predict(X_test))


def test_boosting_is_within_reach_of_sklearns() -> None:
    data = ds.make_friedman1(n_samples=400, seed=109)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=109)
    mine = GradientBoostedTrees(n_estimators=80, learning_rate=0.1, max_depth=3).fit(
        X_train, y_train
    )
    reference = GradientBoostingRegressor(
        n_estimators=80, learning_rate=0.1, max_depth=3, random_state=109
    ).fit(X_train, y_train)
    assert rmse(y_test, mine.predict(X_test)) < 1.15 * rmse(y_test, reference.predict(X_test))


def test_staged_predictions_end_where_predict_does() -> None:
    data = ds.make_friedman1(n_samples=200, seed=110)
    fitted = GradientBoostedTrees(n_estimators=20).fit(data.X, data.y)
    stages = fitted.staged_predict(data.X)
    assert stages.shape == (21, 200)
    np.testing.assert_allclose(stages[0], data.y.mean())
    np.testing.assert_allclose(stages[-1], fitted.predict(data.X), rtol=1e-12)


def test_boosting_training_error_decreases_every_round() -> None:
    """It is fitting the residual, so it must. A round that made training error worse would
    mean the trees are not being fitted to what the ensemble got wrong."""
    data = ds.make_friedman1(n_samples=300, seed=111)
    fitted = GradientBoostedTrees(n_estimators=40, learning_rate=0.1).fit(data.X, data.y)
    errors = [rmse(data.y, stage) for stage in fitted.staged_predict(data.X)]
    assert all(later <= earlier + 1e-12 for earlier, later in pairwise(errors))


def test_a_classification_tree_ranks_better_than_chance() -> None:
    data = ds.make_logistic(n_samples=1500, n_features=5, separation=2.0, seed=112)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=112)
    fitted = DecisionTree(criterion="gini", max_depth=4, min_samples_leaf=10).fit(X_train, y_train)
    assert roc_auc(y_test, fitted.predict_proba(X_test)) > 0.7


def test_the_split_choice_survives_jitter_in_the_impurity_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Determinism against last-bit arithmetic — which is what makes CI's drift check possible.

    ``numpy.sum`` adds in blocks whose size depends on the machine's vector width, so the same
    sum can differ in its final bits between platforms. That is far too small to change which
    split is genuinely best, and quite large enough to change which of two *equally* good
    splits a comparison happens to prefer — and a greedy tree turns one flipped split at the
    root into a different model.

    The jitter injected here is exactly that perturbation: a relative 1e-13 on every impurity
    score, applied where the platform's own differences would appear. Without the quantisation
    in ``_best_split`` this test fails, and the numbers in lesson 5 depend on the hardware.
    """
    data = ds.make_friedman1(n_samples=400, noise=1.0, seed=113)
    original = trees._impurity_total
    rng = np.random.default_rng(114)

    def jittered(criterion: str, count: np.ndarray, total: np.ndarray, total_sq: np.ndarray):  # type: ignore[no-untyped-def]
        out = original(criterion, count, total, total_sq)  # type: ignore[arg-type]
        return out * (1.0 + rng.standard_normal(out.shape) * 1e-13)

    for build in (
        lambda: DecisionTree(max_depth=10),
        lambda: BaggedTrees(n_estimators=8, max_depth=10, max_features=3, seed=1),
        lambda: GradientBoostedTrees(n_estimators=25, max_depth=3),
    ):
        clean = build().fit(data.X, data.y).predict(data.X)
        monkeypatch.setattr(trees, "_impurity_total", jittered)
        noisy = build().fit(data.X, data.y).predict(data.X)
        monkeypatch.setattr(trees, "_impurity_total", original)
        np.testing.assert_allclose(clean, noisy, rtol=1e-9, atol=1e-9)
