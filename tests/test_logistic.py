"""Verification for lesson 4.

The reference comparison needs care. scikit-learn's ``LogisticRegression`` regularises by
default, so a naive comparison against a maximum likelihood implementation disagrees at every
coefficient and looks like a bug in the implementation rather than in the test. Both the
unpenalised case (``penalty=None``) and the penalised case (``C = 1/alpha``) are checked here,
because a wrong scaling convention would pass the first and fail only the second.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from ml_foundations import datasets as ds
from ml_foundations.functions import sigmoid
from ml_foundations.logistic import LogisticRegression
from ml_foundations.metrics import accuracy, log_loss, recall, roc_auc


# scikit-learn 1.8 deprecated `penalty=None` in favour of `C=np.inf`, which older supported
# versions reject. Keeping the spelling that works across the whole supported range and
# silencing the notice is preferable to narrowing what this package installs against.
@pytest.mark.filterwarnings("ignore:'penalty' was deprecated:FutureWarning")
def test_matches_sklearn_maximum_likelihood() -> None:
    data = ds.make_logistic(n_samples=500, n_features=5, seed=80)
    mine = LogisticRegression().fit(data.X, data.y)
    reference = SklearnLogisticRegression(penalty=None, tol=1e-12, max_iter=5000).fit(
        data.X, data.y
    )
    np.testing.assert_allclose(mine.coef_, reference.coef_[0], rtol=1e-5, atol=1e-6)
    assert abs(mine.intercept_ - float(reference.intercept_[0])) < 1e-6


@pytest.mark.parametrize("alpha", [0.5, 2.0, 10.0])
def test_matches_sklearn_with_a_penalty(alpha: float) -> None:
    """Where a mismatched convention would show up. ``C`` is the reciprocal of ``alpha``."""
    data = ds.make_logistic(n_samples=500, n_features=5, seed=81)
    mine = LogisticRegression(alpha=alpha).fit(data.X, data.y)
    reference = SklearnLogisticRegression(C=1.0 / alpha, tol=1e-12, max_iter=5000).fit(
        data.X, data.y
    )
    np.testing.assert_allclose(mine.coef_, reference.coef_[0], rtol=1e-4, atol=1e-5)
    assert abs(mine.intercept_ - float(reference.intercept_[0])) < 1e-5


def test_recovers_the_coefficients_it_was_generated_from() -> None:
    data = ds.make_logistic(n_samples=60000, n_features=4, separation=1.5, seed=82)
    assert data.coef is not None
    fitted = LogisticRegression().fit(data.X, data.y)
    np.testing.assert_allclose(fitted.coef_, data.coef, atol=0.05)
    assert data.intercept is not None
    assert abs(fitted.intercept_ - data.intercept) < 0.05


def test_the_gradient_of_the_log_likelihood_vanishes_at_the_fit() -> None:
    """A first-order condition, checkable without any reference implementation.

    At the maximum likelihood estimate the residual ``y - p`` must be orthogonal to every
    feature, exactly as the least-squares residual was in lesson 1.
    """
    data = ds.make_logistic(n_samples=800, n_features=5, seed=83)
    fitted = LogisticRegression().fit(data.X, data.y)
    residual = data.y - fitted.predict_proba(data.X)
    np.testing.assert_allclose(data.X.T @ residual, 0.0, atol=1e-6)
    assert abs(float(residual.sum())) < 1e-6


def test_newton_converges_in_a_handful_of_iterations() -> None:
    """Quadratic convergence is the reason to use this rather than gradient descent here."""
    data = ds.make_logistic(n_samples=2000, n_features=6, seed=84)
    assert LogisticRegression(max_iter=100).fit(data.X, data.y).n_iter_ < 15


def test_the_fit_is_a_true_optimum_not_merely_a_stopping_point() -> None:
    """Perturb the solution in twenty random directions; the objective must get worse."""
    data = ds.make_logistic(n_samples=400, n_features=4, seed=85)
    fitted = LogisticRegression(alpha=1.0).fit(data.X, data.y)
    design = np.column_stack([data.X, np.ones(data.n_samples)])
    w = np.append(fitted.coef_, fitted.intercept_)

    def objective(v: np.ndarray) -> float:
        return float(
            np.sum(
                -data.y * np.log(sigmoid(design @ v)) - (1 - data.y) * np.log(sigmoid(-design @ v))
            )
            + 0.5 * 1.0 * float(v[:-1] @ v[:-1])
        )

    best = objective(w)
    rng = np.random.default_rng(86)
    for _ in range(20):
        direction = rng.standard_normal(w.shape[0])
        assert objective(w + 1e-3 * direction) > best


def test_probabilities_and_log_odds_rank_identically() -> None:
    data = ds.make_logistic(n_samples=300, seed=87)
    fitted = LogisticRegression().fit(data.X, data.y)
    scores = fitted.decision_function(data.X)
    assert roc_auc(data.y, scores) == pytest.approx(roc_auc(data.y, fitted.predict_proba(data.X)))


def test_lesson_four_separable_training_set_really_is_separable() -> None:
    """The premise of the lesson's last table, checked with an independent implementation.

    A hard-margin linear support vector classifier reaches 100% training accuracy if and only
    if the classes can be split by a hyperplane. Forty rows and twenty features is enough for
    that to happen by accident, which is the point being made.
    """
    from sklearn.svm import LinearSVC

    data = ds.make_logistic(n_samples=4040, n_features=20, separation=1.0, seed=4)
    X_train, y_train = data.X[:40], data.y[:40]
    separator = LinearSVC(C=1e6, max_iter=500_000).fit(X_train, y_train)
    assert separator.score(X_train, y_train) == 1.0


def test_separable_data_sends_the_coefficients_to_infinity() -> None:
    """The lesson-4 claim about maximum likelihood, and the reason a penalty is not optional.

    When a hyperplane separates the classes perfectly, every scaling of that hyperplane
    classifies correctly and a larger one has a strictly higher likelihood. There is no
    maximum. The fit does not fail, error or warn — it returns whatever it had reached when
    the iteration budget ran out, and how confident the model claims to be is decided by that
    budget rather than by the data.
    """
    data = ds.make_logistic(n_samples=4040, n_features=20, separation=1.0, seed=4)
    X_train, y_train = data.X[:40], data.y[:40]
    X_test, y_test = data.X[40:], data.y[40:]

    short = LogisticRegression(max_iter=5, tol=0.0).fit(X_train, y_train)
    longer = LogisticRegression(max_iter=50, tol=0.0).fit(X_train, y_train)
    assert float(np.linalg.norm(longer.coef_)) > 5.0 * float(np.linalg.norm(short.coef_))
    assert np.all(np.isfinite(longer.coef_))

    # The training likelihood improves all the way, and the held-out one falls apart.
    assert log_loss(y_train, longer.decision_function(X_train)) < 1e-6
    assert log_loss(y_test, longer.decision_function(X_test)) > 10.0

    # But only the probabilities are ruined. The ranking barely moves, which is exactly why
    # a metric that only looks at ranking cannot detect this.
    assert (
        abs(
            roc_auc(y_test, longer.decision_function(X_test))
            - roc_auc(y_test, short.decision_function(X_test))
        )
        < 0.05
    )

    # A penalty of any size gives the optimisation something finite to find.
    penalised = LogisticRegression(alpha=1.0, max_iter=200).fit(X_train, y_train)
    assert penalised.n_iter_ < 200
    assert log_loss(y_test, penalised.decision_function(X_test)) < 1.0


def test_accuracy_can_be_excellent_while_the_model_is_useless() -> None:
    """Lesson 4's headline, asserted so the demonstration cannot quietly stop demonstrating."""
    data = ds.make_logistic(n_samples=4000, positive_rate=0.01, separation=1.0, seed=89)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=89, stratify=True)
    fitted = LogisticRegression(alpha=1.0).fit(X_train, y_train)
    predicted = fitted.predict(X_test)
    assert accuracy(y_test, predicted) > 0.98
    assert recall(y_test, predicted) < 0.2
    # And the model is not actually useless — the ranking is informative, the threshold is not.
    assert roc_auc(y_test, fitted.decision_function(X_test)) > 0.6


def test_log_loss_of_the_fit_beats_predicting_the_base_rate() -> None:
    data = ds.make_logistic(n_samples=1000, n_features=4, seed=90)
    fitted = LogisticRegression().fit(data.X, data.y)
    base_rate = float(data.y.mean())
    constant = np.full(data.n_samples, np.log(base_rate / (1 - base_rate)))
    assert log_loss(data.y, fitted.decision_function(data.X)) < log_loss(data.y, constant)


def test_negative_penalties_are_refused() -> None:
    with pytest.raises(ValueError):
        LogisticRegression(alpha=-1.0)
