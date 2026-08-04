"""Verification for lesson 1, at all three tiers.

Tier A — agreement with scikit-learn, on data where both minimise the same objective.
Tier B — algebraic properties that hold exactly, independent of any reference.
Tier C — recovery of coefficients that were known before the data existed.

The last test in the file is different in kind: it pins the *claim the lesson makes*, so that
if the demonstration ever stops demonstrating anything the suite says so.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from ml_foundations import datasets as ds
from ml_foundations.linear import LinearRegression, back_substitute, condition_number
from ml_foundations.metrics import rmse

METHODS = ("normal", "qr", "svd")


@pytest.mark.parametrize("method", METHODS)
def test_matches_sklearn_on_well_conditioned_data(method: str) -> None:
    data = ds.make_linear(n_samples=300, n_features=8, seed=20)
    mine = LinearRegression(method=method).fit(data.X, data.y)  # type: ignore[arg-type]
    reference = SklearnLinearRegression().fit(data.X, data.y)
    np.testing.assert_allclose(mine.coef_, reference.coef_, rtol=1e-9, atol=1e-9)
    assert abs(mine.intercept_ - float(reference.intercept_)) < 1e-9


@pytest.mark.parametrize("method", METHODS)
def test_matches_sklearn_without_an_intercept(method: str) -> None:
    data = ds.make_linear(n_samples=200, n_features=5, intercept=0.0, seed=21)
    mine = LinearRegression(method=method, fit_intercept=False).fit(data.X, data.y)  # type: ignore[arg-type]
    reference = SklearnLinearRegression(fit_intercept=False).fit(data.X, data.y)
    np.testing.assert_allclose(mine.coef_, reference.coef_, rtol=1e-9, atol=1e-9)
    assert mine.intercept_ == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_recovers_the_coefficients_it_was_generated_from(method: str) -> None:
    data = ds.make_linear(n_samples=20000, n_features=6, noise=0.5, seed=22)
    assert data.coef is not None
    fitted = LinearRegression(method=method).fit(data.X, data.y)  # type: ignore[arg-type]
    np.testing.assert_allclose(fitted.coef_, data.coef, atol=0.02)
    assert data.intercept is not None
    assert abs(fitted.intercept_ - data.intercept) < 0.02


def test_the_three_solvers_agree_when_the_problem_is_well_posed() -> None:
    data = ds.make_linear(n_samples=400, n_features=10, seed=23)
    coefficients = [LinearRegression(method=m).fit(data.X, data.y).coef_ for m in METHODS]  # type: ignore[arg-type]
    for other in coefficients[1:]:
        np.testing.assert_allclose(coefficients[0], other, rtol=1e-10, atol=1e-10)


def test_shifting_the_target_moves_only_the_intercept() -> None:
    data = ds.make_linear(n_samples=200, seed=24)
    base = LinearRegression().fit(data.X, data.y)
    shifted = LinearRegression().fit(data.X, data.y + 17.0)
    np.testing.assert_allclose(base.coef_, shifted.coef_, rtol=1e-10, atol=1e-10)
    assert abs((shifted.intercept_ - base.intercept_) - 17.0) < 1e-9


def test_rescaling_a_feature_rescales_exactly_one_coefficient() -> None:
    """Least squares is equivariant under a change of units. Lesson 3 breaks this on purpose."""
    data = ds.make_linear(n_samples=200, n_features=4, seed=25)
    base = LinearRegression().fit(data.X, data.y)
    X = data.X.copy()
    X[:, 2] *= 100.0
    rescaled = LinearRegression().fit(X, data.y)
    expected = base.coef_.copy()
    expected[2] /= 100.0
    np.testing.assert_allclose(rescaled.coef_, expected, rtol=1e-8, atol=1e-8)


def test_predictions_are_orthogonal_to_the_residual() -> None:
    """The defining property of a least-squares fit, and a check no reference is needed for."""
    data = ds.make_linear(n_samples=300, n_features=5, seed=26)
    fitted = LinearRegression().fit(data.X, data.y)
    residual = data.y - fitted.predict(data.X)
    np.testing.assert_allclose(data.X.T @ residual, 0.0, atol=1e-8)
    assert abs(float(residual.sum())) < 1e-8


def test_back_substitution_matches_a_general_solver() -> None:
    rng = np.random.default_rng(27)
    R = np.triu(rng.standard_normal((6, 6))) + np.eye(6) * 5.0
    b = rng.standard_normal(6)
    np.testing.assert_allclose(back_substitute(R, b), np.linalg.solve(R, b), rtol=1e-10)


def test_back_substitution_refuses_a_singular_factor() -> None:
    R = np.array([[1.0, 2.0], [0.0, 0.0]])
    with pytest.raises(np.linalg.LinAlgError):
        back_substitute(R, np.ones(2))


def test_svd_solver_survives_exactly_duplicated_columns() -> None:
    """A duplicated feature makes the coefficients non-unique, not the predictions.

    ``normal`` and ``qr`` have nothing to return here; the truncating pseudo-inverse splits
    the weight between the two identical columns and predicts correctly.
    """
    rng = np.random.default_rng(28)
    base = rng.standard_normal((200, 3))
    X = np.column_stack([base, base[:, 0]])
    y = base @ np.array([2.0, -1.0, 0.5]) + rng.standard_normal(200) * 0.1
    fitted = LinearRegression(method="svd").fit(X, y)
    assert fitted.rank_ == 3
    assert rmse(y, fitted.predict(X)) < 0.15


def test_predict_before_fit_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        LinearRegression().predict(np.zeros((2, 2)))


def test_the_lesson_one_claim_holds() -> None:
    """Normal equations lose accuracy that QR and SVD keep, and none of it shows in the fit.

    This is the finding lesson 1 is built on, asserted here so that a change in numpy or in
    the generator cannot quietly turn the lesson into an illustration of nothing. The
    threshold is deliberately loose — the measured figures live in the lesson, not here.
    """
    data = ds.make_collinear(n_samples=400, n_features=6, independent_scale=1e-5, seed=29)
    assert condition_number(data.X) > 1e5

    solutions = {m: LinearRegression(method=m).fit(data.X, data.y) for m in METHODS}
    disagreement = {
        m: float(np.linalg.norm(solutions[m].coef_ - solutions["svd"].coef_)) for m in METHODS
    }
    assert disagreement["normal"] > 100.0 * max(disagreement["qr"], 1e-12)

    # And yet every one of them fits the data. The damage is to the parameters alone.
    for fitted in solutions.values():
        assert rmse(data.y, fitted.predict(data.X)) < 1.2
