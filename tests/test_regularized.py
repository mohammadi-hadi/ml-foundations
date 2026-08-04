"""Verification for lesson 3.

Tier A is the interesting one here, because it is where a mismatched scaling convention would
hide. Ridge and Lasso are normalised differently in scikit-learn, so both are checked at
several penalty strengths rather than at one — a wrong convention agrees at exactly α = 0 and
nowhere else, and a single-point test would pass.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import Ridge as SklearnRidge

from ml_foundations import datasets as ds
from ml_foundations.linear import LinearRegression
from ml_foundations.regularized import Lasso, Ridge, soft_threshold

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_ridge_matches_sklearn(alpha: float) -> None:
    data = ds.make_linear(n_samples=200, n_features=8, seed=60)
    mine = Ridge(alpha=alpha).fit(data.X, data.y)
    reference = SklearnRidge(alpha=alpha).fit(data.X, data.y)
    np.testing.assert_allclose(mine.coef_, reference.coef_, rtol=1e-8, atol=1e-8)
    assert abs(mine.intercept_ - float(reference.intercept_)) < 1e-8


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2, 1.0])
def test_lasso_matches_sklearn(alpha: float) -> None:
    data = ds.make_sparse_linear(n_samples=150, n_features=30, seed=61)
    mine = Lasso(alpha=alpha, max_iter=20000, tol=1e-12).fit(data.X, data.y)
    reference = SklearnLasso(alpha=alpha, max_iter=20000, tol=1e-12).fit(data.X, data.y)
    np.testing.assert_allclose(mine.coef_, reference.coef_, rtol=1e-5, atol=1e-6)
    assert abs(mine.intercept_ - float(reference.intercept_)) < 1e-6


@pytest.mark.parametrize("alpha", [0.05, 0.5])
def test_lasso_selects_the_same_features_as_sklearn(alpha: float) -> None:
    """Which coefficients are exactly zero, not merely how close the values are."""
    data = ds.make_sparse_linear(n_samples=150, n_features=30, seed=62)
    mine = Lasso(alpha=alpha, max_iter=20000, tol=1e-12).fit(data.X, data.y)
    reference = SklearnLasso(alpha=alpha, max_iter=20000, tol=1e-12).fit(data.X, data.y)
    np.testing.assert_array_equal(mine.coef_ == 0.0, reference.coef_ == 0.0)


def test_ridge_at_zero_penalty_is_ordinary_least_squares() -> None:
    data = ds.make_linear(n_samples=200, n_features=5, seed=63)
    np.testing.assert_allclose(
        Ridge(alpha=0.0).fit(data.X, data.y).coef_,
        LinearRegression().fit(data.X, data.y).coef_,
        rtol=1e-9,
        atol=1e-9,
    )


def test_soft_threshold_is_the_exact_minimiser_it_claims_to_be() -> None:
    """Checked by brute force against the one-dimensional objective it solves."""
    grid = np.linspace(-5.0, 5.0, 200001)
    for z in (-3.0, -0.4, 0.0, 0.4, 3.0):
        for gamma in (0.5, 1.0):
            objective = 0.5 * (grid - z) ** 2 + gamma * np.abs(grid)
            assert float(soft_threshold(z, gamma)) == pytest.approx(
                float(grid[np.argmin(objective)]), abs=1e-4
            )


def test_ridge_shrinks_towards_zero_monotonically() -> None:
    data = ds.make_linear(n_samples=200, n_features=6, seed=64)
    sizes = [
        float(np.linalg.norm(Ridge(alpha=a).fit(data.X, data.y).coef_))
        for a in (0.0, 1.0, 10.0, 100.0, 1000.0)
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_ridge_never_produces_an_exact_zero_and_lasso_does() -> None:
    """The structural difference between the two penalties, stated as a test."""
    data = ds.make_sparse_linear(n_samples=120, n_features=40, n_informative=5, seed=65)
    ridge = Ridge(alpha=10.0).fit(data.X, data.y)
    lasso = Lasso(alpha=0.5).fit(data.X, data.y)
    assert int(np.count_nonzero(ridge.coef_ == 0.0)) == 0
    assert int(np.count_nonzero(lasso.coef_ == 0.0)) > 20


def test_a_large_enough_penalty_zeroes_every_coefficient() -> None:
    """There is a finite threshold, and above it the lasso returns the mean of y."""
    data = ds.make_sparse_linear(n_samples=120, n_features=40, seed=66)
    fitted = Lasso(alpha=1e4).fit(data.X, data.y)
    assert np.all(fitted.coef_ == 0.0)
    assert fitted.intercept_ == pytest.approx(float(data.y.mean()))


def test_ridge_is_not_invariant_to_the_units_a_feature_is_measured_in() -> None:
    """Lesson 1 proved least squares *is* invariant. Adding a penalty destroys that.

    The penalty is a statement about the size of the coefficients, and the size of a
    coefficient depends on the units of its feature. Measure a length in millimetres instead
    of metres and its coefficient shrinks by a thousand, which makes the penalty on it a
    thousand times weaker. This is why standardising is not optional here.
    """
    data = ds.make_linear(n_samples=200, n_features=4, seed=67)
    base = Ridge(alpha=10.0).fit(data.X, data.y)
    X = data.X.copy()
    X[:, 1] *= 1000.0
    rescaled = Ridge(alpha=10.0).fit(X, data.y)
    expected = base.coef_.copy()
    expected[1] /= 1000.0
    assert not np.allclose(rescaled.coef_, expected, rtol=1e-3)


def test_ridge_rescues_the_coefficients_that_lesson_one_lost() -> None:
    """The claim lesson 3 opens on, and the reason the penalty is worth its bias."""
    data = ds.make_collinear(n_samples=200, n_features=6, independent_scale=1e-3, seed=68)
    assert data.coef is not None
    unpenalised = LinearRegression().fit(data.X, data.y)
    penalised = Ridge(alpha=1.0).fit(data.X, data.y)
    assert np.linalg.norm(penalised.coef_ - data.coef) < np.linalg.norm(
        unpenalised.coef_ - data.coef
    )


def test_coordinate_descent_stops_early_when_it_has_converged() -> None:
    data = ds.make_sparse_linear(n_samples=120, n_features=20, seed=69)
    fitted = Lasso(alpha=0.5, max_iter=5000).fit(data.X, data.y)
    assert fitted.n_iter_ < 5000


def test_a_constant_column_does_not_divide_by_zero() -> None:
    data = ds.make_linear(n_samples=100, n_features=3, seed=70)
    X = np.column_stack([data.X, np.full(100, 7.0)])
    fitted = Lasso(alpha=0.1).fit(X, data.y)
    assert np.all(np.isfinite(fitted.coef_))
    assert fitted.coef_[-1] == 0.0


def test_negative_penalties_are_refused() -> None:
    for model in (Ridge, Lasso):
        with pytest.raises(ValueError):
            model(alpha=-1.0)


def test_the_bias_variance_decomposition_actually_adds_up() -> None:
    """Lesson 3's central table is an identity, and this checks it is being computed as one.

    Expected squared error equals bias² plus variance plus noise. All three are measured
    here from the same 60 replicate fits — nothing is inferred from the others — so the
    agreement is evidence and not arithmetic.
    """
    rng = np.random.default_rng(71)
    truth = rng.uniform(-1.0, 1.0, size=10)
    noise = 2.0
    test = ds.make_linear(n_samples=1500, n_features=10, noise=noise, seed=72, coef=truth)
    assert test.intercept is not None
    noiseless = test.X @ truth + test.intercept
    replicates = 60
    targets = noiseless + rng.standard_normal((replicates, test.n_samples)) * noise

    predictions = np.array(
        [
            Ridge(alpha=3.0).fit(*_sample(truth, noise, seed=73 + r)).predict(test.X)
            for r in range(replicates)
        ]
    )
    bias_squared = float(np.mean((predictions.mean(axis=0) - noiseless) ** 2))
    variance = float(np.mean(predictions.var(axis=0)))
    measured = float(np.mean((predictions - targets) ** 2))
    assert bias_squared + variance + noise**2 == pytest.approx(measured, rel=0.03)


def _sample(truth: np.ndarray, noise: float, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = ds.make_linear(n_samples=25, n_features=10, noise=noise, seed=seed, coef=truth)
    return data.X, data.y


def test_a_penalty_makes_feature_selection_depend_on_the_unit_of_measurement() -> None:
    """The lesson-3 claim that costs the most in practice, pinned from both directions."""
    data = ds.make_sparse_linear(n_samples=120, n_features=40, n_informative=5, seed=74)
    X_train, X_test, y_train, _ = ds.train_test_split(data.X, data.y, seed=74)

    def kept_first_feature(factor: float, *, standardise: bool) -> bool:
        train, test = X_train.copy(), X_test.copy()
        train[:, 0] *= factor
        test[:, 0] *= factor
        if standardise:
            train, test = ds.standardize(train, test)
        return bool(Lasso(alpha=0.5, max_iter=20000).fit(train, y_train).coef_[0] != 0.0)

    assert kept_first_feature(1.0, standardise=False)
    assert not kept_first_feature(0.01, standardise=False)
    # Standardising restores the invariance that least squares had for free.
    assert kept_first_feature(1.0, standardise=True)
    assert kept_first_feature(0.01, standardise=True)
