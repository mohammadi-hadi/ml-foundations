"""The generators make promises in their docstrings. These tests are those promises.

If a generator quietly stopped being ill-conditioned, or stopped hitting its requested base
rate, several lessons downstream would still run and would silently stop demonstrating what
they claim to demonstrate. That failure is invisible from the lesson's own output, so it has
to be caught here.
"""

from __future__ import annotations

import numpy as np

from ml_foundations import datasets as ds


def test_linear_truth_is_recoverable() -> None:
    data = ds.make_linear(n_samples=4000, n_features=5, noise=0.5, seed=1)
    design = np.column_stack([data.X, np.ones(data.n_samples)])
    fitted, *_ = np.linalg.lstsq(design, data.y, rcond=None)
    assert data.coef is not None
    np.testing.assert_allclose(fitted[:-1], data.coef, atol=0.05)
    assert data.intercept is not None
    assert abs(fitted[-1] - data.intercept) < 0.05


def test_collinear_is_actually_ill_conditioned() -> None:
    well = ds.make_linear(n_samples=200, n_features=6, seed=2)
    ill = ds.make_collinear(n_samples=200, n_features=6, seed=2)
    assert np.linalg.cond(well.X) < 10.0
    # Four orders of magnitude apart is the whole point; the exact figure is reported by
    # lesson 1 rather than asserted here, because it is a measurement, not a contract.
    assert np.linalg.cond(ill.X) > 10_000.0


def test_collinear_predictions_survive_even_though_coefficients_do_not() -> None:
    """Ill-conditioning damages the parameters, not the fit. Lesson 1 turns on this."""
    data = ds.make_collinear(n_samples=400, n_features=6, noise=1.0, seed=3)
    design = np.column_stack([data.X, np.ones(data.n_samples)])
    fitted, *_ = np.linalg.lstsq(design, data.y, rcond=None)
    residual = data.y - design @ fitted
    assert data.coef is not None
    assert np.linalg.norm(fitted[:-1] - data.coef) > 1.0
    assert float(np.sqrt(np.mean(residual**2))) < 1.2


def test_sparse_linear_has_exactly_the_promised_number_of_live_coefficients() -> None:
    data = ds.make_sparse_linear(n_features=40, n_informative=5, seed=4)
    assert data.coef is not None
    assert int(np.count_nonzero(data.coef)) == 5
    assert np.all(data.coef[5:] == 0.0)


def test_logistic_hits_the_requested_base_rate() -> None:
    for rate in (0.5, 0.2, 0.05):
        data = ds.make_logistic(n_samples=8000, positive_rate=rate, seed=5)
        assert abs(float(data.y.mean()) - rate) < 0.03, rate


def test_xor_features_are_individually_uninformative() -> None:
    data = ds.make_xor(n_samples=4000, seed=6)
    for column in range(2):
        assert abs(float(np.corrcoef(data.X[:, column], data.y)[0, 1])) < 0.05
    interaction = data.X[:, 0] * data.X[:, 1]
    assert abs(float(np.corrcoef(interaction, data.y)[0, 1])) > 0.8


def test_friedman_last_five_features_are_decoration() -> None:
    data = ds.make_friedman1(n_samples=4000, noise=0.1, seed=7)
    for column in range(5, 10):
        assert abs(float(np.corrcoef(data.X[:, column], data.y)[0, 1])) < 0.05


def test_split_is_a_partition_and_is_reproducible() -> None:
    data = ds.make_linear(n_samples=100, seed=8)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, test_size=0.25, seed=9)
    assert X_train.shape[0] == 75
    assert X_test.shape[0] == 25
    assert y_train.shape[0] == 75
    assert y_test.shape[0] == 25
    again = ds.train_test_split(data.X, data.y, test_size=0.25, seed=9)
    np.testing.assert_array_equal(X_train, again[0])
    # Rows are drawn without replacement, so train and test share nothing.
    train_rows = {tuple(row) for row in X_train}
    assert not train_rows & {tuple(row) for row in X_test}


def test_stratified_split_keeps_the_base_rate_in_both_halves() -> None:
    data = ds.make_logistic(n_samples=1000, positive_rate=0.1, seed=10)
    _, _, y_train, y_test = ds.train_test_split(
        data.X, data.y, test_size=0.3, seed=11, stratify=True
    )
    assert abs(float(y_train.mean()) - float(y_test.mean())) < 0.01


def test_standardize_uses_training_statistics_only() -> None:
    data = ds.make_linear(n_samples=400, n_features=3, seed=12)
    X_train, X_test, _, _ = ds.train_test_split(data.X, data.y, seed=13)
    train_scaled, test_scaled = ds.standardize(X_train, X_test)
    np.testing.assert_allclose(train_scaled.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(train_scaled.std(axis=0), 1.0, atol=1e-12)
    # The test half is *not* centred on itself. If it were, the split would be leaking.
    assert np.any(np.abs(test_scaled.mean(axis=0)) > 1e-6)


def test_standardize_survives_a_constant_column() -> None:
    X_train = np.column_stack([np.ones(10), np.arange(10.0)])
    X_test = np.column_stack([np.ones(4), np.arange(4.0)])
    train_scaled, _ = ds.standardize(X_train, X_test)
    assert np.all(np.isfinite(train_scaled))
