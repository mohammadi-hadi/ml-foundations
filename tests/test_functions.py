"""Verification tier A: agreement with an independent implementation of the same function.

``scipy.special`` is not a dependency of this package. It is imported here, in the tests,
for the only purpose it serves in this repo: being something to disagree with.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit

from ml_foundations.functions import log_sigmoid, sigmoid


def test_sigmoid_matches_scipy_on_ordinary_inputs() -> None:
    z = np.linspace(-30.0, 30.0, 601)
    np.testing.assert_allclose(sigmoid(z), expit(z), rtol=0.0, atol=1e-15)


def test_sigmoid_survives_the_range_where_the_textbook_formula_does_not() -> None:
    z = np.array([-1e4, -800.0, -710.0, 0.0, 710.0, 800.0, 1e4])
    result = sigmoid(z)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expit(z), rtol=0.0, atol=1e-15)
    # The naive form does not crash and does not return nan. It raises an overflow warning
    # and hands back a probability of exactly zero, which is the failure mode the docstring
    # describes: quiet, and fatal one line later when something takes its logarithm.
    with np.errstate(over="raise"):
        try:
            naive = 1.0 / (1.0 + np.exp(-z))
        except FloatingPointError:
            naive = None
    assert naive is None
    assert float(result[2]) > 0.0


def test_the_complement_cannot_be_computed_by_subtraction() -> None:
    """``1 - sigmoid(z)`` is not a way to get ``sigmoid(-z)``, and this is where it breaks.

    Near the middle the two agree to the last bit or two. Out in the tail the subtraction
    cancels away every significant digit: ``sigmoid(40)`` rounds to 1.0, so its complement
    rounds to 0, while the true value is 4e-18. Any code that needs the probability of the
    other class must evaluate it, not subtract it.
    """
    middle = np.linspace(0.0, 5.0, 501)
    np.testing.assert_allclose(sigmoid(-middle), 1.0 - sigmoid(middle), rtol=0.0, atol=1e-15)

    tail = np.array([40.0])
    assert float(1.0 - sigmoid(tail)[0]) == 0.0
    assert float(sigmoid(-tail)[0]) > 0.0
    np.testing.assert_allclose(sigmoid(-tail), expit(-tail), rtol=1e-15, atol=0.0)


def test_log_sigmoid_matches_scipy_including_where_the_probability_underflows() -> None:
    z = np.array([-2000.0, -800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0])
    np.testing.assert_allclose(log_sigmoid(z), log_expit(z), rtol=1e-14, atol=0.0)
    # Going through the probability first would give -inf for the first three entries.
    assert np.all(np.isfinite(log_sigmoid(z)))
    with np.errstate(divide="ignore"):
        assert not np.all(np.isfinite(np.log(sigmoid(z))))
