"""Verification for lesson 2. The reference is lesson 1: these problems have exact solutions.

That is a stronger check than any convergence criterion. "The updates got small" is
compatible with having stopped in the wrong place; "the parameters equal the closed-form
minimiser to eight digits" is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_foundations import datasets as ds
from ml_foundations.linear import LinearRegression
from ml_foundations.optim import SGD, Adam, least_squares_parts, minimise, minimise_stochastic


def _problem(seed: int = 50, n_features: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A centred least-squares problem and its exact minimiser."""
    data = ds.make_linear(n_samples=400, n_features=n_features, intercept=0.0, seed=seed)
    X = data.X - data.X.mean(axis=0)
    y = data.y - data.y.mean()
    exact = LinearRegression(fit_intercept=False, method="svd").fit(X, y).coef_
    return X, y, exact


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_gradient_descent_finds_the_closed_form_solution(momentum: float) -> None:
    X, y, exact = _problem()
    gradient, _, curvature = least_squares_parts(X, y)
    trace = minimise(
        gradient,
        np.zeros(X.shape[1]),
        SGD(lr=1.0 / curvature, momentum=momentum),
        n_steps=5000,
        optimum=exact,
        tolerance=1e-10,
    )
    assert trace.steps_to_tolerance is not None
    np.testing.assert_allclose(trace.weights, exact, rtol=1e-8, atol=1e-8)


def test_adam_finds_the_same_solution() -> None:
    X, y, exact = _problem()
    gradient, _, _ = least_squares_parts(X, y)
    trace = minimise(
        gradient, np.zeros(X.shape[1]), Adam(lr=0.05), n_steps=20000, optimum=exact, tolerance=1e-8
    )
    assert trace.steps_to_tolerance is not None


def test_the_stability_threshold_is_exactly_two_over_the_curvature() -> None:
    """The claim lesson 2 is built on, checked from both sides of the boundary.

    Below ``2 / L`` gradient descent on a quadratic contracts every error component and must
    converge. Above it, the component along the steepest direction is multiplied by something
    larger than one every step, and no amount of patience helps.
    """
    X, y, exact = _problem()
    gradient, _, curvature = least_squares_parts(X, y)
    threshold = 2.0 / curvature

    with np.errstate(over="ignore", invalid="ignore"):
        just_below = minimise(
            gradient, np.zeros(X.shape[1]), SGD(lr=threshold * 0.98), n_steps=20000, optimum=exact
        )
        just_above = minimise(
            gradient, np.zeros(X.shape[1]), SGD(lr=threshold * 1.02), n_steps=20000, optimum=exact
        )
    assert not just_below.diverged
    assert just_above.diverged


def test_momentum_beats_plain_descent_when_the_problem_is_ill_conditioned() -> None:
    """The second lesson-2 claim: the payoff scales with the conditioning, so it needs a
    badly conditioned problem to show up at all."""
    data = ds.make_collinear(n_samples=400, n_features=5, independent_scale=0.05, seed=51)
    X = data.X - data.X.mean(axis=0)
    y = data.y - data.y.mean()
    exact = LinearRegression(fit_intercept=False).fit(X, y).coef_
    gradient, _, curvature = least_squares_parts(X, y)

    plain = minimise(
        gradient,
        np.zeros(5),
        SGD(lr=1.0 / curvature),
        n_steps=100000,
        optimum=exact,
        tolerance=1e-6,
    )
    fast = minimise(
        gradient,
        np.zeros(5),
        SGD(lr=1.0 / curvature, momentum=0.95),
        n_steps=100000,
        optimum=exact,
        tolerance=1e-6,
    )
    assert plain.steps_to_tolerance is not None
    assert fast.steps_to_tolerance is not None
    assert fast.steps_to_tolerance < plain.steps_to_tolerance / 5


def test_divergence_is_caught_rather_than_run_to_infinity() -> None:
    X, y, exact = _problem()
    gradient, _, curvature = least_squares_parts(X, y)
    with np.errstate(over="ignore", invalid="ignore"):
        trace = minimise(
            gradient, np.zeros(X.shape[1]), SGD(lr=100.0 / curvature), n_steps=10000, optimum=exact
        )
    assert trace.diverged
    assert trace.n_steps < 10000


def test_a_fixed_step_size_leaves_minibatch_descent_orbiting_the_answer() -> None:
    """The third lesson-2 claim, and the one that surprises people.

    Full-batch descent on this problem reaches the exact minimiser. Mini-batch descent with
    the same fixed step size gets close quickly and then stops improving, no matter how many
    further passes it makes, because each batch gradient carries noise that the step size
    keeps reinjecting. Smaller batches mean more noise and a wider orbit.
    """
    X, y, exact = _problem(seed=52)
    gradient, batch_gradient, curvature = least_squares_parts(X, y)
    n_samples = X.shape[0]

    def floor_for(batch_size: int, decay: float = 0.0) -> float:
        trace = minimise_stochastic(
            batch_gradient,
            np.zeros(X.shape[1]),
            SGD(lr=0.2 / curvature, decay=decay),
            n_samples=n_samples,
            batch_size=batch_size,
            n_epochs=600,
            optimum=exact,
            seed=53,
        )
        return trace.distances[-1]

    full_batch = minimise(
        gradient, np.zeros(X.shape[1]), SGD(lr=0.2 / curvature), n_steps=600, optimum=exact
    )
    assert full_batch.distances[-1] < 1e-12
    assert floor_for(8) > 1e-3
    assert floor_for(64) < floor_for(8)
    # And the classical fix works: let the step size decay and the orbit closes.
    assert floor_for(8, decay=1e-2) < floor_for(8) / 100


def test_minibatch_gradients_average_to_the_full_gradient() -> None:
    """Every batch estimate is wrong; the mean over a partition of the data is exact."""
    X, y, _ = _problem(seed=54)
    gradient, batch_gradient, _ = least_squares_parts(X, y)
    w = np.arange(X.shape[1], dtype=np.float64)
    index = np.arange(X.shape[0])
    batches = [batch_gradient(w, chunk) for chunk in np.array_split(index, 10)]
    np.testing.assert_allclose(np.mean(batches, axis=0), gradient(w), rtol=1e-10)


def test_adam_bias_correction_is_present() -> None:
    """Without it the first step is ``lr * (1 - beta1) / sqrt(1 - beta2)`` times too small.

    With correction, the first step of Adam has length ``lr`` in every coordinate, whatever
    the gradient's magnitude — that is the property the method is named for.
    """
    optimizer = Adam(lr=0.1)
    w = np.zeros(3)
    optimizer.reset(w)
    moved = optimizer.step(w, np.array([5.0, -0.001, 100.0]))
    np.testing.assert_allclose(np.abs(moved - w), 0.1, rtol=1e-4)


def test_a_finished_run_records_one_distance_per_step() -> None:
    X, y, exact = _problem()
    gradient, _, curvature = least_squares_parts(X, y)
    trace = minimise(
        gradient, np.zeros(X.shape[1]), SGD(lr=0.5 / curvature), n_steps=25, optimum=exact
    )
    assert trace.n_steps == 25
    assert len(trace.distances) == 26
