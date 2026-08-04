"""First-order optimisers, and a driver that records what they did on the way.

Everything here follows the gradient downhill. The differences are in what each method
remembers between steps: nothing, a running average of past directions, or a running average
of past *squared* directions used to give every coordinate its own step size.

The problems in lesson 2 are least-squares problems, chosen for one reason: their minimum is
available in closed form from lesson 1. An optimiser can therefore be checked against the
right answer rather than against its own convergence, which is the difference between knowing
that it stopped and knowing that it arrived.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
#: Row positions, not values. Kept distinct so a batch index cannot be passed as data.
IndexArray = NDArray[np.intp]
Gradient = Callable[[Array], Array]
BatchGradient = Callable[[Array, IndexArray], Array]


class Optimizer:
    """Base class. A subclass turns a gradient into a step, and may keep state to do it."""

    def reset(self, w: Array) -> None:
        """Discard any state, sized for the parameter vector about to be optimised."""

    def step(self, w: Array, grad: Array) -> Array:
        raise NotImplementedError


@dataclass
class SGD(Optimizer):
    """Gradient descent, optionally with momentum and a decaying step size.

    Momentum accumulates an exponentially weighted average of past gradients and follows
    that instead of the current one. On a bowl-shaped surface whose sides differ in
    steepness, the plain method spends its time bouncing across the narrow direction; the
    average of those bounces cancels, and what survives is the slow drift along the valley
    floor. That is the entire trick, and lesson 2 measures what it is worth: convergence in
    time proportional to the square root of the condition number rather than to the condition
    number itself.

    ``decay`` shrinks the step size as ``lr / (1 + decay * t)``. On full-batch gradients it
    is a pessimisation, and on mini-batch gradients it is the difference between arriving and
    circling: a batch gradient is the true gradient plus noise, a fixed step size keeps
    reinjecting that noise, and the run settles into an orbit whose radius is set by the step
    size rather than by how long you were willing to wait. The harmonic schedule is the
    classical answer — the steps still sum to infinity, so nothing is out of reach, but their
    squares sum to something finite, so the noise is eventually damped out.
    """

    lr: float
    momentum: float = 0.0
    decay: float = 0.0
    velocity: Array = field(default_factory=lambda: np.empty(0), repr=False)
    t: int = 0

    def reset(self, w: Array) -> None:
        self.velocity = np.zeros_like(w)
        self.t = 0

    def step(self, w: Array, grad: Array) -> Array:
        lr = self.lr / (1.0 + self.decay * self.t)
        self.t += 1
        self.velocity = self.momentum * self.velocity - lr * grad
        return w + self.velocity


@dataclass
class Adam(Optimizer):
    """Adaptive moment estimation.

    Keeps a running mean of the gradient and a running mean of its square, and divides one by
    the root of the other, so every coordinate ends up with a step size scaled to how large
    its own gradients have been. The bias correction matters more than it looks: both running
    means start at zero, so without dividing by ``1 - beta**t`` the first steps are far too
    small, and with ``beta2 = 0.999`` "the first steps" means the first several hundred.

    Kingma and Ba (2015), *Adam: A Method for Stochastic Optimization*, ICLR.
    """

    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    mean: Array = field(default_factory=lambda: np.empty(0), repr=False)
    mean_square: Array = field(default_factory=lambda: np.empty(0), repr=False)
    t: int = 0

    def reset(self, w: Array) -> None:
        self.mean = np.zeros_like(w)
        self.mean_square = np.zeros_like(w)
        self.t = 0

    def step(self, w: Array, grad: Array) -> Array:
        self.t += 1
        self.mean = self.beta1 * self.mean + (1 - self.beta1) * grad
        self.mean_square = self.beta2 * self.mean_square + (1 - self.beta2) * grad**2
        mean_hat = self.mean / (1 - self.beta1**self.t)
        mean_square_hat = self.mean_square / (1 - self.beta2**self.t)
        return w - self.lr * mean_hat / (np.sqrt(mean_square_hat) + self.eps)


@dataclass(frozen=True)
class Trace:
    """What happened during a run, as opposed to only where it stopped."""

    weights: Array
    distances: list[float]
    diverged: bool
    steps_to_tolerance: int | None

    @property
    def n_steps(self) -> int:
        return len(self.distances) - 1


def minimise(
    gradient: Gradient,
    w0: Array,
    optimizer: Optimizer,
    *,
    n_steps: int,
    optimum: Array,
    tolerance: float = 1e-8,
) -> Trace:
    """Run ``optimizer`` on ``gradient`` and record the distance to the known ``optimum``.

    Measuring against the true minimiser rather than against the change between steps is a
    deliberate choice, and only possible because these problems have closed-form solutions.
    A method that has slowed to a crawl looks converged by any self-referential criterion,
    which is exactly the situation lesson 2 is about.

    Divergence is detected and stopped rather than allowed to run to ``inf``: an overflowing
    run costs the same wall-clock as a converging one and produces warnings instead of data.
    """
    optimizer.reset(w0)
    scale = float(np.linalg.norm(optimum)) or 1.0
    w = w0.astype(np.float64).copy()
    distances = [float(np.linalg.norm(w - optimum)) / scale]
    steps_to_tolerance: int | None = 0 if distances[0] <= tolerance else None

    for step in range(1, n_steps + 1):
        w = optimizer.step(w, gradient(w))
        # The distance doubles as the divergence check. A parameter that has overflowed to
        # inf or nan produces a distance that is inf or nan, so testing the scalar catches
        # everything a full `isfinite` pass over the array would — for a fraction of the
        # cost, which matters when the loop runs a quarter of a million times.
        offset = w - optimum
        relative = float(np.sqrt(offset @ offset)) / scale
        if not np.isfinite(relative):
            return Trace(w, distances, diverged=True, steps_to_tolerance=None)
        distances.append(relative)
        if steps_to_tolerance is None and relative <= tolerance:
            steps_to_tolerance = step
    return Trace(
        w, distances, diverged=distances[-1] > distances[0], steps_to_tolerance=steps_to_tolerance
    )


def minimise_stochastic(
    batch_gradient: BatchGradient,
    w0: Array,
    optimizer: Optimizer,
    *,
    n_samples: int,
    batch_size: int,
    n_epochs: int,
    optimum: Array,
    seed: int = 0,
) -> Trace:
    """The same loop over shuffled mini-batches, recording distance once per epoch.

    Per epoch rather than per step, so that runs with different batch sizes are compared on
    equal terms: one pass over the data is the unit of work everyone pays the same price for.
    """
    optimizer.reset(w0)
    rng = np.random.default_rng(seed)
    scale = float(np.linalg.norm(optimum)) or 1.0
    w = w0.astype(np.float64).copy()
    distances = [float(np.linalg.norm(w - optimum)) / scale]

    for _ in range(n_epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            w = optimizer.step(w, batch_gradient(w, order[start : start + batch_size]))
            if not np.all(np.isfinite(w)):
                return Trace(w, distances, diverged=True, steps_to_tolerance=None)
        distances.append(float(np.linalg.norm(w - optimum)) / scale)
    return Trace(w, distances, diverged=distances[-1] > distances[0], steps_to_tolerance=None)


def least_squares_parts(
    X: Array, y: Array
) -> tuple[Gradient, BatchGradient, float]:
    """Gradient, mini-batch gradient, and the largest curvature, for ``mean((Xw - y)**2)``.

    The third return value is what makes lesson 2's central prediction possible. The Hessian
    of this objective is ``2XᵀX / n``, a constant, so its largest eigenvalue ``L`` is the
    steepest curvature anywhere on the surface — and gradient descent on a quadratic is stable
    if and only if the learning rate is below ``2 / L``. That threshold is not a rule of
    thumb. It is exact, it is computable before training starts, and lesson 2 checks it.
    """
    n = X.shape[0]

    def gradient(w: Array) -> Array:
        return np.asarray((2.0 / n) * (X.T @ (X @ w - y)), dtype=np.float64)

    def batch_gradient(w: Array, index: IndexArray) -> Array:
        Xb, yb = X[index], y[index]
        return np.asarray((2.0 / len(index)) * (Xb.T @ (Xb @ w - yb)), dtype=np.float64)

    curvature = 2.0 * float(np.linalg.eigvalsh(X.T @ X).max()) / n
    return gradient, batch_gradient, curvature


def hessian_spectrum(X: Array) -> tuple[float, float]:
    """Largest and smallest curvature of ``mean((Xw - y)**2)``, as ``(L, m)``.

    Their ratio is the condition number of the Hessian, and it is the quantity that decides
    how long gradient descent takes: the error contracts by ``1 - m/L`` per step, so a
    hundredfold worse conditioning is a hundredfold longer wait.
    """
    n = X.shape[0]
    eigenvalues = np.linalg.eigvalsh(X.T @ X) * (2.0 / n)
    return float(eigenvalues[-1]), float(max(eigenvalues[0], 0.0))


def heavy_ball_parameters(L: float, m: float) -> tuple[float, float]:
    """Polyak's step size and momentum for a quadratic with curvature between ``m`` and ``L``.

    These are not defaults anybody tuned. They are the pair that provably minimises the worst
    case, and they turn the ``κ`` steps that plain descent needs into ``√κ`` — which is the
    entire reason momentum exists, and why lesson 2 can predict the second column of its
    convergence table before running it.

    Polyak (1964), *Some methods of speeding up the convergence of iteration methods*.
    """
    root = np.sqrt(L) + np.sqrt(m)
    step = 4.0 / (root * root)
    ratio = np.sqrt(L / m) if m > 0 else np.inf
    momentum = ((ratio - 1.0) / (ratio + 1.0)) ** 2 if np.isfinite(ratio) else 1.0
    return float(step), float(momentum)
