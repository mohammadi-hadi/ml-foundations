"""Lesson 2: the three things that decide whether gradient descent works.

Every problem here is least squares, so the exact answer is available from lesson 1 and every
optimiser can be scored on how close it got to the right point rather than on whether it
stopped moving. Three questions get measured:

1. How large a step is too large? There is an exact answer for a quadratic, computable before
   training starts, and it is checked here from both sides.
2. How many steps does it take? The same condition number that broke the normal equations in
   lesson 1 sets the answer, and momentum changes the exponent.
3. What does using mini-batches cost? Speed early, and an error floor that no amount of
   further training removes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_foundations import datasets as ds
from ml_foundations.figures import ACCENT, MARKERS, MUTED, SHADES, figure
from ml_foundations.linear import LinearRegression, condition_number
from ml_foundations.optim import (
    SGD,
    Adam,
    Gradient,
    Optimizer,
    Trace,
    heavy_ball_parameters,
    hessian_spectrum,
    least_squares_parts,
    minimise,
    minimise_stochastic,
)
from ml_foundations.report import table

#: Below this, a distance is rounding error rather than a measurement, and its digits differ
#: between machines. Reporting the bound instead keeps the drift check meaningful.
NEGLIGIBLE = 1e-12
#: Significant figures for a step count. The step at which a smoothly converging run crosses a
#: threshold is not determined to more than this: near the end the error falls by a factor of
#: 1.0005 per step, so an arithmetic difference of a fraction of a per cent — the kind a
#: different BLAS produces — moves the crossing by tens of steps. Two figures is what the
#: measurement supports, and it is all the lesson's argument about scaling needs.
STEP_FIGURES = 2

TOLERANCE = 1e-8
SCALES = (0.3, 0.1, 0.05, 0.02)
BATCH_EPOCHS = (1, 3, 10, 30, 100, 300, 1000)
LR_MULTIPLES = (0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0)


def _distance(value: float) -> str:
    return f"< {NEGLIGIBLE:.0e}" if value < NEGLIGIBLE else f"{value:.1e}"


def _steps_taken(count: int) -> str:
    """A step count, rounded to :data:`STEP_FIGURES` significant figures."""
    if count == 0:
        return "0"
    magnitude = 10 ** max(0, len(str(count)) - STEP_FIGURES)
    return f"{round(count / magnitude) * magnitude:,}"


def _outcome(trace: Trace) -> str:
    if trace.diverged:
        return "diverged"
    if trace.steps_to_tolerance is None:
        return "did not arrive"
    return _steps_taken(trace.steps_to_tolerance)


def _centred(data: ds.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = data.X - data.X.mean(axis=0)
    y = data.y - data.y.mean()
    exact = LinearRegression(fit_intercept=False, method="svd").fit(X, y).coef_
    return X, y, exact


def _learning_rate_table(seed: int) -> str:
    """Step sizes either side of the stability limit, plus the one theory says is fastest.

    Two different quantities get confused constantly. ``2/L`` is where gradient descent stops
    converging; ``2/(L + m)`` is where it converges *fastest*. On a well-conditioned problem
    they are close together and it hardly matters. The gap between the two rows here is the
    price of aiming at the wrong one.
    """
    X, y, exact = _centred(ds.make_linear(n_samples=400, n_features=6, intercept=0.0, seed=seed))
    gradient, _, _ = least_squares_parts(X, y)
    L, m = hessian_spectrum(X)
    threshold = 2.0 / L
    fastest = 2.0 / (L + m)

    settings: list[tuple[str, float]] = [
        (f"{multiple:g} × 2/L", multiple * threshold) for multiple in LR_MULTIPLES
    ]
    settings.append(("2/(L + m) — the optimum", fastest))
    settings.sort(key=lambda item: item[1])

    rows = []
    for label, lr in settings:
        with np.errstate(over="ignore", invalid="ignore"):
            trace = minimise(
                gradient,
                np.zeros(X.shape[1]),
                SGD(lr=lr),
                n_steps=20_000,
                optimum=exact,
                tolerance=TOLERANCE,
            )
        rows.append(
            [
                label,
                f"{lr:.4f}",
                _outcome(trace),
                "—" if trace.diverged else _distance(trace.distances[-1]),
            ]
        )
    return table(
        ["Step size", "Value", "Steps to a relative error of 1e-8", "Final distance"], rows
    )


STEP_BUDGET = 300_000


def _steps(gradient: Gradient, exact: np.ndarray, optimizer: Optimizer) -> str:
    """Steps to reach the exact minimiser, or why it never did."""
    trace = minimise(
        gradient,
        np.zeros_like(exact),
        optimizer,
        n_steps=STEP_BUDGET,
        optimum=exact,
        tolerance=TOLERANCE,
    )
    if trace.steps_to_tolerance is None:
        return "diverged" if trace.diverged else f"> {STEP_BUDGET:,}"
    return _steps_taken(trace.steps_to_tolerance)


def _conditioning_table(seed: int) -> str:
    """Every optimiser at a setting derived from the problem, not from a search.

    A hyperparameter sweep would make each column the best of several runs, and the reader
    would learn what the winner scored rather than why. These settings are the classical
    ones: ``1/L`` for plain descent, Polyak's pair for momentum, and Adam at its published
    default. Whether that is generous to Adam is a fair question and the lesson says so.
    """
    rows = []
    for scale in SCALES:
        data = ds.make_collinear(n_samples=100, n_features=5, independent_scale=scale, seed=seed)
        X, y, exact = _centred(data)
        gradient, _, _ = least_squares_parts(X, y)
        L, m = hessian_spectrum(X)
        heavy_step, heavy_momentum = heavy_ball_parameters(L, m)

        rows.append(
            [
                f"{condition_number(X):.0f}",
                f"{L / m:.0f}",
                f"{np.sqrt(L / m):.0f}",
                _steps(gradient, exact, SGD(lr=1.0 / L)),
                _steps(gradient, exact, SGD(lr=heavy_step, momentum=heavy_momentum)),
                _steps(gradient, exact, Adam(lr=0.01)),
            ]
        )
    return table(
        [
            "Condition number of X",
            "…of the Hessian (κ)",
            "√κ",
            "Plain descent",
            "Momentum",
            "Adam",
        ],
        rows,
    )


def _batch_size_table(seed: int) -> tuple[str, dict[int, list[float]]]:
    data = ds.make_linear(n_samples=512, n_features=6, noise=1.0, intercept=0.0, seed=seed)
    X, y, exact = _centred(data)
    _, batch_gradient, curvature = least_squares_parts(X, y)

    sizes = (512, 64, 8, 1)
    traces = {
        size: minimise_stochastic(
            batch_gradient,
            np.zeros(X.shape[1]),
            SGD(lr=0.2 / curvature),
            n_samples=X.shape[0],
            batch_size=size,
            n_epochs=max(BATCH_EPOCHS),
            optimum=exact,
            seed=seed + 5,
        )
        for size in sizes
    }
    rows = [
        [f"{epoch:,}"] + [_distance(traces[size].distances[epoch]) for size in sizes]
        for epoch in BATCH_EPOCHS
    ]
    rendered = table(
        ["Passes over the data", "Full batch", "Batch of 64", "Batch of 8", "Batch of 1"],
        rows,
    )
    return rendered, {size: traces[size].distances for size in sizes}


def _decay_table(seed: int) -> str:
    data = ds.make_linear(n_samples=512, n_features=6, noise=1.0, intercept=0.0, seed=seed)
    X, y, exact = _centred(data)
    _, batch_gradient, curvature = least_squares_parts(X, y)

    rows = []
    for decay in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
        trace = minimise_stochastic(
            batch_gradient,
            np.zeros(X.shape[1]),
            SGD(lr=0.2 / curvature, decay=decay),
            n_samples=X.shape[0],
            batch_size=8,
            n_epochs=1000,
            optimum=exact,
            seed=seed + 5,
        )
        label = "none (fixed step)" if decay == 0.0 else f"{decay:g}"
        rows.append([label, _distance(trace.distances[-1])])
    return table(["Decay rate", "Distance after 1,000 passes"], rows)


def _plot(path: Path, curves: dict[int, list[float]]) -> None:
    with figure(path, size=(7.0, 4.2)) as ax:
        if ax is None:
            return
        for index, (size, distances) in enumerate(curves.items()):
            epochs = np.arange(len(distances))
            ax.semilogy(
                epochs[1:],
                np.maximum(distances[1:], 1e-16),
                marker=MARKERS[index % len(MARKERS)],
                markevery=max(1, len(distances) // 12),
                markersize=4,
                linewidth=1.5,
                color=SHADES[index % len(SHADES)],
                label="full batch" if size == 512 else f"batch of {size}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("passes over the data")
        ax.set_ylabel("relative distance from the exact minimiser")
        ax.set_title("Small batches are faster to start and never finish", fontsize=10)
        # Centre left is the only empty quadrant: the noisy runs fill the top and the
        # full-batch curve sweeps the diagonal.
        ax.legend(frameon=False, fontsize=8, loc="center left")
        ax.axhline(1e-12, color=MUTED, linewidth=1, linestyle="--")
        ax.text(1.2, 1.6e-12, "machine precision", fontsize=7, color=ACCENT)


def run(figures_dir: Path | None = None, *, seed: int = 2) -> dict[str, str]:
    batch_table, curves = _batch_size_table(seed)
    if figures_dir is not None:
        _plot(figures_dir / "02-batch-size.png", curves)
    return {
        "gd-learning-rate": _learning_rate_table(seed),
        "gd-conditioning": _conditioning_table(seed),
        "gd-batch-size": batch_table,
        "gd-decay": _decay_table(seed),
    }
