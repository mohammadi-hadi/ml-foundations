"""Lesson 1: what the choice of least-squares solver is worth, and when.

Two measurements, deliberately kept apart, because conflating them is the usual mistake:

*Numerical* damage is the solvers disagreeing with each other on the same data. Exact
arithmetic would make it zero, so whatever is left is the algorithm's fault.

*Statistical* damage is every solver being far from the truth together. No algorithm can fix
it, because the data does not contain the answer; only more data or a different question can.

Ill-conditioning produces both, and a lesson that shows only the first leaves the reader
believing a better solver would have saved them.

Numerical damage is reported as a count of surviving digits rather than as a floating-point
distance. That is not a simplification for the reader's benefit: the distance itself is
rounding error, and rounding error depends on which BLAS the machine has, so a table of
mantissas would differ between this laptop and the continuous integration runner and the
drift check would have to be switched off. The count of digits is a property of the problem.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ml_foundations import datasets as ds
from ml_foundations.figures import ACCENT, ALARM, MUTED, figure
from ml_foundations.linear import LinearRegression, condition_number
from ml_foundations.metrics import rmse
from ml_foundations.report import fmt, table

METHODS = ("normal", "qr", "svd")
SCALES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
#: float64 carries about sixteen significant decimal digits; nothing can share more.
FLOAT64_DIGITS = 16


def shared_digits(estimate: np.ndarray, reference: np.ndarray) -> int:
    """How many significant decimal digits two solutions of the same problem agree on."""
    scale = float(np.linalg.norm(reference))
    gap = float(np.linalg.norm(estimate - reference))
    if gap == 0.0 or scale == 0.0:
        return FLOAT64_DIGITS
    return max(0, min(FLOAT64_DIGITS, math.floor(-math.log10(gap / scale))))


def _fit_all(X: np.ndarray, y: np.ndarray) -> dict[str, LinearRegression | None]:
    """Fit every solver, recording a refusal as a refusal rather than crashing the report."""
    out: dict[str, LinearRegression | None] = {}
    for method in METHODS:
        try:
            out[method] = LinearRegression(method=method).fit(X, y)  # type: ignore[arg-type]
        except np.linalg.LinAlgError:
            out[method] = None
    return out


def _solver_agreement(seed: int) -> str:
    data = ds.make_linear(n_samples=400, n_features=8, noise=1.0, seed=seed)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=seed)
    assert data.coef is not None

    fits = _fit_all(X_train, y_train)
    reference = fits["svd"]
    assert reference is not None

    rows = []
    for method in METHODS:
        fitted = fits[method]
        assert fitted is not None
        rows.append(
            [
                f"`{method}`",
                "—" if method == "svd" else str(shared_digits(fitted.coef_, reference.coef_)),
                fmt(float(np.linalg.norm(fitted.coef_ - data.coef))),
                fmt(rmse(y_test, fitted.predict(X_test))),
            ]
        )
    return table(
        ["Solver", "Digits shared with `svd`", "Distance from the truth", "Test RMSE"],
        rows,
    )


def _conditioning_sweep(seed: int) -> tuple[str, list[tuple[float, float, float, float]]]:
    rows = []
    plot_data: list[tuple[float, float, float, float]] = []
    for scale in SCALES:
        data = ds.make_collinear(
            n_samples=400, n_features=6, independent_scale=scale, noise=1.0, seed=seed
        )
        X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=seed)
        assert data.coef is not None

        kappa = condition_number(X_train)
        fits = _fit_all(X_train, y_train)
        reference = fits["svd"]
        assert reference is not None

        digits = {}
        raw_gap = {}
        for method in ("normal", "qr"):
            fitted = fits[method]
            digits[method] = (
                "refused" if fitted is None else str(shared_digits(fitted.coef_, reference.coef_))
            )
            raw_gap[method] = (
                float("nan")
                if fitted is None
                else float(np.linalg.norm(fitted.coef_ - reference.coef_))
            )

        truth_gap = float(np.linalg.norm(reference.coef_ - data.coef))
        test_error = rmse(y_test, reference.predict(X_test))

        rows.append(
            [
                f"{kappa:.0e}",
                digits["normal"],
                digits["qr"],
                f"{truth_gap:.1e}",
                fmt(test_error),
            ]
        )
        plot_data.append((kappa, raw_gap["normal"], raw_gap["qr"], test_error))

    rendered = table(
        [
            "Condition number of X",
            "Digits kept by `normal`",
            "Digits kept by `qr`",
            "Distance from the truth",
            "Test RMSE",
        ],
        rows,
    )
    return rendered, plot_data


def _headline(seed: int) -> str:
    """The single comparison the README quotes: same data, same objective, different answers."""
    data = ds.make_collinear(n_samples=400, n_features=6, independent_scale=1e-6, seed=seed)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=seed)
    fits = _fit_all(X_train, y_train)
    reference = fits["svd"]
    assert reference is not None
    rows = []
    for method in METHODS:
        fitted = fits[method]
        if fitted is None:
            rows.append([f"`{method}`", "refused to solve", "—"])
            continue
        rows.append(
            [
                f"`{method}`",
                "— (reference)"
                if method == "svd"
                else str(shared_digits(fitted.coef_, reference.coef_)),
                fmt(rmse(y_test, fitted.predict(X_test))),
            ]
        )
    return table(["Solver", "Correct digits in the coefficients", "Test RMSE"], rows)


def _plot(path: Path, plot_data: list[tuple[float, float, float, float]]) -> None:
    with figure(path, size=(7.0, 4.2)) as ax:
        if ax is None:
            return
        kappa = [row[0] for row in plot_data]
        ax.loglog(
            kappa,
            [max(row[1], 1e-18) for row in plot_data],
            marker="o",
            color=ALARM,
            label="normal equations, distance from the SVD solution",
        )
        ax.loglog(
            kappa,
            [max(row[2], 1e-18) for row in plot_data],
            marker="s",
            color=ACCENT,
            label="QR, distance from the SVD solution",
        )
        ax.loglog(
            kappa,
            [row[3] for row in plot_data],
            marker="^",
            color=MUTED,
            label="test RMSE, every solver",
        )
        ax.set_xlabel("condition number of the design matrix")
        ax.set_ylabel("error")
        ax.set_title(
            "Ill-conditioning wrecks the coefficients and leaves the predictions alone",
            fontsize=10,
        )
        ax.legend(frameon=False, fontsize=8, loc="upper left")


def run(figures_dir: Path | None = None, *, seed: int = 1) -> dict[str, str]:
    sweep, plot_data = _conditioning_sweep(seed)
    if figures_dir is not None:
        _plot(figures_dir / "01-conditioning.png", plot_data)
    return {
        "ols-solver-agreement": _solver_agreement(seed),
        "ols-conditioning": sweep,
        "ols-headline": _headline(seed),
    }
