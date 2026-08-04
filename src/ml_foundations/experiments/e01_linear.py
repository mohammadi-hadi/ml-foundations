"""Lesson 1: what the choice of least-squares solver is worth, and when.

Two measurements, deliberately kept apart, because conflating them is the usual mistake:

*Numerical* damage is the solvers disagreeing with each other on the same data. Exact
arithmetic would make it zero, so whatever is left is the algorithm's fault.

*Statistical* damage is every solver being far from the truth together. No algorithm can fix
it, because the data does not contain the answer; only more data or a different question can.

Ill-conditioning produces both, and a lesson that shows only the first leaves the reader
believing a better solver would have saved them.

Numerical damage is reported as a count of surviving digits rather than as a floating-point
distance. That is not a simplification for the reader's benefit: the distance itself *is*
rounding error, and rounding error depends on which BLAS the machine has, so a table of
mantissas would differ between this laptop and the continuous integration runner and the
drift check would have to be switched off.

Two further precautions make the count itself stable, and both were added after measuring how
much it moves. First, each figure is the **median over nine datasets** rather than one draw:
a single draw varies by about half a decade from seed to seed, which is the same size as the
variation a different BLAS would cause, and a cell sitting a tenth of a decade from an integer
boundary would flip between machines. Second, anything at or above fourteen digits is reported
as ``>= 14`` rather than as a number, because the difference between fourteen and fifteen
significant digits out of a possible sixteen is not a measurement of anything — it is the last
bits of an arithmetic that got the answer right.
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
#: At or above this, nothing measurable was lost and the exact count is not a measurement.
FULL_PRECISION = 14
#: Datasets each figure is a median over. Nine is enough to stop the seed deciding the digit.
REPLICATES = tuple(range(1, 10))


def shared_digits(estimate: np.ndarray, reference: np.ndarray) -> float:
    """How many significant decimal digits two solutions of the same problem agree on."""
    scale = float(np.linalg.norm(reference))
    gap = float(np.linalg.norm(estimate - reference))
    if gap == 0.0 or scale == 0.0:
        return float(FLOAT64_DIGITS)
    return max(0.0, min(float(FLOAT64_DIGITS), -math.log10(gap / scale)))


def render_digits(values: list[float]) -> str:
    """Median digit count, floored, with everything near machine precision collapsed."""
    median = float(np.median(values))
    return f">= {FULL_PRECISION}" if median >= FULL_PRECISION else str(math.floor(median))


def render_digits_lost(values: list[float], reference: list[float]) -> str:
    """Digits lost *relative to the best-conditioned problem in the sweep*.

    Reported as a difference rather than as an absolute count for a reason worth stating: the
    absolute count is about half a digit lower on the continuous integration runner than on
    the machine these numbers were generated on, because Linux and macOS ship different BLAS
    libraries and the two accumulate ``XᵀX`` in a different order. That offset applies to
    every row at once, so subtracting the first row removes it entirely and leaves the thing
    the lesson is actually claiming — the *rate* at which digits are lost — which is a
    property of the arithmetic rather than of the vendor who implemented it.
    """
    lost = float(np.median(reference)) - float(np.median(values))
    return str(max(0, round(lost)))


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
    """On well-behaved data the three solvers are the same estimator. Median over replicates."""
    digits: dict[str, list[float]] = {method: [] for method in METHODS}
    truth_gap: dict[str, list[float]] = {method: [] for method in METHODS}
    test_error: dict[str, list[float]] = {method: [] for method in METHODS}

    for replicate in REPLICATES:
        data = ds.make_linear(n_samples=400, n_features=8, noise=1.0, seed=seed * replicate)
        X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=replicate)
        assert data.coef is not None
        fits = _fit_all(X_train, y_train)
        reference = fits["svd"]
        assert reference is not None
        for method in METHODS:
            fitted = fits[method]
            assert fitted is not None
            digits[method].append(shared_digits(fitted.coef_, reference.coef_))
            truth_gap[method].append(float(np.linalg.norm(fitted.coef_ - data.coef)))
            test_error[method].append(rmse(y_test, fitted.predict(X_test)))

    rows = [
        [
            f"`{method}`",
            "— (reference)" if method == "svd" else render_digits(digits[method]),
            fmt(float(np.median(truth_gap[method]))),
            fmt(float(np.median(test_error[method]))),
        ]
        for method in METHODS
    ]
    return table(
        ["Solver", "Digits shared with `svd`", "Distance from the truth", "Test RMSE"],
        rows,
    )


def _sweep_row(scale: float, seed: int) -> tuple[float, dict[str, float], float, float]:
    """One dataset at one conditioning level: κ, per-solver digits, truth gap, test error."""
    data = ds.make_collinear(
        n_samples=400, n_features=6, independent_scale=scale, noise=1.0, seed=seed
    )
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=seed)
    assert data.coef is not None
    fits = _fit_all(X_train, y_train)
    reference = fits["svd"]
    assert reference is not None

    digits = {
        method: float("nan")
        if fits[method] is None
        else shared_digits(fits[method].coef_, reference.coef_)  # type: ignore[union-attr]
        for method in ("normal", "qr")
    }
    return (
        condition_number(X_train),
        digits,
        float(np.linalg.norm(reference.coef_ - data.coef)),
        rmse(y_test, reference.predict(X_test)),
    )


def _conditioning_sweep(seed: int) -> tuple[str, list[tuple[float, float, float, float]]]:
    rows = []
    plot_data: list[tuple[float, float, float, float]] = []
    baseline: list[float] = []
    for scale in SCALES:
        measured = [_sweep_row(scale, seed * replicate) for replicate in REPLICATES]
        kappa = float(np.median([row[0] for row in measured]))
        normal = [row[1]["normal"] for row in measured]
        qr = [row[1]["qr"] for row in measured]
        if not baseline:
            baseline = normal
        truth_gap = float(np.median([row[2] for row in measured]))
        test_error = float(np.median([row[3] for row in measured]))

        rows.append(
            [
                f"{kappa:.0e}",
                "refused" if np.isnan(normal).all() else render_digits_lost(normal, baseline),
                render_digits(qr),
                f"{truth_gap:.1e}",
                fmt(test_error),
            ]
        )
        plot_data.append((kappa, float(np.median(normal)), float(np.median(qr)), test_error))

    rendered = table(
        [
            "Condition number of X",
            "Digits lost by `normal`",
            "Digits kept by `qr`",
            "Distance from the truth",
            "Test RMSE",
        ],
        rows,
    )
    return rendered, plot_data


def _rank_deficient(seed: int) -> str:
    """Push past *nearly* dependent to *exactly* dependent, and watch the three come apart.

    Every number here is categorical or robust on purpose. Whether a solver raises is a fact,
    not a measurement; the rank the SVD reports is an integer; and the coefficient sizes differ
    by thirteen orders of magnitude, so a bucket separates them with room to spare.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((200, 3))
    X = np.column_stack([base, base[:, 0]])
    y = base @ np.array([2.0, -1.0, 0.5]) + rng.standard_normal(200) * 0.1
    X_train, X_test, y_train, y_test = ds.train_test_split(X, y, seed=seed)

    rows = []
    for method in METHODS:
        try:
            fitted = LinearRegression(method=method).fit(X_train, y_train)  # type: ignore[arg-type]
        except np.linalg.LinAlgError:
            rows.append([f"`{method}`", "refuses to solve", "—", "—"])
            continue
        size = float(np.linalg.norm(fitted.coef_))
        rows.append(
            [
                f"`{method}`",
                "returns an answer",
                "larger than 1e10" if size > 1e10 else fmt(size),
                fmt(rmse(y_test, fitted.predict(X_test)), 2),
            ]
        )
    return table(
        ["Solver", "On an exactly duplicated column", "Size of the coefficients", "Test RMSE"],
        rows,
    )


def _plot(path: Path, plot_data: list[tuple[float, float, float, float]]) -> None:
    with figure(path, size=(7.0, 4.2)) as ax:
        if ax is None:
            return
        kappa = [row[0] for row in plot_data]
        ax.semilogx(
            kappa,
            [row[1] for row in plot_data],
            marker="o",
            color=ALARM,
            linewidth=1.8,
            label="normal equations",
        )
        ax.semilogx(
            kappa,
            [row[2] for row in plot_data],
            marker="s",
            color=ACCENT,
            linewidth=1.8,
            label="QR",
        )
        ax.axhline(FLOAT64_DIGITS, color=MUTED, linestyle="--", linewidth=1)
        ax.text(kappa[0], FLOAT64_DIGITS + 0.25, "everything float64 has", fontsize=7, color=MUTED)
        ax.set_ylim(0, FLOAT64_DIGITS + 1.5)
        ax.set_xlabel("condition number of the design matrix")
        ax.set_ylabel("correct digits in the coefficients")
        ax.set_title("Two digits lost per decade, and the predictions never notice", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower left")


def run(figures_dir: Path | None = None, *, seed: int = 1) -> dict[str, str]:
    sweep, plot_data = _conditioning_sweep(seed)
    if figures_dir is not None:
        _plot(figures_dir / "01-conditioning.png", plot_data)
    return {
        "ols-solver-agreement": _solver_agreement(seed),
        "ols-conditioning": sweep,
        "ols-rank-deficient": _rank_deficient(seed),
    }
