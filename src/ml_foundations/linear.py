"""Ordinary least squares, solved three ways.

All three solvers minimise the same thing — the sum of squared residuals — so on data that is
even mildly well behaved they return the same coefficients to eleven or twelve decimal
places. They are here because they stop agreeing exactly when it matters, and the way they
come apart is the first genuinely useful piece of numerical judgement in machine learning:

``normal``
    Form ``XᵀX`` and solve. It is the form every textbook derives, it is the fastest, and it
    squares the condition number of the problem before solving it — so it enters the solve
    having already thrown away half of the significant digits it was given.
``qr``
    Factor ``X = QR`` and back-substitute. Never forms ``XᵀX``, so it works at the
    conditioning of ``X`` rather than its square. This is what a numerical linear algebra
    course tells you to use, and it is what most libraries actually do.
``svd``
    Factor ``X = USVᵀ`` and invert the singular values that are large enough to trust. The
    slowest, the most robust, and the only one of the three that still returns something
    defensible when the columns are genuinely linearly dependent.

Lesson 1 measures the gap. The short version: the choice is irrelevant until it is decisive,
and the condition number tells you which regime you are in.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Method = Literal["normal", "qr", "svd"]


def condition_number(X: Array) -> float:
    """Ratio of largest to smallest singular value: how much the answer can move.

    Read it as digits. A condition number of ``10^k`` means a solve can lose about ``k`` of
    the sixteen significant decimal digits a float64 carries. Reaching ``10^16`` means there
    are none left, and the answer is whatever the rounding happened to produce.
    """
    return float(np.linalg.cond(X))


def back_substitute(R: Array, b: Array) -> Array:
    """Solve ``R w = b`` for upper-triangular ``R``, bottom row first.

    Spelled out rather than delegated because it is the step that makes QR worth doing: once
    the factorisation is in hand the remaining work is this loop, which touches each entry of
    ``R`` once and introduces no error beyond the arithmetic it performs.
    """
    n = R.shape[0]
    w = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if R[i, i] == 0.0:
            raise np.linalg.LinAlgError("triangular factor is singular")
        w[i] = (b[i] - R[i, i + 1 :] @ w[i + 1 :]) / R[i, i]
    return w


class LinearRegression:
    """Least squares with a choice of solver.

    ``fit_intercept`` is handled by centring the columns and the target rather than by
    appending a column of ones. The two are algebraically identical; centring is preferred
    because it keeps the intercept out of the matrix being factorised, where a column of ones
    alongside features on a different scale is itself a source of ill-conditioning — and, in
    the regularised version in lesson 3, because it keeps the intercept out of the penalty,
    where it does not belong.
    """

    def __init__(
        self,
        *,
        method: Method = "svd",
        fit_intercept: bool = True,
        rcond: float = 1e-12,
    ) -> None:
        self.method: Method = method
        self.fit_intercept = fit_intercept
        self.rcond = rcond
        self.coef_: Array = np.empty(0, dtype=np.float64)
        self.intercept_: float = 0.0
        self.rank_: int = 0
        self._fitted = False

    def fit(self, X: Array, y: Array) -> Self:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows and y has {y.shape[0]}")

        if self.fit_intercept:
            x_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc, yc = X - x_mean, y - y_mean
        else:
            x_mean = np.zeros(X.shape[1])
            y_mean = 0.0
            Xc, yc = X, y

        if self.method == "normal":
            self.coef_ = self._solve_normal(Xc, yc)
        elif self.method == "qr":
            self.coef_ = self._solve_qr(Xc, yc)
        elif self.method == "svd":
            self.coef_ = self._solve_svd(Xc, yc)
        else:  # pragma: no cover - Literal makes this unreachable from typed callers
            raise ValueError(f"unknown method {self.method!r}")

        self.intercept_ = y_mean - float(x_mean @ self.coef_)
        self._fitted = True
        return self

    def _solve_normal(self, X: Array, y: Array) -> Array:
        gram = X.T @ X
        self.rank_ = int(np.linalg.matrix_rank(gram))
        return np.linalg.solve(gram, X.T @ y).astype(np.float64)

    def _solve_qr(self, X: Array, y: Array) -> Array:
        Q, R = np.linalg.qr(X, mode="reduced")
        self.rank_ = int(np.linalg.matrix_rank(R))
        return back_substitute(R, Q.T @ y)

    def _solve_svd(self, X: Array, y: Array) -> Array:
        U, singular, Vt = np.linalg.svd(X, full_matrices=False)
        # Directions the data barely varies in carry almost no information about the
        # coefficient along them, and dividing by their singular value would amplify noise by
        # exactly the factor that makes the problem ill-conditioned. Dropping them returns
        # the minimum-norm solution instead: the honest answer to an under-determined
        # question, rather than a confident answer to a question the data cannot settle.
        cutoff = self.rcond * float(singular[0]) if singular.size else 0.0
        keep = singular > cutoff
        self.rank_ = int(np.sum(keep))
        inverse = np.where(keep, 1.0 / np.where(keep, singular, 1.0), 0.0)
        return (Vt.T @ (inverse * (U.T @ y))).astype(np.float64)

    def predict(self, X: Array) -> Array:
        if not self._fitted:
            raise RuntimeError("call fit before predict")
        return (np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_).astype(np.float64)
