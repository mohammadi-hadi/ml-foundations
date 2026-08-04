"""Least squares with a penalty on the size of the coefficients.

Lesson 1 ended with a problem no solver could fix: when two columns carry the same
information, the data does not say how to divide the credit between them, and the fitted
coefficients are whatever the noise suggested. A penalty resolves it by adding a preference
of your own — smaller coefficients — and paying for it in bias.

Two penalties, and the difference between them is not a matter of degree:

``Ridge``
    Penalises the sum of squares. The gradient of ``w²`` vanishes at zero, so shrinking a
    coefficient that is already small buys almost nothing and the solution keeps every
    feature with a small weight. Closed form, and the closed form is an SVD away from
    lesson 1's.
``Lasso``
    Penalises the sum of absolute values. The gradient of ``|w|`` does *not* vanish at
    zero — it jumps from -1 to +1 — so there is a finite range of evidence that the penalty
    can overpower entirely, and those coefficients come out as exactly zero. No closed form;
    coordinate descent instead.

**On matching the reference.** The two objectives are scaled differently in scikit-learn, and
the difference is not cosmetic. Ridge minimises ``||y - Xw||² + α||w||²`` while Lasso
minimises ``||y - Xw||² / (2n) + α||w||₁``. An implementation that uses one convention and
compares against the other looks broken at every value of α. The conventions here are
scikit-learn's, so that the tests can be exact rather than approximate.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def soft_threshold(z: Array | float, gamma: float) -> Array | float:
    """Move ``z`` towards zero by ``gamma``, stopping there. The lasso in one line.

    This is the exact minimiser of ``0.5(w - z)² + γ|w|``, and the reason lasso coefficients
    are *exactly* zero rather than very small: everything in ``[-γ, γ]`` maps to zero, not
    near it.
    """
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


class _Centred:
    """Shared handling of the intercept, which is never penalised.

    Shrinking the intercept would mean preferring models that predict near zero, which is a
    statement about the units the target happens to be measured in and not about the problem.
    Centring puts it outside the penalty by construction rather than by a special case in the
    solver.
    """

    fit_intercept: bool
    coef_: Array
    intercept_: float

    def _prepare(self, X: Array, y: Array) -> tuple[Array, Array, Array, float]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), float(y.mean())
            return X - x_mean, y - y_mean, x_mean, y_mean
        return X, y, np.zeros(X.shape[1]), 0.0

    def _finish(self, x_mean: Array, y_mean: float) -> None:
        self.intercept_ = y_mean - float(x_mean @ self.coef_) if self.fit_intercept else 0.0

    def predict(self, X: Array) -> Array:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


class Ridge(_Centred):
    """Minimises ``||y - Xw||² + alpha * ||w||²``.

    Solved through the SVD rather than by forming ``XᵀX + αI``, for the reason lesson 1 gave:
    the factorisation is available at the conditioning of ``X`` instead of its square. Written
    in terms of the singular values the whole method becomes legible — each one is replaced by
    ``s / (s² + α)`` instead of ``1 / s``, which leaves the large ones almost untouched and
    smoothly kills the small ones. Those small singular values are exactly the directions that
    made lesson 1's coefficients meaningless.
    """

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True) -> None:
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = np.empty(0)
        self.intercept_ = 0.0

    def fit(self, X: Array, y: Array) -> Self:
        Xc, yc, x_mean, y_mean = self._prepare(X, y)
        U, singular, Vt = np.linalg.svd(Xc, full_matrices=False)
        filtered = singular / (singular**2 + self.alpha)
        self.coef_ = Vt.T @ (filtered * (U.T @ yc))
        self._finish(x_mean, y_mean)
        return self


class Lasso(_Centred):
    """Minimises ``||y - Xw||² / (2n) + alpha * ||w||₁`` by cyclic coordinate descent.

    One coordinate at a time, each solved exactly with the current values of the others held
    fixed. That works because the non-smooth part of the objective is separable — the
    absolute value applies to each coefficient on its own — so the one-dimensional
    subproblems have the closed form in :func:`soft_threshold` and no line search is needed.

    Friedman, Hastie and Tibshirani (2010), *Regularization Paths for Generalized Linear
    Models via Coordinate Descent*, Journal of Statistical Software 33(1).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        max_iter: int = 2000,
        tol: float = 1e-8,
    ) -> None:
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = np.empty(0)
        self.intercept_ = 0.0
        self.n_iter_ = 0

    def fit(self, X: Array, y: Array) -> Self:
        Xc, yc, x_mean, y_mean = self._prepare(X, y)
        n_samples, n_features = Xc.shape
        w = np.zeros(n_features)
        # Column norms are fixed for the whole run, so they are computed once. A zero column
        # carries no information and its coefficient must stay at zero rather than divide.
        column_energy = np.einsum("ij,ij->j", Xc, Xc) / n_samples
        residual = yc.copy()

        self.n_iter_ = self.max_iter
        for iteration in range(self.max_iter):
            largest_change = 0.0
            for j in range(n_features):
                if column_energy[j] == 0.0:
                    continue
                previous = w[j]
                # Add this coordinate's own contribution back before re-solving for it.
                rho = float(Xc[:, j] @ residual) / n_samples + column_energy[j] * previous
                w[j] = float(soft_threshold(rho, self.alpha)) / column_energy[j]
                if w[j] != previous:
                    # Keep the residual in step instead of recomputing `y - Xw` per
                    # coordinate: an update touches one column, so the correction is a rank
                    # one change and the loop stays linear in the number of features.
                    residual -= Xc[:, j] * (w[j] - previous)
                    largest_change = max(largest_change, abs(w[j] - previous))
            if largest_change < self.tol:
                self.n_iter_ = iteration + 1
                break

        self.coef_ = w
        self._finish(x_mean, y_mean)
        return self


def coefficient_path(
    model: type[Ridge] | type[Lasso], X: Array, y: Array, alphas: NDArray[np.float64]
) -> Array:
    """Coefficients at each penalty strength, as a ``(len(alphas), n_features)`` array."""
    return np.array([model(alpha=float(a)).fit(X, y).coef_ for a in alphas])
