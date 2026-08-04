"""Logistic regression, fitted by iteratively reweighted least squares.

The model says the log-odds are linear in the features. Fitting it means maximising the
likelihood, and there is no closed form — but there is something better than a generic
optimiser. Newton's method applied to this likelihood turns out to be *least squares, done
repeatedly*: at each step, form a working response and a per-row weight, and solve exactly the
problem lesson 1 solved. That is where the name comes from, and it means the numerical
judgement from lesson 1 carries over unchanged, so the solve here goes through the same
SVD-based least squares rather than through an explicitly formed Hessian.

Convergence is quadratic — six or seven iterations from a cold start, not thousands — which is
worth knowing before reaching for gradient descent on a problem this shape.

**Scaling convention.** The objective is ``sum of negative log likelihood + 0.5 * alpha *
||w||²`` with the intercept left out of the penalty. scikit-learn parameterises the same
objective by ``C = 1 / alpha``, so ``alpha=0`` corresponds to ``penalty=None`` and ``alpha=2``
to ``C=0.5``. Getting this wrong is invisible at ``alpha=0`` and wrong everywhere else.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from ml_foundations.functions import sigmoid

Array = NDArray[np.float64]

#: Rows the model is already certain about carry weight ``p(1-p) ≈ 0``, and the working
#: response divides by exactly that. Flooring it keeps the step finite. The floor is small
#: enough that it never binds on a problem where the maximum likelihood estimate exists, and
#: on a separable problem — where it does not — it is what stops the fit from producing nan
#: instead of the large coefficients that are the honest answer.
MIN_WEIGHT = 1e-10


class LogisticRegression:
    """Binary logistic regression by iteratively reweighted least squares.

    ``alpha`` is the strength of an L2 penalty on the coefficients. It defaults to zero, which
    is the maximum likelihood estimate — and which, on data that can be separated by a
    hyperplane, does not exist: the likelihood keeps improving as the coefficients grow
    without bound. Lesson 4 measures that happening and shows the penalty stopping it.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> None:
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Array = np.empty(0)
        self.intercept_: float = 0.0
        self.n_iter_ = 0

    def _design(self, X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        if not self.fit_intercept:
            return X
        return np.column_stack([X, np.ones(X.shape[0])])

    def fit(self, X: Array, y: Array) -> Self:
        design = self._design(X)
        y = np.asarray(y, dtype=np.float64)
        n_columns = design.shape[1]

        # The intercept column, if there is one, is last and is not penalised: shrinking it
        # would express a preference about the base rate, which is not what the penalty is for.
        penalised = np.ones(n_columns)
        if self.fit_intercept:
            penalised[-1] = 0.0
        penalty_rows = np.sqrt(self.alpha) * np.diag(penalised)

        w = np.zeros(n_columns)
        self.n_iter_ = self.max_iter
        for iteration in range(self.max_iter):
            eta = design @ w
            p = sigmoid(eta)
            weight = np.maximum(p * (1.0 - p), MIN_WEIGHT)
            # The working response: where a weighted least squares fit would have to land to
            # take the Newton step. Everything below is lesson 1 applied to (√W X, √W z).
            working = eta + (y - p) / weight
            root = np.sqrt(weight)

            A = np.vstack([root[:, None] * design, penalty_rows])
            b = np.concatenate([root * working, np.zeros(n_columns)])
            updated, *_ = np.linalg.lstsq(A, b, rcond=None)

            step = float(np.max(np.abs(updated - w)))
            w = updated
            if step < self.tol:
                self.n_iter_ = iteration + 1
                break

        if self.fit_intercept:
            self.coef_, self.intercept_ = w[:-1], float(w[-1])
        else:
            self.coef_, self.intercept_ = w, 0.0
        return self

    def decision_function(self, X: Array) -> Array:
        """The log-odds. Every metric that needs a ranking should use this, not the probability.

        The two give identical rankings, since the logistic function is increasing, but the
        log-odds keep their resolution out in the tails where the probability has rounded to
        zero or one.
        """
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X: Array) -> Array:
        return sigmoid(self.decision_function(X))

    def predict(self, X: Array, *, threshold: float = 0.5) -> Array:
        """Hard labels. ``threshold`` is a choice, and lesson 4 is largely about that choice.

        A default of 0.5 is not a neutral one — it is the decision that minimises the error
        count when the two kinds of error cost the same and the classes are balanced. Neither
        condition holds often.
        """
        return (self.predict_proba(X) >= threshold).astype(np.float64)
