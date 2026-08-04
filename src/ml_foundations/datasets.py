"""Seeded data generators whose true parameters are known by construction.

Almost every dataset here is synthetic on purpose. When you write an estimator yourself
you need to answer a question no benchmark score can answer: *is it right?* Synthetic data
answers it directly — you drew the coefficients, so you can check whether the estimator
recovers them, and you know what the best achievable error is because you chose the noise.

Two generators are deliberately degenerate. :func:`make_collinear` produces a design matrix
that is nearly rank-deficient, so that a method which "works" on well-behaved data can be
shown failing. :func:`make_noise` produces features and labels that are independent, so the
honest score is exactly chance — anything above it is a bug in the evaluation, and lesson 6
uses it to catch one.

Every function takes a ``seed`` and uses it for all randomness, so results reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ml_foundations.functions import sigmoid

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Dataset:
    """Features, targets, and — where it exists — the truth they were generated from.

    ``coef`` and ``intercept`` are ``None`` only when the generating process has no linear
    parameters to report (a nonlinear surface, or pure noise).
    """

    X: Array
    y: Array
    name: str
    coef: Array | None = None
    intercept: float | None = None

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_linear(
    n_samples: int = 200,
    n_features: int = 5,
    *,
    noise: float = 1.0,
    seed: int = 0,
    intercept: float = 3.0,
) -> Dataset:
    """A well-conditioned linear problem: independent standard-normal features.

    The irreducible error is ``noise``: no estimator can do better than that RMSE on fresh
    data, which is the number to compare a fitted model against.
    """
    rng = _rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    coef = rng.uniform(-3.0, 3.0, size=n_features)
    y = X @ coef + intercept + rng.standard_normal(n_samples) * noise
    return Dataset(X=X, y=y, name="linear", coef=coef, intercept=intercept)


def make_collinear(
    n_samples: int = 200,
    n_features: int = 6,
    *,
    independent_scale: float = 1e-4,
    noise: float = 1.0,
    seed: int = 0,
) -> Dataset:
    """A linear problem whose features are nearly the same feature.

    Every column is one shared latent factor plus ``independent_scale`` times its own noise.
    Shrink that scale and the columns approach linear dependence: the condition number of
    the design matrix runs at roughly ``sqrt(n_features) / independent_scale``, so the
    default of ``1e-4`` buys about four orders of magnitude of trouble.

    The regression *function* stays perfectly learnable — predictions remain accurate,
    because the direction the data actually varies in is still well measured. It is the
    individual coefficients that stop being identifiable, and lesson 1 measures exactly that
    gap between a model that predicts well and parameters that mean anything.
    """
    if independent_scale <= 0.0:
        raise ValueError("independent_scale must be positive; at zero the columns are identical")
    rng = _rng(seed)
    latent = rng.standard_normal(n_samples)
    private = rng.standard_normal((n_samples, n_features))
    X = latent[:, None] + independent_scale * private
    coef = rng.uniform(-3.0, 3.0, size=n_features)
    y = X @ coef + rng.standard_normal(n_samples) * noise
    return Dataset(X=X, y=y, name="collinear", coef=coef, intercept=0.0)


def make_sparse_linear(
    n_samples: int = 120,
    n_features: int = 40,
    *,
    n_informative: int = 5,
    noise: float = 1.0,
    seed: int = 0,
) -> Dataset:
    """More features than the signal needs: only the first ``n_informative`` matter.

    The rest have a true coefficient of exactly zero. This is the setting where the choice
    of penalty stops being cosmetic — lesson 3 shows one penalty shrinking the irrelevant
    coefficients towards zero and another setting them to it.
    """
    rng = _rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    coef = np.zeros(n_features)
    coef[:n_informative] = rng.uniform(2.0, 5.0, size=n_informative) * rng.choice(
        [-1.0, 1.0], size=n_informative
    )
    y = X @ coef + rng.standard_normal(n_samples) * noise
    return Dataset(X=X, y=y, name="sparse-linear", coef=coef, intercept=0.0)


def make_logistic(
    n_samples: int = 400,
    n_features: int = 4,
    *,
    positive_rate: float = 0.5,
    separation: float = 1.5,
    seed: int = 0,
) -> Dataset:
    """A binary problem with a controllable base rate.

    ``positive_rate`` is achieved by solving for the intercept that produces it (see
    :func:`_intercept_for_rate`) rather than by resampling, so the two classes keep the same
    feature distribution and only their proportions change. That is what makes the imbalance
    in lesson 4 a fair comparison: nothing differs except how rare the positive class is.
    """
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("positive_rate must be strictly between 0 and 1")
    rng = _rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    coef = rng.uniform(-1.0, 1.0, size=n_features) * separation
    scores = X @ coef
    intercept = _intercept_for_rate(scores, positive_rate)
    y = (rng.random(n_samples) < sigmoid(scores + intercept)).astype(np.float64)
    return Dataset(X=X, y=y, name="logistic", coef=coef, intercept=intercept)


def _intercept_for_rate(scores: Array, positive_rate: float) -> float:
    """Find the shift that makes the *average predicted probability* equal ``positive_rate``.

    The obvious shortcut — put the intercept at the matching quantile of the scores — sets
    the fraction of rows whose score clears zero, which is not the same thing and is wrong
    in a direction that matters. The sigmoid squashes everything towards one half, so asking
    for a 20% positive rate that way delivers 28%. Mean probability is monotone in the
    intercept, so bisection solves for it directly; 200 halvings on a 200-wide bracket is
    exact to well past float64's resolution.
    """
    low, high = -100.0, 100.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if float(sigmoid(scores + middle).mean()) < positive_rate:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def make_friedman1(n_samples: int = 400, *, noise: float = 1.0, seed: int = 0) -> Dataset:
    """Friedman's regression surface: two interacting features, one squared, two linear.

    Five further features are pure decoration — they enter the design matrix and not the
    target. A linear model cannot fit the first three terms at all, which is why lesson 5
    uses this to show what a tree buys you.

    Friedman (1991), *Multivariate Adaptive Regression Splines*, Annals of Statistics 19(1).
    """
    rng = _rng(seed)
    X = rng.random((n_samples, 10))
    y = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
        + rng.standard_normal(n_samples) * noise
    )
    return Dataset(X=X, y=y, name="friedman1")


def make_xor(n_samples: int = 400, *, noise: float = 0.35, seed: int = 0) -> Dataset:
    """Four Gaussian blobs whose label is the sign of the product of the coordinates.

    Neither feature carries any signal on its own: the correlation of each with the label is
    zero by construction. Only their interaction is informative, which is the smallest
    honest demonstration that some problems need a nonlinear model rather than a bigger one.
    """
    rng = _rng(seed)
    corners = rng.choice([-1.0, 1.0], size=(n_samples, 2))
    X = corners + rng.standard_normal((n_samples, 2)) * noise
    y = (corners[:, 0] * corners[:, 1] > 0).astype(np.float64)
    return Dataset(X=X, y=y, name="xor")


def make_noise(n_samples: int = 100, n_features: int = 2000, *, seed: int = 0) -> Dataset:
    """Independent features and labels. There is nothing to learn, and that is the point.

    The honest score of any model on fresh data is chance. Lesson 6 runs a routine and
    familiar-looking pipeline over this dataset and gets an accuracy far above chance, which
    is the clearest possible evidence that the pipeline, not the model, was wrong.
    """
    rng = _rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples).astype(np.float64)
    return Dataset(X=X, y=y, name="noise")


def train_test_split(
    X: Array,
    y: Array,
    *,
    test_size: float = 0.3,
    seed: int = 0,
    stratify: bool = False,
) -> tuple[Array, Array, Array, Array]:
    """Split into train and test. Returns ``(X_train, X_test, y_train, y_test)``.

    ``stratify`` keeps the class proportions of ``y`` in both halves. On an imbalanced
    problem an unstratified split can hand the test set a different base rate than the
    training set, and every metric downstream then measures the split as much as the model.
    """
    n_samples = X.shape[0]
    rng = _rng(seed)
    if stratify:
        test_index: list[int] = []
        for value in np.unique(y):
            members = np.flatnonzero(y == value)
            rng.shuffle(members)
            n_test = round(len(members) * test_size)
            test_index.extend(members[:n_test].tolist())
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_index] = True
    else:
        order = rng.permutation(n_samples)
        n_test = round(n_samples * test_size)
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[order[:n_test]] = True
    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]


def standardize(X_train: Array, X_test: Array) -> tuple[Array, Array]:
    """Centre and scale, with the mean and scale taken from the training half only.

    Fitting the scaler on all the data before splitting is the most common form of leakage,
    and the least dramatic-looking: the test rows influence the numbers the model is trained
    on. Lesson 6 measures how much that is worth.
    """
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (X_train - mean) / scale, (X_test - mean) / scale
