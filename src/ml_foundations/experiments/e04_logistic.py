"""Lesson 4: the same model as lesson 1, and the scoring mistakes that come with it.

Nothing here is about logistic regression being hard. It is about the fact that a classifier
produces a *number*, and turning that number into a decision requires choices — a threshold, a
metric — that are usually made by default rather than on purpose. The measurements below are
all of the same model on the same features; the only thing that varies is how rare the
positive class is and how the output is scored.

The base rate is varied by moving the intercept rather than by resampling, so the two classes
keep identical feature distributions at every rate. Nothing changes except how often the
positive class occurs, which is what makes the comparison across rows fair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_foundations import datasets as ds
from ml_foundations.figures import ACCENT, ALARM, MARKERS, MUTED, SHADES, figure
from ml_foundations.logistic import LogisticRegression
from ml_foundations.metrics import (
    accuracy,
    average_precision,
    f1,
    log_loss,
    precision,
    recall,
    roc_auc,
)
from ml_foundations.report import fmt, table

BASE_RATES = (0.5, 0.2, 0.05, 0.01)
#: Half as many observations as parameters is where separation starts happening by accident.
N_SEPARABLE = 40
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def _fit(rate: float, seed: int) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    data = ds.make_logistic(
        n_samples=20000, n_features=5, positive_rate=rate, separation=1.2, seed=seed
    )
    X_train, X_test, y_train, y_test = ds.train_test_split(
        data.X, data.y, test_size=0.4, seed=seed, stratify=True
    )
    return LogisticRegression(alpha=1.0).fit(X_train, y_train), X_test, y_test


def _imbalance_table(seed: int) -> tuple[str, list[tuple[float, float, float, float, float]]]:
    rows = []
    plot_data: list[tuple[float, float, float, float, float]] = []
    for rate in BASE_RATES:
        fitted, X_test, y_test = _fit(rate, seed)
        scores = fitted.decision_function(X_test)
        predicted = fitted.predict(X_test)

        measurements = (
            accuracy(y_test, predicted),
            1.0 - float(y_test.mean()),
            recall(y_test, predicted),
            f1(y_test, predicted),
            roc_auc(y_test, scores),
            average_precision(y_test, scores),
        )
        rows.append([f"{rate:.0%}", *(fmt(value) for value in measurements)])
        plot_data.append((rate, measurements[0], measurements[3], measurements[4], measurements[5]))

    rendered = table(
        [
            "Positive rate",
            "Accuracy",
            "…of always saying no",
            "Recall",
            "F1",
            "ROC AUC",
            "Average precision",
        ],
        rows,
    )
    return rendered, plot_data


def _threshold_table(seed: int) -> str:
    """One model, one test set, seven thresholds. Nothing is refitted between rows."""
    fitted, X_test, y_test = _fit(0.05, seed)
    probability = fitted.predict_proba(X_test)

    rows = []
    for threshold in THRESHOLDS:
        predicted = (probability >= threshold).astype(np.float64)
        rows.append(
            [
                f"{threshold:g}" + (" (the default)" if threshold == 0.5 else ""),
                str(int(predicted.sum())),
                fmt(precision(y_test, predicted)),
                fmt(recall(y_test, predicted)),
                fmt(f1(y_test, predicted)),
            ]
        )

    # The threshold that maximises F1, found on the same test set. Doing that honestly needs
    # a validation split; lesson 6 is about why, and this row is deliberately the optimistic
    # version so that lesson has something concrete to point back at.
    grid = np.unique(probability)
    scores = [f1(y_test, (probability >= t).astype(np.float64)) for t in grid]
    best = float(grid[int(np.argmax(scores))])
    predicted = (probability >= best).astype(np.float64)
    rows.append(
        [
            f"{best:.3f} (best F1 here)",
            str(int(predicted.sum())),
            fmt(precision(y_test, predicted)),
            fmt(recall(y_test, predicted)),
            fmt(f1(y_test, predicted)),
        ]
    )
    return table(
        ["Threshold", "Predicted positive", "Precision", "Recall", "F1"],
        rows,
    )


def _separable_table(seed: int) -> str:
    """Maximum likelihood on data a hyperplane separates perfectly: there is no maximum.

    Forty rows and twenty features. Nothing about the classes is unusually far apart — the
    separation comes from having half as many observations as parameters, which is the way it
    almost always arises in practice. A separate test asserts that this training set really is
    linearly separable, using an independent implementation.
    """
    data = ds.make_logistic(n_samples=4040, n_features=20, separation=1.0, seed=seed)
    X_train, y_train = data.X[:N_SEPARABLE], data.y[:N_SEPARABLE]
    X_test, y_test = data.X[N_SEPARABLE:], data.y[N_SEPARABLE:]

    def row(label: str, fitted: LogisticRegression) -> list[str]:
        return [
            label,
            f"{float(np.linalg.norm(fitted.coef_)):.3g}",
            f"{log_loss(y_train, fitted.decision_function(X_train)):.1e}",
            fmt(log_loss(y_test, fitted.decision_function(X_test))),
            fmt(roc_auc(y_test, fitted.decision_function(X_test))),
        ]

    rows = [
        row(
            f"{budget} steps, no penalty",
            LogisticRegression(max_iter=budget, tol=0.0).fit(X_train, y_train),
        )
        for budget in (5, 10, 25, 50, 100)
    ]
    for alpha in (0.1, 1.0, 10.0):
        fitted = LogisticRegression(alpha=alpha, max_iter=200).fit(X_train, y_train)
        rows.append(row(f"converged, α = {alpha:g}", fitted))
    return table(
        [
            "Fit",
            "Size of the coefficients",
            "Training log loss",
            "Test log loss",
            "Test ROC AUC",
        ],
        rows,
    )


def _plot(path: Path, plot_data: list[tuple[float, float, float, float, float]]) -> None:
    with figure(path, size=(6.8, 4.2)) as ax:
        if ax is None:
            return
        rates = [row[0] for row in plot_data]
        series = (
            ("accuracy", 1, MUTED),
            ("F1", 2, ALARM),
            ("ROC AUC", 3, SHADES[3]),
            ("average precision", 4, ACCENT),
        )
        for index, (label, column, colour) in enumerate(series):
            ax.semilogx(
                rates,
                [row[column] for row in plot_data],
                marker=MARKERS[index % len(MARKERS)],
                color=colour,
                linewidth=1.8,
                label=label,
            )
        ax.set_xlabel("how often the positive class occurs")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("The same model, scored four ways, as the positives get rarer", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="center left")


def run(figures_dir: Path | None = None, *, seed: int = 4) -> dict[str, str]:
    imbalance, plot_data = _imbalance_table(seed)
    if figures_dir is not None:
        _plot(figures_dir / "04-imbalance.png", plot_data)
    return {
        "logistic-imbalance": imbalance,
        "logistic-threshold": _threshold_table(seed),
        "logistic-separable": _separable_table(seed),
    }
