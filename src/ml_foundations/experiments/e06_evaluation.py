"""Lesson 6: whether any of the numbers in the previous five lessons meant anything.

The demonstrations here are built on :func:`~ml_foundations.datasets.make_noise`, where the
features and the labels are drawn independently. Nothing can be learned from it. The honest
score of any model is chance, and that is not an estimate — it is a fact about how the data was
generated. So every number above chance in this lesson is a measurement of a mistake, and the
size of the mistake is exactly the size of the excess.

That is the only setting in which leakage can be measured rather than argued about. On real
data a suspiciously good score is a suspicion; here it is a quantity.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml_foundations import datasets as ds
from ml_foundations.evaluation import (
    Fold,
    cross_val_scores,
    expected_calibration_error,
    k_fold,
    nested_scores,
    reliability_curve,
    repeated_splits,
    select_k_best,
)
from ml_foundations.figures import ACCENT, ALARM, MARKERS, MUTED, SHADES, figure
from ml_foundations.logistic import LogisticRegression
from ml_foundations.metrics import accuracy, r2, roc_auc
from ml_foundations.regularized import Ridge
from ml_foundations.report import fmt, table
from ml_foundations.trees import DecisionTree

Array = NDArray[np.float64]
FitPredict = Callable[[Array, Array, Array], Array]

N_NOISE_SAMPLES = 100
N_NOISE_FEATURES = 2000
N_SELECTED = 20
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
#: Independent datasets to average the leakage measurements over. One run of a hundred-row
#: dataset is noisy enough that the honest pipeline lands near 0.55 by chance alone.
N_DATASETS = 25
#: Candidate models to choose between when demonstrating selection optimism.
N_CANDIDATES = 60
#: Replicates for that measurement. One run would be a single draw of the noise in question.
N_SELECTION_RUNS = 10


def _leakage_table(seed: int) -> str:
    """Three pipelines on data with no signal in it. One of them finds some anyway.

    Averaged over :data:`N_DATASETS` independently generated datasets. A single run of this
    would put the honest pipeline somewhere near 0.55 rather than near 0.50 — a hundred rows
    is not many, and the estimate has real variance, which is itself the subject of the next
    section. Averaging removes that noise so the excess that remains is the leak.
    """
    scores: dict[str, list[tuple[float, float]]] = {}
    for replicate in range(N_DATASETS):
        for label, (hit, auc) in _one_leakage_run(seed + replicate).items():
            scores.setdefault(label, []).append((hit, auc))
    return table(
        ["Pipeline", "Cross-validated accuracy", "…ROC AUC", "Excess over chance"],
        [
            [
                label,
                fmt(float(np.mean([h for h, _ in runs]))),
                fmt(float(np.mean([a for _, a in runs]))),
                fmt(float(np.mean([a for _, a in runs])) - 0.5),
            ]
            for label, runs in scores.items()
        ],
    )


def _as_labels(fit_predict: FitPredict) -> FitPredict:
    """Turn a scoring pipeline into a hard-label one, so both metrics use the same fit."""

    def labelled(X_train: Array, y_train: Array, X_test: Array) -> Array:
        return (fit_predict(X_train, y_train, X_test) > 0).astype(np.float64)

    return labelled


def _subset_pipeline(subset: NDArray[np.intp]) -> FitPredict:
    """A pipeline that only ever sees ``subset`` of the columns."""

    def fit_predict(X_train: Array, y_train: Array, X_test: Array) -> Array:
        model = LogisticRegression(alpha=1.0).fit(X_train[:, subset], y_train)
        return model.decision_function(X_test[:, subset])

    return fit_predict


def _one_leakage_run(seed: int) -> dict[str, tuple[float, float]]:
    data = ds.make_noise(n_samples=N_NOISE_SAMPLES, n_features=N_NOISE_FEATURES, seed=seed)
    folds = k_fold(data.n_samples, n_splits=5, seed=seed, y=data.y)

    # The mistake: the features are chosen once, looking at every label in the dataset.
    # By the time cross-validation runs, each held-out fold has already influenced which
    # columns the model is allowed to see.
    leaked = select_k_best(data.X, data.y, k=N_SELECTED)

    def leaky_selection(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        model = LogisticRegression(alpha=1.0).fit(X_train[:, leaked], y_train)
        return model.decision_function(X_test[:, leaked])

    def honest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        chosen = select_k_best(X_train, y_train, k=N_SELECTED)
        model = LogisticRegression(alpha=1.0).fit(X_train[:, chosen], y_train)
        return model.decision_function(X_test[:, chosen])

    def no_selection(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        model = LogisticRegression(alpha=1.0).fit(X_train[:, :N_SELECTED], y_train)
        return model.decision_function(X_test[:, :N_SELECTED])

    out = {}
    for label, fit_predict in (
        ("select the best 20 features on all the data, then cross-validate", leaky_selection),
        ("select the best 20 features inside each fold", honest),
        ("no selection at all — take the first 20 features", no_selection),
    ):
        auc = cross_val_scores(fit_predict, data.X, data.y, roc_auc, folds)
        hit = cross_val_scores(_as_labels(fit_predict), data.X, data.y, accuracy, folds)
        out[label] = (float(np.mean(hit)), float(np.mean(auc)))
    return out


def _scaler_table(seed: int) -> str:
    """The quieter version of the same mistake, and an honest report of how much it is worth."""
    runs: dict[str, list[float]] = {}
    for replicate in range(N_DATASETS):
        for label, auc in _one_scaler_run(seed + 500 + replicate).items():
            runs.setdefault(label, []).append(auc)
    return table(
        ["Pipeline", "Cross-validated ROC AUC", "Excess over chance"],
        [
            [label, fmt(float(np.mean(values))), fmt(float(np.mean(values)) - 0.5)]
            for label, values in runs.items()
        ],
    )


def _one_scaler_run(seed: int) -> dict[str, float]:
    data = ds.make_noise(n_samples=N_NOISE_SAMPLES, n_features=200, seed=seed + 1)
    folds = k_fold(data.n_samples, n_splits=5, seed=seed, y=data.y)
    everything_scaled = (data.X - data.X.mean(axis=0)) / np.where(
        data.X.std(axis=0) == 0.0, 1.0, data.X.std(axis=0)
    )

    def leaky(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        return LogisticRegression(alpha=1.0).fit(X_train, y_train).decision_function(X_test)

    def honest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        train_scaled, test_scaled = ds.standardize(X_train, X_test)
        return (
            LogisticRegression(alpha=1.0).fit(train_scaled, y_train).decision_function(test_scaled)
        )

    return {
        label: float(np.mean(cross_val_scores(fit_predict, X, data.y, roc_auc, folds)))
        for label, X, fit_predict in (
            ("scaler fitted on everything, before splitting", everything_scaled, leaky),
            ("scaler fitted inside each fold", data.X, honest),
        )
    }


def _split_variance_table(seed: int) -> tuple[str, list[float]]:
    """How much is one train/test split worth as evidence?"""
    data = ds.make_friedman1(n_samples=300, noise=1.0, seed=seed)

    single = [
        r2(data.y[test], Ridge(alpha=1.0).fit(data.X[train], data.y[train]).predict(data.X[test]))
        for train, test in repeated_splits(data.n_samples, test_size=0.3, repeats=200, seed=seed)
    ]
    folded = cross_val_scores(
        lambda X_train, y_train, X_test: Ridge(alpha=1.0).fit(X_train, y_train).predict(X_test),
        data.X,
        data.y,
        r2,
        k_fold(data.n_samples, n_splits=5, seed=seed),
    )
    rows = [
        [
            "one 70/30 split, repeated 200 times",
            fmt(float(np.mean(single))),
            fmt(float(np.std(single))),
            fmt(float(np.min(single))),
            fmt(float(np.max(single))),
        ],
        [
            "5-fold cross-validation, the five folds",
            fmt(float(np.mean(folded))),
            fmt(float(np.std(folded))),
            fmt(float(np.min(folded))),
            fmt(float(np.max(folded))),
        ],
    ]
    return table(["Estimate", "Mean R²", "Spread", "Worst", "Best"], rows), single


def _selection_table(seed: int) -> str:
    """Trying sixty models and reporting the best one's score, on data with nothing in it.

    Each candidate is a random handful of five features. None is better than any other,
    because none is better than nothing — but sixty noisy estimates of 0.5 have a maximum, and
    the maximum is what gets written down. The gap between the winner's score on the set that
    crowned it and the same model's score on data it never influenced is the whole effect.

    Averaged over :data:`N_SELECTION_RUNS` independent datasets, because a single run of this
    would be measuring one draw of exactly the noise under discussion.
    """
    columns: dict[str, list[float]] = {}
    for replicate in range(N_SELECTION_RUNS):
        for label, value in _one_selection_run(seed + 900 + replicate).items():
            columns.setdefault(label, []).append(value)
    return table(
        ["Estimate", "ROC AUC"],
        [[label, fmt(float(np.mean(values)))] for label, values in columns.items()],
    )


def _one_selection_run(seed: int) -> dict[str, float]:
    data = ds.make_noise(n_samples=1500, n_features=200, seed=seed)
    rng = np.random.default_rng(seed)
    X_train, y_train = data.X[:300], data.y[:300]
    X_pick, y_pick = data.X[300:400], data.y[300:400]
    X_fresh, y_fresh = data.X[400:], data.y[400:]

    subsets = [rng.choice(data.n_features, size=5, replace=False) for _ in range(N_CANDIDATES)]
    fits = [LogisticRegression(alpha=1.0).fit(X_train[:, s], y_train) for s in subsets]
    picked = [
        roc_auc(y_pick, f.decision_function(X_pick[:, s]))
        for f, s in zip(fits, subsets, strict=True)
    ]
    winner = int(np.argmax(picked))

    def choose_then_fit(X_inner: Array, y_inner: Array, X_test: Array, folds: list[Fold]) -> Array:
        """Pick the winner using only the inner folds, then apply it to the outer test rows."""
        inner = [
            float(np.mean(cross_val_scores(_subset_pipeline(s), X_inner, y_inner, roc_auc, folds)))
            for s in subsets
        ]
        return _subset_pipeline(subsets[int(np.argmax(inner))])(X_inner, y_inner, X_test)

    nested = nested_scores(
        choose_then_fit, data.X[:700], data.y[:700], roc_auc, n_outer=3, n_inner=3, seed=seed
    )
    return {
        f"average of all {N_CANDIDATES} candidates, on the set used to choose": float(
            np.mean(picked)
        ),
        f"**best** of {N_CANDIDATES}, on the set used to choose": float(max(picked)),
        "that same winner, on data it did not choose on": roc_auc(
            y_fresh, fits[winner].decision_function(X_fresh[:, subsets[winner]])
        ),
        "nested cross-validation over the whole procedure": float(np.mean(nested)),
    }


def _calibration_table(seed: int) -> tuple[str, dict[str, list[tuple[float, float, int]]]]:
    """Three models that rank comparably well and disagree about how sure they are."""
    data = ds.make_logistic(n_samples=4040, n_features=20, separation=1.0, seed=seed)
    X_train, y_train = data.X[:40], data.y[:40]
    X_test, y_test = data.X[40:], data.y[40:]

    models: dict[str, LogisticRegression | DecisionTree] = {
        "unpenalised, separable": LogisticRegression(max_iter=50, tol=0.0),
        "penalised (α = 1)": LogisticRegression(alpha=1.0, max_iter=200),
        "deep tree": DecisionTree(criterion="gini", max_depth=8),
    }
    rows = []
    curves = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)
        rows.append(
            [
                name,
                fmt(roc_auc(y_test, probability)),
                fmt(expected_calibration_error(y_test, probability)),
                fmt(float(np.mean(np.maximum(probability, 1.0 - probability)))),
            ]
        )
        curves[name] = reliability_curve(y_test, probability)
    return (
        table(
            ["Model", "ROC AUC", "Calibration error", "Average confidence"],
            rows,
        ),
        curves,
    )


def _plot_splits(path: Path, single: list[float]) -> None:
    with figure(path, size=(6.6, 3.6)) as ax:
        if ax is None:
            return
        ax.hist(single, bins=30, color=ACCENT, edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(single)), color=ALARM, linewidth=1.5)
        ax.set_xlabel("test R² from a single 70/30 split")
        ax.set_ylabel("how often")
        ax.set_title("Two hundred splits of the same data, same model", fontsize=10)


def _plot_calibration(path: Path, curves: dict[str, list[tuple[float, float, int]]]) -> None:
    with figure(path, size=(5.4, 4.6)) as ax:
        if ax is None:
            return
        ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
        for index, (name, curve) in enumerate(curves.items()):
            ax.plot(
                [point[0] for point in curve],
                [point[1] for point in curve],
                marker=MARKERS[index % len(MARKERS)],
                markersize=5,
                linewidth=1.5,
                color=SHADES[index * 2 % len(SHADES)],
                label=name,
            )
        ax.set_xlabel("probability the model claimed")
        ax.set_ylabel("how often it was right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Calibration: the dashed line is honesty", fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="upper left")


def run(figures_dir: Path | None = None, *, seed: int = 6) -> dict[str, str]:
    split_variance, single = _split_variance_table(seed)
    calibration, curves = _calibration_table(seed)
    if figures_dir is not None:
        _plot_splits(figures_dir / "06-split-variance.png", single)
        _plot_calibration(figures_dir / "06-calibration.png", curves)
    return {
        "leakage-selection": _leakage_table(seed),
        "leakage-scaler": _scaler_table(seed),
        "split-variance": split_variance,
        "selection-optimism": _selection_table(seed),
        "calibration": calibration,
    }
