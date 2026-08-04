"""Lesson 5: what a tree buys, what it costs, and which ensemble fixes which problem.

The interesting measurement here is not "the forest scored better". It is *which term of the
decomposition moved*. Lesson 3 established that expected error splits into bias, variance and
noise, and that the three are separately measurable on data you generated. Applying the same
machinery to ensembles turns a pair of folk explanations — bagging reduces variance, boosting
reduces bias — into two columns that can be read off a table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_foundations import datasets as ds
from ml_foundations.ensembles import BaggedTrees, GradientBoostedTrees
from ml_foundations.figures import ACCENT, ALARM, MARKERS, MUTED, SHADES, figure
from ml_foundations.metrics import rmse
from ml_foundations.regularized import Ridge
from ml_foundations.report import fmt, table
from ml_foundations.trees import DecisionTree

DEPTHS = (1, 2, 3, 5, 8, 12, 20)
#: Replicates for the bias-variance measurement. Twenty fits of four models is the most
#: expensive thing in this repository; the effects being measured are large enough that more
#: replicates would sharpen the third decimal and change no conclusion.
N_REPLICATES = 20
N_TRAIN = 250
#: Leaves of five rather than one throughout the decomposition. Applied to every model
#: equally, so the comparison is unaffected, and it keeps a forest of deep trees affordable.
MIN_LEAF = 5
NOISE = 1.0
ROUNDS = (1, 5, 10, 25, 50, 100, 200, 400)


def _split(seed: int, n_samples: int = 600) -> tuple[np.ndarray, ...]:
    data = ds.make_friedman1(n_samples=n_samples, noise=NOISE, seed=seed)
    return ds.train_test_split(data.X, data.y, test_size=0.4, seed=seed)


def _depth_table(seed: int) -> tuple[str, list[tuple[int, float, float]]]:
    X_train, X_test, y_train, y_test = _split(seed)
    rows = []
    plot_data: list[tuple[int, float, float]] = []
    for depth in DEPTHS:
        fitted = DecisionTree(max_depth=depth).fit(X_train, y_train)
        train_error = rmse(y_train, fitted.predict(X_train))
        test_error = rmse(y_test, fitted.predict(X_test))
        rows.append(
            [
                str(depth),
                str(fitted.n_leaves),
                fmt(train_error),
                fmt(test_error),
                fmt(test_error - train_error),
            ]
        )
        plot_data.append((depth, train_error, test_error))
    return (
        table(["Max depth", "Leaves", "Training RMSE", "Test RMSE", "Gap"], rows),
        plot_data,
    )


def _decomposition_table(seed: int) -> str:
    """Bias and variance for four models, measured exactly as in lesson 3.

    Twenty training sets are drawn from the same world. The spread of the predictions across
    those twenty fits is the variance; how far their average sits from the noiseless truth is
    the bias. Fitting twenty replicates of four models is the most expensive thing in this
    repository, and it is the only way to get these two columns honestly.
    """
    test = ds.make_friedman1(n_samples=1000, noise=NOISE, seed=seed + 1)
    rng = np.random.default_rng(seed)
    # The Friedman surface, evaluated without noise: the best any model could do.
    noiseless = (
        10.0 * np.sin(np.pi * test.X[:, 0] * test.X[:, 1])
        + 20.0 * (test.X[:, 2] - 0.5) ** 2
        + 10.0 * test.X[:, 3]
        + 5.0 * test.X[:, 4]
    )
    targets = noiseless + rng.standard_normal((N_REPLICATES, test.n_samples)) * NOISE
    training_sets = [
        ds.make_friedman1(n_samples=N_TRAIN, noise=NOISE, seed=seed + 100 + r)
        for r in range(N_REPLICATES)
    ]

    models = (
        ("one deep tree", lambda s: DecisionTree(max_depth=12, min_samples_leaf=MIN_LEAF, seed=s)),
        (
            "bagging, 25 trees",
            lambda s: BaggedTrees(n_estimators=25, max_depth=12, min_samples_leaf=MIN_LEAF, seed=s),
        ),
        (
            "random forest, 25 trees",
            lambda s: BaggedTrees(
                n_estimators=25, max_depth=12, min_samples_leaf=MIN_LEAF, max_features=3, seed=s
            ),
        ),
        (
            "boosting, 100 rounds",
            lambda s: GradientBoostedTrees(
                n_estimators=100, learning_rate=0.1, max_depth=3, seed=s
            ),
        ),
    )

    rows = []
    for name, build in models:
        predictions = np.array(
            [build(r).fit(d.X, d.y).predict(test.X) for r, d in enumerate(training_sets)]
        )
        bias_squared = float(np.mean((predictions.mean(axis=0) - noiseless) ** 2))
        variance = float(np.mean(predictions.var(axis=0)))
        measured = float(np.mean((predictions - targets) ** 2))
        rows.append(
            [
                name,
                fmt(bias_squared),
                fmt(variance),
                fmt(bias_squared + variance + NOISE**2),
                fmt(measured),
            ]
        )
    return table(["Model", "Bias²", "Variance", "Sum", "Measured test MSE"], rows)


def _boosting_table(seed: int) -> tuple[str, dict[float, tuple[np.ndarray, np.ndarray]]]:
    X_train, X_test, y_train, y_test = _split(seed, n_samples=400)
    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for rate in (0.5, 0.1):
        fitted = GradientBoostedTrees(
            n_estimators=max(ROUNDS), learning_rate=rate, max_depth=3, seed=seed
        ).fit(X_train, y_train)
        train_curve = np.array([rmse(y_train, s) for s in fitted.staged_predict(X_train)])
        test_curve = np.array([rmse(y_test, s) for s in fitted.staged_predict(X_test)])
        curves[rate] = (train_curve, test_curve)
        for rounds in ROUNDS:
            rows.append(
                [
                    f"{rate:g}",
                    str(rounds),
                    fmt(float(train_curve[rounds])),
                    fmt(float(test_curve[rounds])),
                ]
            )
        best = int(np.argmin(test_curve))
        rows.append(
            [
                f"{rate:g}",
                f"**{best}** (best)",
                fmt(float(train_curve[best])),
                fmt(float(test_curve[best])),
            ]
        )
    return table(["Learning rate", "Rounds", "Training RMSE", "Test RMSE"], rows), curves


def _model_choice_table(seed: int) -> str:
    """The same four models on a linear surface and on a nonlinear one."""
    linear = ds.make_linear(n_samples=600, n_features=10, noise=1.0, seed=seed)
    friedman = ds.make_friedman1(n_samples=600, noise=1.0, seed=seed)

    rows = []
    for name, data in (("linear", linear), ("Friedman", friedman)):
        X_train, X_test, y_train, y_test = ds.train_test_split(
            data.X, data.y, test_size=0.4, seed=seed
        )
        scores = []
        for build in (
            lambda: Ridge(alpha=1.0),
            lambda: DecisionTree(max_depth=8),
            lambda: BaggedTrees(n_estimators=40, max_depth=12, max_features=3, seed=seed),
            lambda: GradientBoostedTrees(n_estimators=200, learning_rate=0.1, seed=seed),
        ):
            fitted = build().fit(X_train, y_train)
            scores.append(fmt(rmse(y_test, fitted.predict(X_test))))
        rows.append([f"`{name}`", *scores])
    return table(
        ["Data", "Ridge", "One tree", "Random forest", "Boosting"],
        rows,
    )


def _plot_depth(path: Path, plot_data: list[tuple[int, float, float]]) -> None:
    with figure(path, size=(6.6, 4.0)) as ax:
        if ax is None:
            return
        depths = [row[0] for row in plot_data]
        ax.plot(depths, [row[1] for row in plot_data], marker="o", color=MUTED, label="training")
        ax.plot(depths, [row[2] for row in plot_data], marker="s", color=ACCENT, label="test")
        ax.axhline(NOISE, color=ALARM, linestyle="--", linewidth=1)
        ax.text(depths[0], NOISE * 1.05, "noise floor", fontsize=7, color=ALARM)
        ax.set_xlabel("maximum depth")
        ax.set_ylabel("RMSE")
        ax.set_title("One tree: the gap is the memorisation", fontsize=10)
        ax.legend(frameon=False, fontsize=8)


def _plot_boosting(path: Path, curves: dict[float, tuple[np.ndarray, np.ndarray]]) -> None:
    with figure(path, size=(6.6, 4.0)) as ax:
        if ax is None:
            return
        for index, (rate, (train_curve, test_curve)) in enumerate(curves.items()):
            # Round zero has no place on a log axis, and the early rounds are where the
            # interesting thing happens: on a linear axis the minimum at round 14 is a pixel.
            rounds = np.arange(1, len(train_curve))
            train_curve, test_curve = train_curve[1:], test_curve[1:]
            ax.plot(
                rounds,
                train_curve,
                color=SHADES[index * 2],
                linestyle="--",
                linewidth=1.2,
                label=f"training, rate {rate:g}",
            )
            ax.plot(
                rounds,
                test_curve,
                color=SHADES[index * 2],
                marker=MARKERS[index],
                markevery=0.12,
                markersize=4,
                linewidth=1.8,
                label=f"test, rate {rate:g}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("boosting rounds")
        ax.set_ylabel("RMSE")
        ax.set_title("Boosting does not converge — it keeps fitting", fontsize=10)
        ax.legend(frameon=False, fontsize=8)


def run(figures_dir: Path | None = None, *, seed: int = 5) -> dict[str, str]:
    depth, depth_plot = _depth_table(seed)
    boosting, curves = _boosting_table(seed)
    if figures_dir is not None:
        _plot_depth(figures_dir / "05-tree-depth.png", depth_plot)
        _plot_boosting(figures_dir / "05-boosting.png", curves)
    return {
        "tree-depth": depth,
        "ensemble-decomposition": _decomposition_table(seed),
        "boosting-rounds": boosting,
        "model-choice": _model_choice_table(seed),
    }
