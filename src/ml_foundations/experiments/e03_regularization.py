"""Lesson 3: what a penalty buys, measured rather than drawn.

The bias-variance trade-off is usually presented as a picture of two crossing curves with no
numbers on the axes. It does not have to be. The decomposition

    expected squared error  =  bias²  +  variance  +  noise

is an identity, and on synthetic data every term on the right is separately computable: the
truth is known, so bias is measurable; the sample is repeatable, so variance is measurable;
and the noise level was chosen, so it is known exactly. This module computes all three,
checks that they add up to the fourth, and reports the value of α where the sum is smallest.

Everything else in the lesson follows from the same setup: what the penalty rescues on the
collinear data that defeated lesson 1, what the two penalties do differently when most
features are irrelevant, and why forgetting to standardise makes the whole thing meaningless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_foundations import datasets as ds
from ml_foundations.figures import ACCENT, ALARM, MUTED, SHADES, figure
from ml_foundations.metrics import rmse
from ml_foundations.regularized import Lasso, Ridge, coefficient_path
from ml_foundations.report import fmt, table

ALPHAS = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
N_REPLICATES = 200
#: Twenty rows for fifteen features. The trade-off only exists when the sample is small
#: relative to the number of parameters — with plenty of data, least squares is already the
#: answer and every penalty is a way of making it worse.
N_TRAIN = 20
N_FEATURES = 15
NOISE = 2.0
COEF_SCALE = 1.0


def _decomposition(seed: int) -> tuple[str, list[tuple[float, float, float, float]]]:
    """Bias, variance and noise for ridge at each penalty strength.

    Two hundred training sets are drawn from one fixed world. Each model is asked for its
    prediction at the same held-out points, and the spread of those predictions *is* the
    variance while their average distance from the truth *is* the bias. Nothing is estimated
    by a formula; both are sample statistics of things that were actually computed.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-COEF_SCALE, COEF_SCALE, size=N_FEATURES)
    test = ds.make_linear(
        n_samples=2000, n_features=N_FEATURES, noise=NOISE, seed=seed + 1, coef=truth
    )
    # The noiseless regression function at the test points: what a perfect model would say.
    assert test.intercept is not None
    noiseless = test.X @ truth + test.intercept
    # A fresh noise draw per replicate. Reusing one realisation would leave the measured
    # column carrying that sample's particular noise variance instead of the true one, and
    # the identity being checked would appear to fail by a couple of per cent.
    noisy_targets = noiseless + rng.standard_normal((N_REPLICATES, test.n_samples)) * NOISE

    training_sets = [
        ds.make_linear(
            n_samples=N_TRAIN, n_features=N_FEATURES, noise=NOISE, seed=seed + 100 + r, coef=truth
        )
        for r in range(N_REPLICATES)
    ]

    rows = []
    plot_data: list[tuple[float, float, float, float]] = []
    for alpha in ALPHAS:
        predictions = np.array(
            [Ridge(alpha=alpha).fit(d.X, d.y).predict(test.X) for d in training_sets]
        )
        mean_prediction = predictions.mean(axis=0)
        bias_squared = float(np.mean((mean_prediction - noiseless) ** 2))
        variance = float(np.mean(predictions.var(axis=0)))
        noise = NOISE**2
        # Measured directly against noisy targets, as a check on the three terms above.
        measured = float(np.mean((predictions - noisy_targets) ** 2))

        rows.append(
            [
                "0 (no penalty)" if alpha == 0.0 else f"{alpha:g}",
                fmt(bias_squared),
                fmt(variance),
                fmt(noise),
                fmt(bias_squared + variance + noise),
                fmt(measured),
            ]
        )
        plot_data.append((alpha, bias_squared, variance, bias_squared + variance + noise))

    rendered = table(
        [
            "α",
            "Bias²",
            "Variance",
            "Noise",
            "Sum",
            "Measured test MSE",
        ],
        rows,
    )
    return rendered, plot_data


def _collinear_table(seed: int) -> str:
    """What the penalty rescues on the data that defeated lesson 1."""
    data = ds.make_collinear(n_samples=200, n_features=6, independent_scale=1e-3, seed=seed)
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=seed)
    assert data.coef is not None

    rows = []
    for alpha in (0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0):
        fitted = Ridge(alpha=alpha).fit(X_train, y_train)
        rows.append(
            [
                "0 (least squares)" if alpha == 0.0 else f"{alpha:g}",
                f"{float(np.linalg.norm(fitted.coef_ - data.coef)):.1e}",
                f"{float(np.linalg.norm(fitted.coef_)):.1e}",
                fmt(rmse(y_test, fitted.predict(X_test))),
            ]
        )
    return table(["α", "Distance from the truth", "Size of the coefficients", "Test RMSE"], rows)


def _selection_table(seed: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Ridge and lasso on a problem where 35 of 40 features are noise."""
    data = ds.make_sparse_linear(
        n_samples=120, n_features=40, n_informative=5, noise=1.0, seed=seed
    )
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=seed)
    assert data.coef is not None
    informative = data.coef != 0.0

    rows = []
    for name, alpha in (("ridge", 10.0), ("ridge", 100.0), ("lasso", 0.1), ("lasso", 0.5)):
        model = Ridge(alpha=alpha) if name == "ridge" else Lasso(alpha=alpha, max_iter=20000)
        fitted = model.fit(X_train, y_train)
        kept = fitted.coef_ != 0.0
        rows.append(
            [
                f"`{name}` (α = {alpha:g})",
                str(int(kept.sum())),
                f"{int((kept & informative).sum())} of 5",
                str(int((kept & ~informative).sum())),
                fmt(rmse(y_test, fitted.predict(X_test))),
            ]
        )
    rendered = table(
        [
            "Model",
            "Features kept",
            "Real features found",
            "Irrelevant features kept",
            "Test RMSE",
        ],
        rows,
    )
    alphas = np.logspace(-2.2, 0.6, 60)
    return rendered, alphas, coefficient_path(Lasso, X_train, y_train, alphas), informative


def _scaling_table(seed: int) -> str:
    """One real feature, re-expressed in a smaller unit, and what the lasso then does with it.

    Nothing about the problem changes — the same measurement in centimetres rather than
    metres carries the same information. But a hundredfold smaller unit means a hundredfold
    larger coefficient, and the penalty charges for coefficient size, so the feature becomes
    a hundred times more expensive to keep. Lesson 1 proved that plain least squares is
    immune to this. Adding a penalty gives it up.
    """
    data = ds.make_sparse_linear(
        n_samples=120, n_features=40, n_informative=5, noise=1.0, seed=seed
    )
    X_train, X_test, y_train, y_test = ds.train_test_split(data.X, data.y, seed=seed)

    rows = []
    for factor in (1000.0, 1.0, 0.01):
        scaled_train, scaled_test = X_train.copy(), X_test.copy()
        scaled_train[:, 0] *= factor
        scaled_test[:, 0] *= factor
        standard_train, standard_test = ds.standardize(scaled_train, scaled_test)

        for label, (Xtr, Xte) in (
            (f"× {factor:g}", (scaled_train, scaled_test)),
            (f"× {factor:g}, standardised", (standard_train, standard_test)),
        ):
            fitted = Lasso(alpha=0.5, max_iter=20000).fit(Xtr, y_train)
            kept = fitted.coef_ != 0.0
            rows.append(
                [
                    label,
                    "kept" if kept[0] else "**dropped**",
                    f"{int(kept[:5].sum())} of 5",
                    str(int(kept.sum())),
                    fmt(rmse(y_test, fitted.predict(Xte))),
                ]
            )
    return table(
        [
            "Unit of the first feature",
            "That feature",
            "Real features found",
            "Features kept",
            "Test RMSE",
        ],
        rows,
    )


def _plot_decomposition(path: Path, plot_data: list[tuple[float, float, float, float]]) -> None:
    with figure(path, size=(6.6, 4.2)) as ax:
        if ax is None:
            return
        # α = 0 has no place on a log axis and the curve starts there, so the first point is
        # drawn at the smallest positive α and labelled in the lesson rather than the plot.
        points = [row for row in plot_data if row[0] > 0.0]
        alphas = [row[0] for row in points]
        ax.semilogx(alphas, [row[1] for row in points], marker="o", color=ALARM, label="bias²")
        ax.semilogx(alphas, [row[2] for row in points], marker="s", color=ACCENT, label="variance")
        ax.semilogx(
            alphas,
            [row[3] for row in points],
            marker="^",
            color=SHADES[0],
            linewidth=2,
            label="total expected error",
        )
        ax.axhline(NOISE**2, color=MUTED, linestyle="--", linewidth=1)
        ax.text(alphas[0], NOISE**2 * 1.08, "irreducible noise", fontsize=7, color=MUTED)
        best = min(points, key=lambda row: row[3])
        ax.axvline(best[0], color=MUTED, linewidth=1)
        ax.set_xlabel("penalty strength α")
        ax.set_ylabel("squared error")
        ax.set_title("Where the trade-off actually sits", fontsize=10)
        ax.legend(frameon=False, fontsize=8)


def _plot_path(path: Path, alphas: np.ndarray, coefficients: np.ndarray, live: np.ndarray) -> None:
    with figure(path, size=(6.6, 4.2)) as ax:
        if ax is None:
            return
        for j in range(coefficients.shape[1]):
            ax.semilogx(
                alphas,
                coefficients[:, j],
                color=ACCENT if live[j] else MUTED,
                linewidth=1.6 if live[j] else 0.8,
                zorder=3 if live[j] else 1,
            )
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xlabel("penalty strength α")
        ax.set_ylabel("coefficient")
        ax.set_title("Lasso paths: the five real features and the thirty-five decoys", fontsize=10)


def run(figures_dir: Path | None = None, *, seed: int = 3) -> dict[str, str]:
    decomposition, plot_data = _decomposition(seed)
    selection, alphas, path, live = _selection_table(seed)
    if figures_dir is not None:
        _plot_decomposition(figures_dir / "03-bias-variance.png", plot_data)
        _plot_path(figures_dir / "03-lasso-path.png", alphas, path, live)
    return {
        "ridge-bias-variance": decomposition,
        "ridge-collinear": _collinear_table(seed),
        "lasso-vs-ridge-selection": selection,
        "regularization-scaling": _scaling_table(seed),
    }
