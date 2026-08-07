# ML Foundations

**Machine learning fundamentals implemented from scratch in numpy — where every implementation
is checked against an independent reference, and every number in every lesson is regenerated
from the code by continuous integration.**

[![CI](https://github.com/mohammadi-hadi/ml-foundations/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadi-hadi/ml-foundations/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21797931.svg)](https://doi.org/10.5281/zenodo.21797931)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

There are a great many "ML from scratch" repositories. Most of them share two problems: the
implementations are never checked against anything, so subtle errors survive indefinitely, and
the claims in the prose — *regularisation helps*, *bagging reduces variance* — are assertions
rather than measurements.

This one is built the other way round. Nothing in these lessons is asserted. Every table below
was produced by a script in this repository, `make report` rewrites all of them from the code,
and CI fails the build if a committed number no longer follows from committed code. Every
estimator is pinned by a test at one of the tiers described below.

I am a machine learning scientist — PhD in NLP, now building production LLM and ranking
systems — and this is the material I use when I teach fundamentals. It is a companion to
[ml-learning-paths](https://github.com/mohammadi-hadi/ml-learning-paths), which says what to
study; this says what to build while you study it.

## The lessons

| # | Lesson | The measurement it turns on |
|---|---|---|
| 1 | [Linear regression, solved three ways](lessons/01-linear-regression.md) | Normal equations lose two digits per decade of conditioning; QR loses none; the predictions never notice |
| 2 | [Gradient descent and its friends](lessons/02-gradient-descent.md) | The stable step size is exactly `2/L`, the fastest is `2/(L+m)`, and mini-batch descent with a fixed step never converges |
| 3 | [Regularisation and the bias-variance trade-off](lessons/03-regularization.md) | The decomposition is an identity — all three terms measured, and they add up |
| 4 | [Logistic regression, and metrics that lie](lessons/04-logistic-regression.md) | 99% accurate and 0.000 recall on the same model; F1 improves 15× by changing the threshold |
| 5 | [Trees, and why one is never enough](lessons/05-trees-and-ensembles.md) | Bagging cuts variance 5× and leaves bias alone; boosting halves the bias |
| 6 | [Evaluation](lessons/06-evaluation.md) | An ROC AUC of 0.923 on data that contains nothing at all |

If you read one, read **[lesson 6](lessons/06-evaluation.md)**. It is the discipline most
people add last and need first, and it is the only one where the mistakes cost you the whole
result rather than a few per cent.

## The headline

Lesson 6 runs three pipelines over a dataset whose features and labels are drawn
independently. Nothing can be learned from it — the honest score is chance, by construction —
so every point above 0.5 is a measurement of a mistake rather than an estimate of anything:

<!-- results: leakage-selection -->
| Pipeline | Cross-validated accuracy | …ROC AUC | Excess over chance |
|---|---:|---:|---:|
| select the best 20 features on all the data, then cross-validate | 0.848 | 0.923 | 0.423 |
| select the best 20 features inside each fold | 0.504 | 0.497 | -0.003 |
| no selection at all — take the first 20 features | 0.498 | 0.479 | -0.021 |
<!-- /results -->

The difference between the first two rows is where one line of code sits. Everything else about
them is identical.

## Run it

```bash
git clone https://github.com/mohammadi-hadi/ml-foundations
cd ml-foundations
make setup          # a virtualenv and the dev dependencies
make test           # the full suite, about a minute, no network
make report         # re-run every experiment and rewrite the numbers in the lessons
make check          # test + lint + fail if any committed number moved
```

Everything runs on a laptop CPU in under two minutes. No downloads, no GPU, no API keys, no
datasets to fetch — every dataset is generated from a seed by
[`datasets.py`](src/ml_foundations/datasets.py).

## How the implementations are verified

"Tested against scikit-learn" is not true of everything here, and claiming it would be the
overclaim worth catching. Objectives differ, ties break differently, and some of these
estimators have no equivalent to compare with. So verification is tiered, and each test says
which tier it is at:

| Tier | Method | Where it applies |
|---|---|---|
| **A** | Exact agreement with an independent implementation | Least squares, ridge, lasso, logistic regression, trees to depth 4-5, every metric, the cross-validation aggregation |
| **B** | Properties that must hold exactly, checked without any reference | Residual orthogonality, first-order conditions, scale equivariance, split optimality by brute force, Adam's bias correction |
| **C** | Recovery of parameters known before the data existed | Every generator's truth, on samples large enough for the estimate to be tight |

Two places where tier A is genuinely unavailable are worth naming, because they are where a
careless test would quietly pass:

- **Scaling conventions differ between estimators.** scikit-learn's ridge minimises
  `‖y-Xw‖² + α‖w‖²`; its lasso minimises `‖y-Xw‖²/2n + α‖w‖₁`; its logistic regression
  regularises by default and is parameterised by `C = 1/α`. An implementation that mixes these
  up agrees at exactly α = 0 and disagrees everywhere else, so every one of them is checked at
  several penalty strengths rather than one.
- **Deep trees have ties.** Past depth four, several candidate splits achieve *identical*
  impurity and which one wins depends on iteration order. Demanding equality there would be
  testing a coincidence, so the test asserts the inequality that is meaningful: this
  implementation's partition is never worse by the objective both are optimising. (At depth
  five the two are equal; at depth six this one is better.)

A third category is the one that keeps the lessons honest: several tests assert **the claim a
lesson is built on**, so that a change which quietly turns a demonstration into an illustration
of nothing fails the suite rather than the reader.

## How the numbers stay true

Each lesson marks where its results belong:

```markdown
<!-- results: ols-conditioning -->
...anything here is overwritten...
<!-- /results -->
```

`make report` runs every experiment and writes the tables in. CI runs the same command and
fails if the working tree changed, so a committed number that no longer follows from committed
code cannot survive a pull request. The check is enforced on one operating system and one
Python version with pinned `numpy`, `scipy` and `scikit-learn` — the last digits of a matrix
factorisation depend on which BLAS is installed, and a check that failed for that reason would
be switched off within a week.

Everything is rounded to three decimals for the same reason. Where a quantity is *inherently*
rounding error — how far two solvers of the same problem drift apart — it is reported as a count
of surviving digits rather than as a mantissa, because the count is a property of the problem
and the mantissa is a property of the machine.

Figures are committed so the lessons render, and are never diffed: two matplotlib versions
produce visually identical PNGs that differ in thousands of bytes. **The numbers are guarded,
the pixels are not**, and that is the right split.

## Dependencies

The implementations depend on **numpy and nothing else**. scikit-learn appears only as the
thing they are checked against, and in the experiments that need a reference model. The command
line is stdlib `argparse`. matplotlib is optional — without it every table still regenerates,
and only the figures are skipped.

## Layout

```
src/ml_foundations/
  datasets.py      seeded generators whose true parameters are known
  functions.py     numerically stable sigmoid and log-sigmoid
  linear.py        least squares: normal equations, QR, SVD
  optim.py         SGD, momentum, Adam, and the drivers that record what they did
  regularized.py   ridge in closed form, lasso by coordinate descent
  logistic.py      logistic regression by iteratively reweighted least squares
  trees.py         CART, one split search serving both criteria
  ensembles.py     bagging, random forests, gradient boosting
  metrics.py       every score, written out from its definition
  evaluation.py    folds, nested cross-validation, calibration
  report.py        writes computed results into the lessons
  experiments/     one module per lesson, producing that lesson's numbers
lessons/           the prose, with the results injected
figures/           committed so the lessons render on GitHub
tests/             the verification described above
```

## Limitations

- **The datasets are synthetic.** That is what makes the truth knowable and the claims
  checkable, and it is also the limit: nothing here involves missing values, categorical
  encodings, temporal structure, or the parts of real data that consume most of the work.
- **The implementations are written to be read, not used.** They are correct and they are slow.
  Use scikit-learn.
- **Nothing here covers deep learning, sequence models, or anything a GPU would help with.**
  Different subject; the fundamentals come first.
- **Lesson 6's leakage results are measured on one shape of problem** — many features, few
  rows. Leakage severity depends on the setting, and the scaler result is a reminder that not
  every kind of leakage is worth the same.

## References

The material is standard; the framing and the measurements are mine. Where a specific method
has an origin worth citing it is cited in the docstring of the code that implements it —
Polyak (1964) for momentum, Friedman (1991) for the regression surface, Friedman (2001) for
gradient boosting, Friedman, Hastie and Tibshirani (2010) for coordinate descent, Kingma and Ba
(2015) for Adam.

For the underlying theory: Hastie, Tibshirani and Friedman, *The Elements of Statistical
Learning*; Bishop, *Pattern Recognition and Machine Learning*; Trefethen and Bau, *Numerical
Linear Algebra* for lesson 1.

## Related

- [ml-learning-paths](https://github.com/mohammadi-hadi/ml-learning-paths) — which courses to
  take, in what order, for which role
- [modern-ai-engineering](https://github.com/mohammadi-hadi/modern-ai-engineering) — the same
  approach applied to production LLM systems
- [trajectory-judge](https://github.com/mohammadi-hadi/trajectory-judge) — what an LLM judge
  misses when an agent reaches the right answer the wrong way

## Contributing

Corrections are welcome, particularly to the mathematics or to a claim that does not hold up.
A pull request that changes a result must run `make report` so the lessons and the code stay in
step — CI will insist on it.

MIT licensed. Written by [Hadi Mohammadi](https://mohammadi.cv).
