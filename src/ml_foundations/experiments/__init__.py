"""One module per lesson, each producing the numbers that lesson quotes.

An experiment is a function of a seed and nothing else. It reads no files, takes no
arguments a caller could get wrong, and returns a dictionary from result-block name to
rendered markdown. That shape is what lets ``ml-foundations report`` regenerate the whole
repository in one pass, and what lets a reader re-derive any figure in the prose by running
one function and reading its source.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml_foundations.experiments import e01_linear, e02_gradient_descent, e03_regularization
from ml_foundations.report import ReportError


@dataclass(frozen=True)
class Lesson:
    number: int
    slug: str
    title: str
    run: Callable[[Path | None], dict[str, str]]

    @property
    def path(self) -> str:
        return f"lessons/{self.number:02d}-{self.slug}.md"


LESSONS: tuple[Lesson, ...] = (
    Lesson(1, "linear-regression", "Linear regression, solved three ways", e01_linear.run),
    Lesson(2, "gradient-descent", "Gradient descent and its friends", e02_gradient_descent.run),
    Lesson(
        3,
        "regularization",
        "Regularisation and the bias-variance trade-off",
        e03_regularization.run,
    ),
)


def run_all(figures_dir: Path | None = None) -> dict[str, str]:
    """Run every lesson's experiment and merge the result blocks.

    Duplicate keys are refused rather than merged. Two lessons quietly writing to the same
    block would mean one of them silently wins, and the losing lesson would display numbers
    computed for a different question.
    """
    blocks: dict[str, str] = {}
    for lesson in LESSONS:
        for key, rendered in lesson.run(figures_dir).items():
            if key in blocks:
                raise ReportError(f"two lessons both produce the result block {key!r}")
            blocks[key] = rendered
    return blocks
