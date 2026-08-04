"""The join between the code and the prose.

Every other test checks that something computes the right answer. These check that the right
answer reaches the page — that no lesson references a result nobody produces, that no lesson
file is missing, and that running the report twice is a no-op, which is the property the
continuous integration check depends on.

This module runs every experiment, so it is also the repository's stopwatch. If the suite
starts feeling slow, it is because an experiment here got expensive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_foundations import report
from ml_foundations.cli import main
from ml_foundations.experiments import LESSONS, run_all

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def blocks() -> dict[str, str]:
    return run_all(None)


def test_every_lesson_has_a_file() -> None:
    for lesson in LESSONS:
        assert (ROOT / lesson.path).is_file(), lesson.path


def test_lessons_are_numbered_in_order_without_gaps() -> None:
    assert [lesson.number for lesson in LESSONS] == list(range(1, len(LESSONS) + 1))


def test_every_result_a_document_asks_for_is_produced(blocks: dict[str, str]) -> None:
    for path in report.documents(ROOT):
        for key in report.keys_in(path.read_text(encoding="utf-8")):
            assert key in blocks, f"{path.name} asks for {key!r}, which no experiment produces"


def test_every_result_produced_is_shown_somewhere(blocks: dict[str, str]) -> None:
    """The other direction. An experiment nobody reads is dead weight in a teaching repo."""
    shown = {
        key
        for path in report.documents(ROOT)
        for key in report.keys_in(path.read_text(encoding="utf-8"))
    }
    assert set(blocks) <= shown, f"computed but never displayed: {sorted(set(blocks) - shown)}"


def test_the_committed_numbers_are_the_computed_numbers(blocks: dict[str, str]) -> None:
    """What CI enforces with a git diff, asserted here so it fails locally first."""
    for path in report.documents(ROOT):
        text = path.read_text(encoding="utf-8")
        assert report.inject(text, blocks) == text, f"{path.name} is out of date; run `make report`"


def test_one_lesson_reruns_identically(blocks: dict[str, str]) -> None:
    """Determinism, spot-checked cheaply.

    Re-running every experiment would double the cost of the slowest test in the suite for a
    guarantee the committed-numbers test above already provides — that one compares against
    what is on disk, which a non-deterministic experiment could not match twice running.
    """
    lesson = LESSONS[0]
    assert lesson.run(None) == lesson.run(None)


def test_the_lessons_command_runs() -> None:
    assert main(["lessons"]) == 0
