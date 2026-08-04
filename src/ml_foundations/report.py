"""Write computed numbers into the prose, and refuse to let the two drift apart.

A lesson that says "regularisation helps" has taught nothing. A lesson that says "test RMSE
falls from 6.31 to 2.84" has taught something, and has also taken on a debt: the day the code
changes and the sentence does not, the lesson starts lying — confidently, in a document
someone is trying to learn from.

So no number in this repository is typed by hand. Each markdown file marks the places where
results belong::

    <!-- results: ols-vs-ridge -->
    ...anything here is overwritten...
    <!-- /results -->

``ml-foundations report`` runs every experiment, renders each block, and writes it in.
Continuous integration runs the same command and fails if the working tree changed, so a
committed number that no longer follows from committed code cannot survive a pull request.

Everything is rounded to three decimals before it is written. That is not false modesty about
precision — it is what keeps the check meaningful across machines, since the last bits of a
matrix decomposition depend on which BLAS is installed and a check that fails for that reason
would be turned off within a week.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

PLACES = 3

_BLOCK = re.compile(
    r"<!-- results: (?P<key>[a-z0-9_\-]+) -->\n(?P<body>.*?)<!-- /results -->",
    re.DOTALL,
)
_FENCE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by fenced code blocks.

    Documentation about this machinery has to be able to *show* a result marker without one
    being injected into it. The README does exactly that, and before this existed it had a
    real table written into the example explaining how tables get written in.
    """
    return [match.span() for match in _FENCE.finditer(text)]


def _outside_fences(text: str) -> list[re.Match[str]]:
    fenced = _fenced_spans(text)
    return [
        match
        for match in _BLOCK.finditer(text)
        if not any(start <= match.start() < end for start, end in fenced)
    ]


class ReportError(RuntimeError):
    """A lesson asks for a result that no experiment produces, or the reverse."""


def fmt(value: float, places: int = PLACES) -> str:
    """Format a number for a table, collapsing negative zero to zero.

    ``-0.000`` is arithmetically fine and reads as a mistake, and worse, its sign flips with
    the wind — which would make the drift check fail for no reason anyone could act on.
    """
    rendered = f"{value:.{places}f}"
    return rendered[1:] if rendered.startswith("-") and float(rendered) == 0.0 else rendered


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    align: Sequence[str] | None = None,
) -> str:
    """Render a markdown table. ``align`` takes one of ``l``, ``c``, ``r`` per column."""
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    if len(align) != len(headers):
        raise ReportError(f"{len(align)} alignments for {len(headers)} columns")
    rule = {"l": "---", "c": ":---:", "r": "---:"}
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(rule[a] for a in align) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def keys_in(text: str) -> list[str]:
    """Every result key a document asks for, in the order it asks. Code blocks do not count."""
    return [match.group("key") for match in _outside_fences(text)]


def inject(text: str, blocks: dict[str, str]) -> str:
    """Replace the body of every marked block with the rendered result of the same name."""
    # Right to left, so that replacing one block does not move the offsets of the next.
    for match in reversed(_outside_fences(text)):
        key = match.group("key")
        if key not in blocks:
            raise ReportError(f"no experiment produces the result block {key!r}")
        replacement = f"<!-- results: {key} -->\n{blocks[key].strip()}\n<!-- /results -->"
        text = text[: match.start()] + replacement + text[match.end() :]
    return text


def documents(root: Path) -> list[Path]:
    """The files results are written into: the README and every lesson, in lesson order."""
    found = [root / "README.md"]
    lessons = root / "lessons"
    if lessons.is_dir():
        found += sorted(lessons.glob("*.md"))
    return [path for path in found if path.is_file()]


def write(root: Path, blocks: dict[str, str]) -> tuple[list[Path], list[str]]:
    """Inject ``blocks`` into every document. Returns the files changed and the unused keys.

    An unused key is reported rather than raised on: a result computed but never shown is
    waste, not corruption, and during writing it is a perfectly normal intermediate state.
    The opposite case — a document asking for a result nobody computes — raises, because
    leaving it alone would silently preserve whatever stale number is sitting there.
    """
    changed: list[Path] = []
    asked: set[str] = set()
    for path in documents(root):
        before = path.read_text(encoding="utf-8")
        asked.update(keys_in(before))
        after = inject(before, blocks)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed.append(path)
    return changed, sorted(set(blocks) - asked)
