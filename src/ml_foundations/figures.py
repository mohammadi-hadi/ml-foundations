"""Shared plotting style, and the one rule the figures follow.

Figures are committed so the lessons render on GitHub, but they are never compared against a
regenerated copy the way the numbers are: two matplotlib versions produce visually identical
PNGs that differ in thousands of bytes. The numbers are guarded, the pixels are not, and
pretending otherwise would mean a permanently red check.

One accent colour, one muted colour, one alarm colour. A plot that needs a fourth is a plot
that is trying to say two things.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ACCENT = "#1E3A5F"
MUTED = "#B0B7C3"
ALARM = "#A23B33"
SHADES = ("#12263A", "#1E3A5F", "#3C6288", "#6E96BC", "#A8C0D8")
MARKERS = ("o", "s", "^", "D", "v")


@contextmanager
def figure(path: Path, *, size: tuple[float, float] = (7.0, 4.0)) -> Iterator[Any]:
    """Open an axis, hand it to the caller, and save to ``path`` on the way out.

    Yields ``None`` instead of an axis when matplotlib is not installed, so that a machine
    with only the core dependency can still run every experiment and regenerate every table.
    Callers guard on it; the tables are the part that has to work everywhere.
    """
    try:
        import matplotlib
    except ImportError:
        yield None
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size)
    try:
        yield ax
        ax.spines[["top", "right"]].set_visible(False)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)
