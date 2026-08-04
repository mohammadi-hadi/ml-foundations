"""Command line entry point.

Deliberately small. There is one command anybody needs — ``report`` — and its contract is
that running it twice in a row changes nothing the second time. Continuous integration runs
it once and asks git whether anything moved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_foundations import report
from ml_foundations.experiments import LESSONS, run_all


def _report(root: Path, *, write_figures: bool) -> int:
    figures_dir = root / "figures" if write_figures else None
    blocks = run_all(figures_dir)
    changed, unused = report.write(root, blocks)

    print(f"{len(blocks)} result blocks from {len(LESSONS)} lessons")
    for path in changed:
        print(f"  updated {path.relative_to(root)}")
    if not changed:
        print("  nothing changed")
    for key in unused:
        print(f"  warning: result {key!r} is computed but no document shows it")
    return 0


def _lessons() -> int:
    for lesson in LESSONS:
        print(f"{lesson.number:>2}  {lesson.title}\n    {lesson.path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml-foundations", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    report_parser = subcommands.add_parser(
        "report", help="run every experiment and write the numbers into the lessons"
    )
    report_parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository root (default: current directory)"
    )
    report_parser.add_argument(
        "--no-figures", action="store_true", help="skip plotting; tables only"
    )

    subcommands.add_parser("lessons", help="list the lessons and where they live")

    args = parser.parse_args(argv)
    if args.command == "lessons":
        return _lessons()
    return _report(args.root.resolve(), write_figures=not args.no_figures)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
