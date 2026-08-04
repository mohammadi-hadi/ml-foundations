"""The injection machinery is load-bearing: if it silently no-ops, every lesson goes stale.

These tests care much less about how a table looks than about the two failure modes that
would quietly break the repository's central promise — a block that is left untouched, and a
number that changes shape depending on the machine.
"""

from __future__ import annotations

import pytest

from ml_foundations import report


def test_a_block_is_replaced_in_place() -> None:
    text = "before\n\n<!-- results: demo -->\nstale rubbish\n<!-- /results -->\n\nafter\n"
    out = report.inject(text, {"demo": "| a |\n|---|\n| 1 |\n"})
    assert out.startswith("before\n")
    assert out.endswith("after\n")
    assert "stale rubbish" not in out
    assert "| 1 |" in out


def test_injection_is_idempotent() -> None:
    """Running the report twice must not change the file the second time, or CI is a lottery."""
    text = "<!-- results: demo -->\n\n<!-- /results -->\n"
    once = report.inject(text, {"demo": "x"})
    assert report.inject(once, {"demo": "x"}) == once


def test_every_block_in_a_document_is_replaced() -> None:
    text = "<!-- results: a -->\nx\n<!-- /results -->\n<!-- results: b -->\ny\n<!-- /results -->\n"
    out = report.inject(text, {"a": "A", "b": "B"})
    assert "A" in out
    assert "B" in out
    assert "x" not in out
    assert "y" not in out


def test_a_document_asking_for_a_result_nobody_computes_is_an_error() -> None:
    """The dangerous direction. Skipping it would leave a stale number looking authoritative."""
    text = "<!-- results: missing -->\n0.42\n<!-- /results -->\n"
    with pytest.raises(report.ReportError, match="missing"):
        report.inject(text, {})


def test_unused_results_are_reported_but_do_not_stop_the_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "README.md").write_text("<!-- results: used -->\n\n<!-- /results -->\n")
    changed, unused = report.write(tmp_path, {"used": "1", "orphan": "2"})
    assert changed == [tmp_path / "README.md"]
    assert unused == ["orphan"]


def test_keys_are_reported_in_document_order() -> None:
    text = "<!-- results: z -->\n\n<!-- /results -->\n<!-- results: a -->\n\n<!-- /results -->\n"
    assert report.keys_in(text) == ["z", "a"]


def test_negative_zero_never_reaches_a_table() -> None:
    """Its sign depends on the last bit of a subtraction, which is not stable across BLAS."""
    assert report.fmt(-1e-9) == "0.000"
    assert report.fmt(-0.0) == "0.000"
    assert report.fmt(-0.5) == "-0.500"


def test_rounding_is_fixed_width_so_columns_line_up() -> None:
    assert report.fmt(1.0) == "1.000"
    assert report.fmt(0.5, places=1) == "0.5"
    assert report.fmt(12.3456) == "12.346"


def test_table_alignment_defaults_to_text_left_numbers_right() -> None:
    out = report.table(["name", "value"], [["a", "1.000"]])
    assert out.splitlines()[1] == "|---|---:|"


def test_table_rejects_a_mismatched_alignment_spec() -> None:
    with pytest.raises(report.ReportError):
        report.table(["a", "b"], [], align=["l"])
