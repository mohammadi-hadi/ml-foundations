PY ?= python
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup test lint report check clean

help:
	@echo "setup   create a virtualenv and install the package with dev extras"
	@echo "test    run the test suite"
	@echo "lint    run ruff and mypy"
	@echo "report  re-run every experiment and write the numbers into the lessons"
	@echo "check   test + lint + fail if the committed numbers drifted"

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests
	$(BIN)/mypy src

report:
	$(BIN)/ml-foundations report

# What CI runs. The last two lines are the promise the README makes: regenerate
# every number from the code, and fail if what is committed no longer matches.
check: test lint report
	git diff --exit-code -- README.md lessons

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
