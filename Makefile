.PHONY: install install-dev lint format typecheck test ci

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check .

format:
	python -m ruff format .

typecheck:
	python -m mypy libs packages tools

test:
	python -m pytest

ci: lint typecheck test
