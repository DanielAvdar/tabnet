.PHONY: help

.PHONY: default
default: install


install:
	uv sync --all-extras --all-groups --frozen
	uv pip install pre-commit
	uv run pre-commit install

install-docs:
	uv sync --group docs --frozen --no-group dev

test:
	uv run pytest

check:
	uv run pre-commit run --all-files


build:
	uv build

coverage:
	uv run pytest --cov=pytorch_tabnet --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

cov:
	uv run pytest --cov=pytorch_tabnet --cov-report=term-missing

clear:
	uv venv --python 3.10

update:
	uv lock

	uvx pre-commit autoupdate
	$(MAKE) install


mypy:
	uv tool run mypy pytorch_tabnet --config-file pyproject.toml

doc:
	uv run sphinx-build -M html docs/source docs/build/
