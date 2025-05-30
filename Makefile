.PHONY: help

.PHONY: default
default: install


install:
	uv sync --all-extras --all-groups --frozen
	uvx pre-commit install

install-docs:
	uv sync --group docs --frozen --no-group dev

test: install
	uv run pytest

check: install
	uvx  pre-commit run --all-files


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



# Add doctests target to specifically run doctest validation
doctest: install-docs doc

# Update doc target to run doctests as part of documentation build
doc:
	uv run --no-project sphinx-build -M doctest docs/source docs/build/ -W --keep-going --fresh-env
	uv run --no-project sphinx-build -M html docs/source docs/build/ -W --keep-going --fresh-env

# Optional target that builds docs but ignores warnings
doc-ignore-warnings:
	uv run sphinx-build -M html docs/source docs/build/

# Run all checks in sequence: tests, code quality, type checking, and documentation
check-all: check test mypy doc

vs-code:
	uvx --from dev-kit-mcp-server dkmcp-vscode
