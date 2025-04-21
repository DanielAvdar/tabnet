.PHONY: help

install:
	uv sync --all-extras --all-groups --frozen
	uv run pip install pre-commit

.PHONY: default
default: install

test:
	uv run pytest

check:
	uvx pre-commit run --all-files


build:
	uv build

coverage:
	uv run pytest --cov=pytorch_tabnet --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

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
