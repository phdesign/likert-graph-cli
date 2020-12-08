VENV_NAME ?= .venv
PYTHON ?= python3

ifdef OS
        VENV_ACTIVATE ?= $(VENV_NAME)/Scripts/activate
else
        VENV_ACTIVATE ?= $(VENV_NAME)/bin/activate
endif

init:
	test -e $(VENV_ACTIVATE) || $(PYTHON) -m venv $(VENV_NAME)
	source $(VENV_ACTIVATE); \
		pip install -e .[dev]

lint:
	source $(VENV_ACTIVATE); \
		black -l 120 *.py; \
		isort *.py; \
		flake8 --exit-zero *.py

test: lint
	source $(VENV_ACTIVATE); \
		coverage run -m pytest; \
		coverage report

.PHONY: init lint test
.SILENT:
