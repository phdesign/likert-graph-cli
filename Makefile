init:
	test -e .venv || python3 -m venv .venv
	source .venv/bin/activate; \
		pip install -e .[dev]

lint:
	source .venv/bin/activate; \
		black -l 120 *.py; \
		isort *.py; \
		flake8 --exit-zero *.py

.PHONY: init lint