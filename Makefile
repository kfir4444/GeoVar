# GeoVar Makefile
# "The principle is that you should always be the best at everything." - Dwight K. Schrute

PYTHON = .venv/bin/python
PYTEST = .venv/bin/pytest
PYTHONPATH = src

.PHONY: test lint clean help

help:
	@echo "Available commands:"
	@echo "  make test  - Run all unit tests (The only way to ensure survival)"
	@echo "  make lint  - Check code quality (Identity theft is not a joke!)"
	@echo "  make clean - Remove temporary files"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests

lint:
	$(PYTHON) -m ruff check src tests

clean:
	rm -rf .pytest_cache
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Workspace sanitized."
