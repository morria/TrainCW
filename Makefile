.PHONY: help install install-dev test lint format check clean build docs

# Default target
.DEFAULT_GOAL := help

# Python executable
PYTHON := python3

# Package name
PACKAGE := traincw

help: ## Show this help message
	@echo "TrainCW - Morse Code Neural Network Training System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install package in development mode
	$(PYTHON) -m pip install -e .

install-dev: ## Install package with development dependencies
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

install-all: ## Install package with all optional dependencies
	$(PYTHON) -m pip install -e ".[all]"

test: ## Run tests with pytest
	pytest -v --cov=$(PACKAGE) --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	pytest -v -x

test-parallel: ## Run tests in parallel
	pytest -v -n auto

test-slow: ## Run all tests including slow tests
	pytest -v -m ""

lint: ## Run linter (ruff)
	ruff check .

lint-fix: ## Run linter with auto-fix
	ruff check --fix .

format: ## Format code with ruff
	ruff format .

format-check: ## Check code formatting without making changes
	ruff format --check .

typecheck: ## Run type checker (mypy)
	mypy src/$(PACKAGE)

check: ## Run all checks (lint, format, typecheck, test)
	@echo "Running all checks..."
	@echo "\n=== Formatting Check ==="
	ruff format --check .
	@echo "\n=== Linting ==="
	ruff check .
	@echo "\n=== Type Checking ==="
	mypy src/$(PACKAGE) --ignore-missing-imports || true
	@echo "\n=== Tests ==="
	pytest -v --cov=$(PACKAGE) --cov-report=term-missing
	@echo "\n✅ All checks complete!"

clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete

build: clean ## Build distribution packages
	$(PYTHON) -m build

upload-test: build ## Upload to Test PyPI
	twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	twine upload dist/*

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Development helpers
dev-setup: ## Complete development environment setup
	@echo "Setting up development environment..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install
	@echo "\n✅ Development environment ready!"
	@echo "Run 'make check' to verify everything works."

quick-test: ## Quick smoke test
	$(PYTHON) -c "import $(PACKAGE); print(f'$(PACKAGE) {$(PACKAGE).__version__} imported successfully')"
	traincw --version

watch-test: ## Watch for changes and run tests automatically
	ptw -- -v

.PHONY: dev-setup quick-test watch-test
