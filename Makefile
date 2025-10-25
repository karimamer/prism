# Makefile for PRISM Entity Resolution System

.PHONY: help install install-dev test test-unit test-integration test-benchmark lint format type-check security clean build docs

# Default target
help:
	@echo "PRISM Entity Resolution System - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make setup-hooks      Setup pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-benchmark   Run performance benchmarks"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linter (ruff)"
	@echo "  make format           Format code with ruff"
	@echo "  make type-check       Run type checker (mypy)"
	@echo "  make security         Run security scanner (bandit)"
	@echo "  make check-all        Run all quality checks"
	@echo ""
	@echo "Build & Clean:"
	@echo "  make clean            Remove build artifacts and cache"
	@echo "  make build            Build distribution packages"
	@echo ""

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

setup-hooks:
	uv pip install pre-commit
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-benchmark:
	pytest tests/benchmarks/ -v -m benchmark --benchmark-only

test-coverage:
	pytest tests/ -v --cov=src/entity_resolution --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	pytest tests/unit/ -v -m "unit and not slow" -n auto

# Code Quality
lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	ruff format .

format-check:
	ruff format --check .

type-check:
	mypy src/entity_resolution --ignore-missing-imports --no-strict-optional

security:
	bandit -r src/entity_resolution -f json -o bandit-report.json
	@echo "Security report generated: bandit-report.json"

check-all: lint format-check type-check security
	@echo "All quality checks passed!"

# Build
build:
	uv build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf bandit-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleaned build artifacts and cache"

# Development
run-example:
	python src/entity_resolution/run_entity_resolution.py \
		--input test_input.txt \
		--entities sample_entities.json \
		--output results.json

watch-test:
	pytest-watch tests/ -v

# CI/CD simulation
ci-local: clean install-dev lint format-check type-check test-coverage
	@echo "Local CI checks completed successfully!"
