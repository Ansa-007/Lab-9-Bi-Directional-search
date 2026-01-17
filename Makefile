# Makefile for Bi-Directional Search Laboratory

.PHONY: help install test lint format clean run-examples run-dashboard docs

# Default target
help:
	@echo "Bi-Directional Search Laboratory - Available Commands:"
	@echo ""
	@echo "  install     - Install dependencies"
	@echo "  test        - Run unit tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean temporary files"
	@echo "  run-examples - Run basic usage examples"
	@echo "  run-dashboard - Start interactive dashboard"
	@echo "  docs        - Generate documentation"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python -m pytest tests/ -v

# Run linting
lint:
	flake8 bidirectional_search/ examples/ tests/
	mypy bidirectional_search/

# Format code
format:
	black bidirectional_search/ examples/ tests/

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/

# Run basic examples
run-examples:
	python examples/basic_usage.py

# Run advanced benchmarking examples
run-benchmarks:
	python examples/advanced_benchmarking.py

# Start interactive dashboard
run-dashboard:
	python examples/interactive_dashboard.py

# Generate documentation (placeholder)
docs:
	@echo "Documentation is available in the docs/ directory"
	@echo "Main files:"
	@echo "  docs/theory.md - Algorithm theory"
	@echo "  docs/api.md - API reference"

# Development setup
dev-setup: install
	@echo "Development environment setup complete"
	@echo "Run 'make test' to verify installation"

# Quick test
quick-test:
	python -c "from bidirectional_search import BiDirectionalSearch, GraphGenerator; print('Import successful!')"
