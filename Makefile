.PHONY: dev test test-all fit-baseline eval-baseline predict-baseline

dev:
	uv run ruff check . --fix --unsafe-fixes
	uv run ruff format .
	uv run ty check .

test:
	# Run fast tests only (excludes slow integration tests)
	uv run --only-dev -m pytest -m "not slow"

test-all:
	# Run all tests including slow integration tests
	uv run --only-dev -m pytest

# Generic fit and eval commands that take strategy as parameter
fit:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make fit STRATEGY=<strategy_name>"; \
		echo "Example: make fit STRATEGY=baseline"; \
		exit 1; \
	fi
	uv run -m kaggle_map.cli run $(STRATEGY) fit

eval:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make eval STRATEGY=<strategy_name>"; \
		echo "Example: make eval STRATEGY=baseline"; \
		exit 1; \
	fi
	uv run -m kaggle_map.cli run $(STRATEGY) eval
