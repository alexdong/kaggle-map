.PHONY: dev test test-all fit eval search search-grid compare list-studies analyze

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

# Hyperparameter search commands
search:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make search STRATEGY=<strategy_name> [TRIALS=100] [JOBS=1]"; \
		echo "Example: make search STRATEGY=mlp TRIALS=50 JOBS=4"; \
		exit 1; \
	fi
	uv run -m kaggle_map.hypersearch $(STRATEGY) \
		--n-trials $(or $(TRIALS),100) \
		--n-jobs $(or $(JOBS),1) \
		--metric $(or $(METRIC),accuracy) \
		--sampler $(or $(SAMPLER),tpe)

search-grid:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make search-grid STRATEGY=<strategy_name>"; \
		echo "Example: make search-grid STRATEGY=mlp"; \
		exit 1; \
	fi
	uv run -m kaggle_map.hypersearch $(STRATEGY) \
		--n-trials 1000 \
		--n-jobs $(or $(JOBS),4) \
		--sampler grid

search-quick:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make search-quick STRATEGY=<strategy_name>"; \
		echo "Example: make search-quick STRATEGY=mlp"; \
		echo "Runs 20 trials for quick exploration"; \
		exit 1; \
	fi
	uv run -m kaggle_map.hypersearch $(STRATEGY) \
		--n-trials 20 \
		--n-jobs $(or $(JOBS),2) \
		--metric accuracy

# Compare and analyze hypersearch results
list-studies:
	uv run -m kaggle_map.compare list-studies

compare:
	@if [ -z "$(STUDIES)" ]; then \
		echo "Usage: make compare STUDIES='study1 study2 study3'"; \
		echo "Example: make compare STUDIES='mlp_20240101_120000 mlp_20240102_140000'"; \
		exit 1; \
	fi
	uv run -m kaggle_map.compare compare $(STUDIES)

analyze:
	@if [ -z "$(STUDY)" ]; then \
		echo "Usage: make analyze STUDY=<study_name>"; \
		echo "Example: make analyze STUDY=mlp_20240101_120000"; \
		exit 1; \
	fi
	uv run -m kaggle_map.compare analyze $(STUDY)

# Launch Optuna dashboard for visualization (requires optuna-dashboard)
dashboard:
	@echo "Starting Optuna Dashboard on http://localhost:8080"
	@echo "Press Ctrl+C to stop"
	optuna-dashboard sqlite:///optuna.db
