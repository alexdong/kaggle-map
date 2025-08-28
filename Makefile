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

# Hyperparameter search commands for 4-8 hours blocks
search:
	@echo "========================================="
	@echo "Starting 4-Hour Focused Search"
	@echo "Start time: $$(date)"
	@echo "========================================="
	@echo ""
	@echo "Configuration:"
	@echo "- Strategy: mlp"
	@echo "- Estimated trials: ~80-100 (4 hours)"
	@echo "- Parallel jobs: 1 (single-threaded for stability)"
	@echo "- Stage: Exploitation focus"
	@echo ""
	@echo "Monitor progress at: https://wandb.ai/alex-xun-dong/kaggle-map-mlp"
	@echo ""
	uv run -m kaggle_map.optimise search mlp \
		--trials 500 \
		--jobs 1 \
		--timeout 14400

# Compare and analyze optimization results
#  - Use make list-studies for listing all studies
#  - Use make analyze STUDY=<study_name> for individual analysis
#  - Use make compare STUDIES='study1 study2 study3' to compare multiple studies
list-studies:
	uv run -m kaggle_map.optimise list-studies

compare:
	@if [ -z "$(STUDIES)" ]; then \
		echo "Usage: make compare STUDIES='study1 study2 study3'"; \
		echo "Example: make compare STUDIES='mlp_20240101_120000 mlp_20240102_140000'"; \
		exit 1; \
	fi
	uv run -m kaggle_map.optimise compare $(STUDIES)

analyze:
	@if [ -z "$(STUDY)" ]; then \
		echo "Usage: make analyze STUDY=<study_name>"; \
		echo "Example: make analyze STUDY=mlp_20240101_120000"; \
		exit 1; \
	fi
	uv run -m kaggle_map.optimise analyze $(STUDY)
