.PHONY: dev test test-all fit eval search search-balanced compare list-studies analyze

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
# Examples:
#   make fit STRATEGY=baseline                                          # Train baseline with original dataset
#   make fit STRATEGY=mlp                                               # Train MLP with original dataset  
#   make fit STRATEGY=mlp DATASET=datasets/synth_median_balanced_354210_total.csv  # Train MLP with synthetic dataset
fit:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make fit STRATEGY=<strategy_name> [DATASET=<path>]"; \
		echo "Example: make fit STRATEGY=baseline"; \
		echo "Example: make fit STRATEGY=mlp DATASET=datasets/synth_median_balanced_354210_total.csv"; \
		exit 1; \
	fi
	@if [ -n "$(DATASET)" ]; then \
		uv run -m kaggle_map.cli run $(STRATEGY) fit --train-data $(DATASET); \
	else \
		uv run -m kaggle_map.cli run $(STRATEGY) fit; \
	fi

# Examples:
#   make eval STRATEGY=baseline                                         # Evaluate baseline with original dataset
#   make eval STRATEGY=mlp                                              # Evaluate MLP with original dataset
#   make eval STRATEGY=mlp DATASET=datasets/synth_median_balanced_354210_total.csv  # Evaluate MLP with synthetic dataset
eval:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make eval STRATEGY=<strategy_name> [DATASET=<path>]"; \
		echo "Example: make eval STRATEGY=baseline"; \
		echo "Example: make eval STRATEGY=mlp DATASET=datasets/synth_median_balanced_354210_total.csv"; \
		exit 1; \
	fi
	@if [ -n "$(DATASET)" ]; then \
		uv run -m kaggle_map.cli run $(STRATEGY) eval --train-data $(DATASET); \
	else \
		uv run -m kaggle_map.cli run $(STRATEGY) eval; \
	fi

# Hyperparameter search commands for 4-8 hours blocks
search:
	@echo "========================================="
	@echo "Starting 4-Hour Focused Search (Original Dataset)"
	@echo "Start time: $$(date)"
	@echo "========================================="
	@echo ""
	@echo "Configuration:"
	@echo "- Strategy: mlp"
	@echo "- Dataset: datasets/train.csv (original)"
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

search-balanced:
	@echo "========================================="
	@echo "Starting 4-Hour Focused Search (Balanced 10x Dataset)"
	@echo "Start time: $$(date)"
	@echo "========================================="
	@echo ""
	@echo "Configuration:"
	@echo "- Strategy: mlp"
	@echo "- Dataset: datasets/synth_median_balanced_354210_total.csv (10x balanced)"
	@echo "- Estimated trials: ~60-80 (4 hours, slower due to larger dataset)"
	@echo "- Parallel jobs: 1 (single-threaded for stability)"
	@echo "- Starting with best known params and exploring nearby"
	@echo ""
	@echo "Best Known Parameters (MAP@3: 0.9114):"
	@echo "  learning_rate: 0.0002126932668569146"
	@echo "  batch_size: 384"
	@echo "  dropout: 0.30108955018524314"
	@echo "  architecture_size: xlarge"
	@echo "  optimizer: adamw"
	@echo "  weight_decay: 0.003225818218347925"
	@echo "  activation: silu"
	@echo "  scheduler: none"
	@echo "  patience: 17"
	@echo "  epochs: 36"
	@echo ""
	@echo "Monitor progress at: https://wandb.ai/alex-xun-dong/kaggle-map-mlp"
	@echo ""
	uv run -m kaggle_map.optimise search mlp \
		--trials 500 \
		--jobs 1 \
		--timeout 14400 \
		--train-data datasets/synth_median_balanced_354210_total.csv

# Compare and analyze optimization results
list-studies:
	@echo "Press Ctrl+C to exit auto-refresh"
	@while true; do \
		clear; \
		echo "=== Studies List (Auto-refreshing every 5 seconds) ==="; \
		echo "Last updated: $$(date)"; \
		echo ""; \
		uv run -m kaggle_map.optimise list-studies; \
		echo ""; \
		echo "Press Ctrl+C to exit..."; \
		sleep 5; \
	done

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
