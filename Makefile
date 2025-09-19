# Default target when just running 'make'
.DEFAULT_GOAL := help

# ============================================================================
# Development Commands
# ============================================================================
.PHONY: dev test test-all

dev:
	uv run ruff check . --fix --unsafe-fixes
	uv run ruff format .
	@uv run ty check 2>&1 | grep -v "^WARN ty is pre-release" || true
	@uv run pyrefly check 2>&1 | grep -v "INFO" | grep -v "Building kaggle-map" | grep -v "Built kaggle-map" | grep -v "Uninstalled" | grep -v "Installed" || [ $$? -eq 1 ]

test:
	# Run fast tests only (excludes slow integration tests)
	uv run --only-dev -m pytest --durations=10 --randomly-seed=auto -m "not slow"

test-all:
	# Run all tests including slow integration tests
	uv run --only-dev -m pytest --durations=10 --randomly-seed=auto

# ============================================================================
# Model Training & Evaluation
# ============================================================================
.PHONY: fit eval

# Train the MLP model
# Usage:
#   make fit                                       # Train MLP with default dataset  
#   make fit DATASET=datasets/synth_median_balanced_354210_total.csv
#   make fit EPOCHS=100 LEARNING_RATE=0.001
fit:
	@echo "Training MLP model..."
	@if [ -n "$(DATASET)" ]; then \
		python -m kaggle_map.mlp fit --train-data $(DATASET) \
			$${EPOCHS:+--epochs $$EPOCHS} \
			$${LEARNING_RATE:+--learning-rate $$LEARNING_RATE} \
			$${BATCH_SIZE:+--batch-size $$BATCH_SIZE}; \
	else \
		python -m kaggle_map.mlp fit \
			$${EPOCHS:+--epochs $$EPOCHS} \
			$${LEARNING_RATE:+--learning-rate $$LEARNING_RATE} \
			$${BATCH_SIZE:+--batch-size $$BATCH_SIZE}; \
	fi

# Evaluate the MLP model
# Usage:
#   make eval                                      # Evaluate MLP with default model
#   make eval MODEL_PATH=models/custom.pkl
#   make eval DATASET=datasets/synth_median_balanced_354210_total.csv
eval:
	@echo "Evaluating MLP model..."
	@python -m kaggle_map.mlp eval \
		$${MODEL_PATH:+--model-path $$MODEL_PATH} \
		$${DATASET:+--train-data $$DATASET} \
		$${TRAIN_SPLIT:+--train-split $$TRAIN_SPLIT}


# ============================================================================
# Hyperparameter Optimization - MLP
# ============================================================================
.PHONY: search

# Standard hyperparameter search (4-hour session)
search:
	@echo "========================================="
	@echo "MLP Hyperparameter Search (4-hour session)"
	@echo "Start time: $$(date)"
	@echo "========================================="
	@echo ""
	@echo "Configuration:"
	@echo "- Strategy: mlp"
	@echo "- Dataset: datasets/train.csv (original)"
	@echo "- Max trials: 500"
	@echo "- Parallel jobs: 1"
	@echo ""
	@echo ""
	uv run -m kaggle_map.optimise.tune \
		--study-name mlp_search \
		--trials 500 \
		--jobs 1
