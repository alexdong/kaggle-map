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
	uv run --only-dev -m pytest -m "not slow"

test-all:
	# Run all tests including slow integration tests
	uv run --only-dev -m pytest

# ============================================================================
# Model Training & Evaluation
# ============================================================================
.PHONY: fit eval predict

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

# Generate predictions
# Usage:
#   make predict INPUT=test.csv OUTPUT=submission.csv
#   make predict INPUT=test.csv OUTPUT=submission.csv MODEL_PATH=models/custom.pkl
predict:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make predict INPUT=<input_file> OUTPUT=<output_file> [MODEL_PATH=<path>]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make predict INPUT=test.csv OUTPUT=submission.csv"; \
		exit 1; \
	fi
	@echo "Generating predictions..."
	@python -m kaggle_map.mlp predict \
		--input-file $(INPUT) \
		--output-file $(OUTPUT) \
		$${MODEL_PATH:+--model-path $$MODEL_PATH}

# ============================================================================
# Hyperparameter Optimization - MLP
# ============================================================================
.PHONY: search search-balanced search-embeddings

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
	@echo "- Max trials: 500 (or 4 hours)"
	@echo "- Parallel jobs: 1"
	@echo ""
	@echo ""
	uv run -m kaggle_map.optimise mlp search \
		--trials 500 \
		--jobs 1 \
		--timeout 14400

# Search with balanced synthetic dataset
search-balanced:
	@echo "========================================="
	@echo "MLP Search with Balanced Dataset (4-hour session)"
	@echo "Start time: $$(date)"
	@echo "========================================="
	@echo ""
	@echo "Configuration:"
	@echo "- Strategy: mlp"
	@echo "- Dataset: datasets/synth_median_balanced_354210_total.csv"
	@echo "- Max trials: 500 (or 4 hours)"
	@echo "- Parallel jobs: 1"
	@echo ""
	@echo ""
	uv run -m kaggle_map.optimise mlp search \
		--trials 500 \
		--jobs 1 \
		--timeout 14400 \
		--train-data datasets/synth_median_balanced_354210_total.csv

# Compare different embedding models
search-embeddings:
	@echo "========================================="
	@echo "Embedding Model Comparison (6-hour session)"
	@echo "Start time: $$(date)"
	@echo "========================================="
	@echo ""
	@echo "Testing 7 embedding models:"
	@echo "  - MiniLM-L6-v2 (384 dim)"
	@echo "  - E5-base-v2 (768 dim)"
	@echo "  - Instructor-base (768 dim)"
	@echo "  - BGE-base-en-v1.5 (768 dim)"
	@echo "  - Contriever (768 dim)"
	@echo "  - Sentence-T5-base (768 dim)"
	@echo "  - MiniLM-L12-v2 (384 dim)"
	@echo ""
	uv run -m kaggle_map.optimise mlp search-embeddings \
		--trials 70 \
		--jobs 1 \
		--timeout 21600

# ============================================================================
# Hyperparameter Optimization - LLM
# ============================================================================
.PHONY: search-llm

# Compare LLM quantization options
search-llm:
	@echo "========================================="
	@echo "LLM GGUF Quantization Comparison"
	@echo "Start time: $$(date)"
	@echo "========================================="
	uv run -m kaggle_map.optimise llm compare --sample-size 100

# ============================================================================
# Study Analysis & Visualization
# ============================================================================
.PHONY: list-studies dashboard analyze

# List all optimization studies
list-studies:
	@uv run -m kaggle_map.optimise list-studies

# Launch interactive Optuna dashboard
dashboard:
	@echo "Launching Optuna Dashboard..."
	@echo "Open http://127.0.0.1:8080 in your browser"
	@echo "Press Ctrl+C to stop"
	@echo ""
	@uv run -m kaggle_map.optimise dashboard

# Analyze a specific study
# Usage: make analyze STUDY=mlp_20240101_120000
analyze:
	@if [ -z "$(STUDY)" ]; then \
		echo "Usage: make analyze STUDY=<study_name>"; \
		echo ""; \
		echo "First run 'make list-studies' to see available studies"; \
		echo ""; \
		echo "Example:"; \
		echo "  make analyze STUDY=mlp_20240101_120000"; \
		exit 1; \
	fi
	@uv run -m kaggle_map.optimise mlp analyze $(STUDY)

# ============================================================================
# Help
# ============================================================================
.PHONY: help

help:
	@echo "Kaggle MAP Competition - Makefile Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Run linting and type checking"
	@echo "  make test         - Run fast tests only"
	@echo "  make test-all     - Run all tests including slow ones"
	@echo ""
	@echo "MLP Model Training & Evaluation:"
	@echo "  make fit                   - Train MLP model"
	@echo "  make eval                  - Evaluate MLP model"
	@echo "  make predict INPUT=<file> OUTPUT=<file>  - Generate predictions"
	@echo ""
	@echo "Optional parameters:"
	@echo "  DATASET=<path>       - Custom training data path"
	@echo "  MODEL_PATH=<path>    - Custom model path"
	@echo "  EPOCHS=<n>           - Number of training epochs"
	@echo "  LEARNING_RATE=<f>    - Learning rate value"
	@echo "  BATCH_SIZE=<n>       - Batch size for training"
	@echo ""
	@echo "Hyperparameter Optimization:"
	@echo "  make search              - MLP hyperparameter search (4h)"
	@echo "  make search-balanced     - MLP search with balanced dataset (4h)"
	@echo "  make search-embeddings   - Compare embedding models (6h)"
	@echo "  make search-llm          - Compare LLM quantizations"
	@echo ""
	@echo "Analysis & Visualization:"
	@echo "  make list-studies        - List all optimization studies"
	@echo "  make dashboard           - Launch Optuna dashboard"
	@echo "  make analyze STUDY=<id>  - Analyze specific study"
	@echo ""
	@echo "Examples:"
	@echo "  make fit EPOCHS=100 LEARNING_RATE=0.001"
	@echo "  make eval MODEL_PATH=models/custom.pkl"
	@echo "  make predict INPUT=test.csv OUTPUT=submission.csv"
