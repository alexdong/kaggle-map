# Default target when just running 'make'
.DEFAULT_GOAL := help

# ============================================================================
# Development Commands
# ============================================================================
.PHONY: dev test test-all

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

# ============================================================================
# Model Training & Evaluation
# ============================================================================
.PHONY: fit eval

# Train a strategy model
# Usage:
#   make fit STRATEGY=baseline                    # Train baseline with default dataset
#   make fit STRATEGY=mlp                         # Train MLP with default dataset  
#   make fit STRATEGY=mlp DATASET=datasets/synth_median_balanced_354210_total.csv
fit:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make fit STRATEGY=<strategy_name> [DATASET=<path>]"; \
		echo ""; \
		echo "Available strategies:"; \
		echo "  baseline - Baseline strategy using misconception frequency"; \
		echo "  mlp      - Multi-layer perceptron with embeddings"; \
		echo "  llm      - Large language model approach"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make fit STRATEGY=baseline"; \
		echo "  make fit STRATEGY=mlp DATASET=datasets/synth_median_balanced_354210_total.csv"; \
		exit 1; \
	fi
	@if [ -n "$(DATASET)" ]; then \
		uv run -m kaggle_map.cli run $(STRATEGY) fit --train-data $(DATASET); \
	else \
		uv run -m kaggle_map.cli run $(STRATEGY) fit; \
	fi

# Evaluate a trained model
# Usage:
#   make eval STRATEGY=baseline                   # Evaluate baseline
#   make eval STRATEGY=mlp                        # Evaluate MLP
#   make eval STRATEGY=mlp DATASET=datasets/synth_median_balanced_354210_total.csv
eval:
	@if [ -z "$(STRATEGY)" ]; then \
		echo "Usage: make eval STRATEGY=<strategy_name> [DATASET=<path>]"; \
		echo ""; \
		echo "Available strategies:"; \
		echo "  baseline - Baseline strategy using misconception frequency"; \
		echo "  mlp      - Multi-layer perceptron with embeddings"; \
		echo "  llm      - Large language model approach"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make eval STRATEGY=baseline"; \
		echo "  make eval STRATEGY=mlp DATASET=datasets/synth_median_balanced_354210_total.csv"; \
		exit 1; \
	fi
	@if [ -n "$(DATASET)" ]; then \
		uv run -m kaggle_map.cli run $(STRATEGY) eval --train-data $(DATASET); \
	else \
		uv run -m kaggle_map.cli run $(STRATEGY) eval; \
	fi

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
	@echo "Monitor at: https://wandb.ai/alex-xun-dong/kaggle-map-mlp"
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
	@echo "Monitor at: https://wandb.ai/alex-xun-dong/kaggle-map-mlp"
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
	@echo ""
	@echo "Testing quantizations:"
	@echo "  IQ4_XS  (6.55 GB)"
	@echo "  IQ4_NL  (6.89 GB)"
	@echo "  Q4_0    (6.91 GB)"
	@echo "  Q4_1    (7.56 GB)"
	@echo "  Q4_K_S  (6.94 GB)"
	@echo "  Q4_K_M  (7.30 GB) - recommended"
	@echo "  Q4_K_XL (7.43 GB)"
	@echo ""
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
	@echo "Training & Evaluation:"
	@echo "  make fit STRATEGY=<name>   - Train a strategy"
	@echo "  make eval STRATEGY=<name>  - Evaluate a strategy"
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
	@echo "Run 'make <target> STRATEGY=?' for available strategies"