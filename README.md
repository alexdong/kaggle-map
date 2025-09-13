# kaggle-map

Map charting student math misunderstandings with comprehensive tooling and documentation.

## Purpose

This project implements machine learning strategies to predict student misconceptions in mathematics problems for the Kaggle MAP competition. It provides a modular framework for training, evaluating, and generating predictions using various prediction strategies.

## Quick Start

1. **Install dependencies**: `uv install`
2. **Train MLP model**: `python -m kaggle_map.mlp fit`
3. **Evaluate performance**: `python -m kaggle_map.mlp eval`
4. **Generate predictions**: `python -m kaggle_map.mlp predict --input-file test.csv --output-file submission.csv`

## Basic Commands

### MLP Module Direct Execution

The MLP module can now be executed directly without the centralized CLI:

```bash
# Display help and available commands
python -m kaggle_map.mlp --help
```

### Model Training

```bash
# Train with default settings (70% training split, 50 epochs)
python -m kaggle_map.mlp fit

# Train with custom parameters
python -m kaggle_map.mlp fit --epochs 100 --learning-rate 0.001 --batch-size 512

# Train with specific data and save location
python -m kaggle_map.mlp fit --train-data datasets/custom.csv --model-path models/custom.pkl

# Enable verbose logging
python -m kaggle_map.mlp fit -v
```

### Model Evaluation

```bash
# Evaluate using default model path
python -m kaggle_map.mlp eval

# Evaluate with custom model path
python -m kaggle_map.mlp eval --model-path models/custom.pkl

# Evaluate with specific train/test split
python -m kaggle_map.mlp eval --train-split 0.8
```

### Prediction Generation

```bash
# Generate predictions for submission (requires input and output files)
python -m kaggle_map.mlp predict --input-file test.csv --output-file submission.csv

# Use custom model for predictions
python -m kaggle_map.mlp predict --model-path models/custom.pkl --input-file test.csv --output-file predictions.csv

# Enable verbose logging during prediction
python -m kaggle_map.mlp predict --input-file test.csv --output-file submission.csv -v
```

### Additional Options

- `--verbose, -v`: Show detailed logging and progress information
- `--train-split`: Fraction of data for training (default: 0.7)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 256)
- `--learning-rate`: Learning rate for optimizer (default: 1e-4)

## Development Commands

```bash
# Run linting and type checking
make dev

# Run tests (fast tests only)
make test

# Run all tests including slow integration tests
make test-all
```

## Module Architecture

The project is organized into self-contained modules that can be executed independently:

- **MLP Module** (`kaggle_map.mlp`): Multi-Layer Perceptron model with embeddings for misconception prediction
- **LLM Module** (`kaggle_map.llm`): Large Language Model approaches using GGUF models
- **Core Module** (`kaggle_map.core`): Shared data structures and utilities

Each module provides its own command-line interface and can be executed directly using `python -m kaggle_map.<module_name>`.
