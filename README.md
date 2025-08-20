# kaggle-map

Map charting student math misunderstandings with comprehensive tooling and documentation.

## Purpose

This project implements machine learning strategies to predict student misconceptions in mathematics problems for the Kaggle MAP competition. It provides a modular framework for training, evaluating, and generating predictions using various prediction strategies.

## Quick Start

1. **Install dependencies**: `uv install`
2. **List available strategies**: `uv run -m kaggle_map.cli list-strategies`
3. **Train a model**: `uv run -m kaggle_map.cli run baseline fit`
4. **Evaluate performance**: `uv run -m kaggle_map.cli run baseline eval`

## Basic Commands

### Strategy Management

```bash
# List all available prediction strategies
uv run -m kaggle_map.cli list-strategies
```

### Model Training

```bash
# Train a strategy with default settings (80% training split, seed 42)
uv run -m kaggle_map.cli run <strategy> fit

# Train with custom parameters
uv run -m kaggle_map.cli run <strategy> fit --train-split 0.7 --random-seed 123

# Train with pre-computed embeddings
uv run -m kaggle_map.cli run <strategy> fit --embeddings-path embeddings.npz

# Save model to custom location
uv run -m kaggle_map.cli run <strategy> fit --output-path my_model.pkl
```

### Model Evaluation

```bash
# Evaluate using default model path
uv run -m kaggle_map.cli run <strategy> eval

# Evaluate with custom model path
uv run -m kaggle_map.cli run <strategy> eval --model-path my_model.pkl

# Evaluate with specific train/test split
uv run -m kaggle_map.cli run <strategy> eval --train-split 0.8
```

### Prediction Generation

```bash
# Generate predictions (not yet implemented)
uv run -m kaggle_map.cli run <strategy> predict

# Generate predictions with custom model and output paths
uv run -m kaggle_map.cli run <strategy> predict --model-path my_model.pkl --output-path predictions.csv
```

### Additional Options

- `--verbose, -v`: Show detailed model information
- `--train-split`: Fraction of data for training (default: 0.8)
- `--random-seed`: Random seed for reproducible results (default: 42)

## Development Commands

```bash
# Run linting and type checking
make dev

# Run tests (fast tests only)
make test

# Run all tests including slow integration tests
make test-all
```

## Available Strategies

The project uses a dynamic strategy discovery system that automatically finds and registers prediction strategies in the `kaggle_map/strategies/` directory. Each strategy implements the base `Strategy` class and provides methods for training (`fit`), evaluation (`eval`), and prediction (`predict`).

Run `uv run -m kaggle_map.cli list-strategies` to see currently available strategies and their descriptions.
