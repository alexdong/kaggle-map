# kaggle-map

Map charting student math misunderstandings with comprehensive tooling and documentation.

## Features

- **Modern Python tooling**: uv, ruff, pytest, pydantic
- **Comprehensive documentation**: Ready-to-use guides for common Python packages
- **GitHub Pages ready**: The `docs/` folder can be directly hosted as a static documentation site
- **Best practices**: Pre-configured with Python coding standards and project structure

## Documentation

The `docs/` folder contains comprehensive documentation that can be hosted directly on GitHub Pages:

1. Go to your repository Settings → Pages
2. Under "Source", select "Deploy from a branch"
3. Choose "main" branch and "/docs" folder
4. Click Save

Your documentation will be available at: `https://[username].github.io/[repository-name]/`

## Quick Start

### Running the Baseline Model

Test the baseline model for student misconception prediction:

```bash
# Run the model demonstration
uv run kaggle_map/models.py

# This will:
# - Fit the model from dataset/train.csv
# - Display model statistics
# - Save the model to baseline_model.json
# - Test serialization by loading the saved model
```

### Development Commands

```bash
# Run code quality checks
make dev

# Run tests
make test
```

## Project Structure

```
./
├── docs/                   # Documentation (GitHub Pages ready)
├── kaggle_map/             # Application code
├── dataset/                # Training and test data
├── tests/                  # Test files
├── logs/                   # Implementation logs
├── llms/                   # LLM-friendly documentation
├── Makefile               # Task automation
├── pyproject.toml         # Project configuration
├── Python.md              # Python coding standards
└── CLAUDE.md              # AI assistant instructions
```