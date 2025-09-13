# MLP Module Localization Plan

## Executive Summary
Restructure the MLP module to be self-contained within `kaggle_map/mlp/` with direct execution via `python -m kaggle_map.mlp`, removing dependency on centralized CLI infrastructure.

## WHY: Design Principles

### Core Motivations
1. **Simplicity Over Abstraction**: Remove unnecessary CLI layer that adds cognitive overhead
2. **Fail Fast and Clear**: Direct module execution provides clearer error paths
3. **Locality of Behavior**: All MLP logic stays within the MLP directory
4. **Parse, Don't Validate**: Use strong typing at module boundaries

### Benefits
- **Independence**: MLP module can evolve without affecting other modules
- **Testability**: Direct execution simplifies testing and debugging
- **Clarity**: No need to trace through CLI routing to understand execution
- **Maintenance**: Changes are localized, reducing risk of breaking other components

## Data Architecture

### Core Data Structures

```
TrainingConfig (Pydantic Model)
├── train_csv_path: Path
├── train_split: float (0.7)
├── epochs: int (50)
├── batch_size: int (256)
├── learning_rate: float (1e-4)
├── embedding_strategy: EmbeddingStrategy
└── architecture_size: ArchitectureSize

QuestionSpecificMLP (nn.Module)
├── trunk: Sequential layers (shared)
├── true_heads: ModuleDict (per question, correct answers)
├── false_heads: ModuleDict (per question, incorrect answers)
├── true_label_encoders: dict[QuestionId, LabelEncoder]
└── false_label_encoders: dict[QuestionId, LabelEncoder]

DatasetArrays
├── embeddings: np.ndarray [n_samples, embedding_dim]
├── question_ids: np.ndarray [n_samples]
├── predictions: np.ndarray [n_samples]
└── mc_answers: np.ndarray [n_samples]
```

### Data Flow Pipeline

```
1. CSV Loading
   datasets/train.csv → load_training_data() → list[TrainingRow]

2. Embedding Generation
   TrainingRow → EvaluationRow → encode() → torch.Tensor[embedding_dim]

3. Dataset Preparation
   embeddings + metadata → DatasetArrays → MLPDataset → DataLoader

4. Training Loop
   DataLoader → QuestionSpecificMLP → ListMLELoss → optimizer.step()

5. Prediction
   EvaluationRow → embedding → model.forward() → softmax → top-3 predictions

6. Persistence
   model.state_dict() → torch.save() → models/mlp.pkl
```

## File Structure Changes

### Files to Modify
1. `pyproject.toml` - Remove CLI entry point
2. `kaggle_map/mlp/predictor.py` → `kaggle_map/mlp/main.py` - Rename and add __main__
3. `kaggle_map/mlp/__init__.py` - Update imports
4. `README.md` - Update documentation
5. `Makefile` - Update commands for new execution pattern

### New Module Structure
```
kaggle_map/mlp/
├── __init__.py        # Public API exports
├── main.py            # Entry point with __main__ block
├── model.py           # Neural network architecture
├── trainer.py         # Training loop and utilities
├── dataset.py         # PyTorch dataset implementation
├── loss.py            # ListMLE loss function
├── label_encoder.py   # Label encoding utilities
└── *_test.py          # Test files
```

## Implementation Tasks

### Task 1: Remove CLI Infrastructure
**Goal**: Clean up pyproject.toml and remove CLI references

- [x] Remove `[project.scripts]` section from pyproject.toml
- [x] Verify no other files reference `kaggle_map.cli:main`
- [x] Check for any CLI-related imports in other modules
- [x] Ensure pyproject.toml remains valid TOML syntax
- [x] Document removal in git commit message

### Task 2: Rename and Restructure Main Module
**Goal**: Rename predictor.py to main.py maintaining all functionality

- [x] Git move `kaggle_map/mlp/predictor.py` to `kaggle_map/mlp/main.py`
- [x] Update module docstring to reflect new role as entry point
- [x] Verify all existing functions remain unchanged
- [x] Check for any hardcoded references to "predictor" in comments
- [x] Ensure all imports within the file are still valid

### Task 3: Implement Command-Line Interface
**Goal**: Add argparse-based CLI in __main__ block

- [x] Import argparse and sys modules
- [x] Create ArgumentParser with proper description
- [x] Add subparsers for fit, eval, predict commands
- [x] Define arguments for each subcommand:
  - fit: --train-data, --epochs, --batch-size, --learning-rate, --train-split, --model-path
  - eval: --model-path, --train-data, --train-split
  - predict: --model-path, --input-file, --output-file
- [x] Add proper help messages for each argument
- [x] Implement command dispatch logic
- [x] Add error handling for missing arguments
- [x] Include verbose/quiet logging options

### Task 4: Update Module Imports
**Goal**: Fix imports throughout MLP module

- [x] Update `kaggle_map/mlp/__init__.py` to import from `main` instead of `predictor`
- [x] Verify public API remains unchanged (fit, predict, evaluate, save, load)
- [x] Check for any cross-module imports that need updating
- [x] Test that `from kaggle_map.mlp import fit` still works
- [x] Update any test files that import from predictor

### Task 5: Implement Command Handlers
**Goal**: Create handler functions for each CLI command

- [x] Implement `handle_fit()` function:
  - Parse TrainingConfig from arguments
  - Call fit() with config
  - Save model to specified path
  - Display training metrics
- [x] Implement `handle_eval()` function:
  - Load model from path
  - Load and split data
  - Call evaluate()
  - Display MAP@3 score
- [x] Implement `handle_predict()` function:
  - Load model from path
  - Read input CSV
  - Generate predictions
  - Write output CSV
- [x] Add proper logging with loguru
- [x] Include progress indicators for long operations

### Task 6: Update Documentation
**Goal**: Update README and docstrings for new execution pattern

- [x] Update README.md Quick Start section:
  - Replace `kaggle-map run mlp` with `python -m kaggle_map.mlp`
  - Update all command examples
  - Add note about module-specific execution
- [x] Update Basic Commands section with new syntax
- [x] Add troubleshooting section for common issues
- [x] Update module docstrings to reflect new structure
- [x] Create examples for each command with expected output

### Task 7: Update Makefile Targets
**Goal**: Modify Makefile to use new execution pattern

- [x] Update `fit` target to use `python -m kaggle_map.mlp fit`
- [x] Update `eval` target to use `python -m kaggle_map.mlp eval`
- [x] Remove references to strategy selection (MLP-specific now)
- [x] Add new convenience targets if needed
- [x] Test all modified targets work correctly
- [x] Update help text to reflect changes

### Task 8: Error Handling and Validation
**Goal**: Add robust error handling throughout

- [ ] Validate file paths exist before processing
- [ ] Check model compatibility when loading
- [ ] Handle missing embeddings gracefully
- [ ] Add informative error messages for common failures:
  - Missing training data
  - Incompatible model versions
  - Out of memory errors
  - CUDA availability issues
- [ ] Implement graceful shutdown on Ctrl+C
- [ ] Add --dry-run option for testing configurations

### Task 9: Testing and Validation
**Goal**: Ensure all functionality works correctly

- [x] Test `python -m kaggle_map.mlp fit` with default settings
- [x] Test `python -m kaggle_map.mlp eval` with saved model
- [x] Test `python -m kaggle_map.mlp predict` with sample data
- [x] Verify backward compatibility of Python API
- [x] Test error cases (missing files, invalid arguments)
- [x] Run existing test suite and fix any failures
- [x] Add new tests for CLI functionality
- [x] Benchmark performance vs old implementation

### Task 10: Integration and Cleanup
**Goal**: Final integration and code cleanup

- [ ] Run `make dev` to ensure code quality
- [ ] Fix any linting or type checking issues
- [ ] Remove any dead code or unused imports
- [ ] Update type hints for new functions
- [ ] Ensure consistent logging throughout
- [ ] Create migration guide for users of old CLI
- [ ] Test on fresh environment
- [ ] Document any breaking changes

## Notes

- Keep changes atomic and reversible via git
- Maintain backward compatibility for Python API
- Consider adding `__main__.py` file as alternative to main.py
- Future consideration: Apply same pattern to LLM and other modules
