# MLP Neural Network Strategy Implementation Plan

## 1. Overview

### 1.1 Objective  
[ ] Implement a question-specific MLP for **misconception detection**

### 1.2 Key Architectural Insight
[ ] **Category is mostly deterministic**: `True_/False_` prefix from answer correctness, suffix from misconception + quality
[ ] **Focus MLP on**: Question-specific misconception detection
[ ] **Leverage existing signals**: Use `MC_Answer` correctness like existing strategies

### 1.3 Input-Output Specification
[ ] **Input**: 
   - "Question: {}; Student's Answer: {}; Student's Explanation: {}" ->Embedding ( 384-dim)
   - `is_correct_answer` boolean (derived from `MC_Answer` vs ground truth)

[ ] **Output**:
   - **Question-specific misconception prediction**: Separate output head per question
   - Each head predicts misconceptions valid for that question + "NA" class

### 1.4 Integration Requirements
[ ] Follow existing `Strategy` pattern from `kaggle_map/strategies/base.py`
[ ] Maintain compatibility with existing evaluation pipeline
[ ] Use same data structures: `TrainingRow`, `TestRow`, `SubmissionRow`, `Prediction`

## 2. Architecture Design

### 2.1 Text Encoding Strategy
[x] **Tokenizer**: `utils/tokenizer.py` (implemented with `build_qae_text` and truncation logic)
[x] **Formula normalization**: `utils/formula.py` (LaTeX answer cleaning, text composition)
[x] **Embedding model registry**: `utils/embeddings.py` (EmbeddingModel enum with specs)

### 2.2 Question-Specific Architecture
[ ] **Shared Trunk**: 
   - Input: embedding(384) + is_correct(1) = 385 dimensions
   - Hidden 1: 512 units, ReLU, 0.3 dropout
   - Hidden 2: 256 units, ReLU, 0.3 dropout
   - Shared representation: 128 units

[ ] **Question-Specific Misconception Heads** (`nn.ModuleDict`): 
   - Q31772: 3 outputs → [Incomplete, WNB, NA]
   - Q31774: 4 outputs → [Mult, SwapDividend, FlipChange, NA]  
   - Q31778: 4 outputs → [Additive, Irrelevant, WNB, NA]
   - Q32829: 4 outputs → [Not_variable, Adding_terms, Inverse_operation, NA]
   - ... (one head per question with question-specific misconceptions + NA)

### 2.3 Loss Function Design
[ ] **Misconception Loss**: `BCEWithLogitsLoss` for question-specific multi-label prediction
[ ] **Training**: Each question's head trained on all samples for that question
[ ] **Loss aggregation**: Average loss across all question heads

## 3. Data Processing Pipeline

### 3.1 Question-Specific Data Preparation
[x] Parse training CSV and extract correct answers (existing in `strategies/baseline.py`)
[x] Extract misconceptions per question (existing in `strategies/baseline.py`)
[ ] Refactor `_extract_correct_answers` and `_extract_most_common_misconceptions` into shared `parsers.py`
[ ] Build question-specific misconception mappings:
   ```python
   question_misconceptions = {
       31772: ["Incomplete", "WNB", "NA"],
       31774: ["Mult", "SwapDividend", "FlipChange", "NA"],  
       31778: ["Additive", "Irrelevant", "WNB", "NA"],
       # ... for all 15 questions
   }
   ```

### 3.2 Label Engineering
[ ] **Misconception Labels**: Question-specific multi-hot encoding per question
   - Q31772 example: [1, 0, 0] for "Incomplete", [0, 1, 0] for "WNB", [0, 0, 1] for "NA"
[ ] **Answer Correctness**: Boolean feature from `MC_Answer` == correct answer

### 3.3 Category Reconstruction Logic
[ ] **Deterministic category generation** from MLP outputs:
   ```python
   def reconstruct_predictions(misconception_probs, is_correct, question_id):
       prefix = "True_" if is_correct else "False_"
       misconception_vocab = question_misconceptions[question_id]
       predictions = []
       
       # Check for misconceptions (exclude NA)
       for i, prob in enumerate(misconception_probs[:-1]):  # Skip NA
           if prob > 0.5:
               misconception = misconception_vocab[i]
               pred = Prediction(
                   category=Category(prefix + "Misconception"),
                   misconception=misconception
               )
               predictions.append((pred, prob))
       
       # If no misconceptions detected, use NA probability for Neither
       if not predictions:
           na_prob = misconception_probs[-1]  # Last element is NA
           pred = Prediction(category=Category(prefix + "Neither"))
           predictions.append((pred, na_prob))
       
       return sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
   ```

### 3.4 Data Splitting
[ ] Use 80% train.csv for training and use the rest 20% for validation

## 4. Implementation Details

### 4.1 Technology Stack
[ ] **Framework**: PyTorch for neural network implementation
[x] **Embeddings**: `sentence-transformers` library (model registry in `utils/embeddings.py`)
[x] **Strategy Integration**: Inherit from existing `Strategy` base class (interface defined)
[ ] **Serialization**: 
    - Save embeddings using `np.savez` for (embedding, label); 
    - model weights + metadata as JSON/pickle

### 4.2 File Structure
[ ] Create `kaggle_map/strategies/mlp.py` with `MLPStrategy` class
[ ] Add `mlp_test.py` for unit tests following existing patterns
[ ] Cache embeddings in `cache/` directory (gitignored)

### 4.3 Training Configuration
[ ] **Optimizer**: Adam with learning rate 1e-3
[ ] **Batch Size**: 64 (adjust based on memory constraints)
[ ] **Epochs**: 50 with early stopping (patience=10)
[ ] **Hardware**: CPU training initially, GPU if available

## 5. Strategy Class Implementation

### 5.1 Core Methods (from base.Strategy)
[x] **Strategy interface defined**: All abstract methods specified in `base.py`
[ ] `name` property: return `"mlp"`
[ ] `description` property: return `"Question-specific MLP for misconception detection and reasoning quality"`
[ ] `fit(train_csv_path)` → `MLPStrategy`: Train the neural network
[ ] `predict(test_data)` → `list[SubmissionRow]`: Generate predictions
[ ] `save(filepath)` and `load(filepath)`: Model persistence
[ ] `display_stats()`, `display_detailed_info()`, `demonstrate_predictions()`

## 6. Testing and Validation

### 6.1 Unit Tests
[x] **Strategy discovery system**: Dynamic registry implemented in `strategies/__init__.py`
[x] **Test infrastructure**: Unit test structure exists (`*_test.py` files)
[ ] Test fit() method with small synthetic dataset
[ ] Test predict() method output format compliance
[ ] Test save/load roundtrip preservation
[ ] Test embedding generation consistency

### 6.2 Integration Tests
[x] **Evaluation pipeline**: Exists in `eval.py`
[x] **CLI interface**: Exists in `fit.py`, `predict.py`
[ ] Test with existing evaluation pipeline (`eval.py`)
[ ] Verify compatibility with CLI interface (`fit.py`, `predict.py`)
[ ] Test performance measurement and logging

### 6.3 Quality Assurance
[x] **Quality tools configured**: `make dev` and `make test` implemented
[ ] Run `make dev` for linting and type checking
[ ] Run `make test` for full test suite
[ ] Validate MAP@3 evaluation integration

## 7. Error Handling and Robustness

### 7.1 Input Validation
[ ] Assert training data is not empty
[ ] Validate embedding dimensions match expected size
[ ] Check for missing or malformed text inputs
[ ] Handle edge cases (empty explanations, very long texts)

### 7.2 Training Robustness
[ ] Implement gradient clipping to prevent exploding gradients
[ ] Add loss monitoring and NaN detection
[ ] Graceful handling of CUDA out-of-memory errors
[ ] Save model checkpoints during training

## 8. Documentation and Logging

### 8.1 Code Documentation
[ ] Add comprehensive docstrings following existing style
[ ] Document architecture decisions and hyperparameter choices
[ ] Include usage examples in docstrings

### 8.2 Training Logging
[ ] Log training/validation loss curves
[ ] Log embedding generation progress
[ ] Log model statistics (parameters, misconception vocab size)
[ ] Integration with existing `loguru` logging