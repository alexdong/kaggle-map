# MLP Neural Network Strategy Implementation Plan

Implement a question-specific MLP for **misconception detection**.

[ ] **Shared Trunk**: 
   - Input: embedding(384) 
   - Hidden 1: 768 units, ReLU, 0.3 dropout
   - Hidden 2: 384 units, ReLU, 0.3 dropout
   - Shared: 192 units, ReLU, 0.3 dropout

[ ] **Question-Specific Misconception Heads** (`nn.ModuleDict`): 
   - Q31772: 3 outputs → [Incomplete, WNB, NA]
   - Q31774: 4 outputs → [Mult, SwapDividend, FlipChange, NA]  
   - Q31778: 4 outputs → [Additive, Irrelevant, WNB, NA]
   - Q32829: 4 outputs → [Not_variable, Adding_terms, Inverse_operation, NA]
   - ... (one head per question with question-specific misconceptions + NA)

**X**: "Question: {}; Expected Answer: {}; Student's Answer: {}; Student's Explanation: {}" ->Embedding (384-dim)
**label**: Misconception | "NA"

## Integration Requirements
[ ] Follow existing `Strategy` pattern from `kaggle_map/strategies/base.py`
[ ] Maintain compatibility with existing evaluation pipeline
[ ] Use same data structures: `TrainingRow`, `TestRow`, `SubmissionRow`, `Prediction`
[ ] Use 70% train.csv for training, 15% for validation and 15% for testing

### Embedding Strategy
[x] **Formula normalization**: `utils/formula.py` (LaTeX answer cleaning, text composition)
[ ] **Tokenizer**: `utils/tokenizer.py`
[ ] **Embedding model registry**: `utils/embeddings.py` (EmbeddingModel enum with specs)


### Loss Function Design
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

## 4. Implementation Details

### 4.1 Technology Stack
[ ] **Framework**: PyTorch for neural network implementation
[x] **Embeddings**: `sentence-transformers` library (model registry in `utils/embeddings.py`)
[x] **Strategy Integration**: Inherit from existing `Strategy` base class (interface defined)
[ ] **Serialization**: 
    - Save embeddings using `np.savez` for (embedding, label); 
    - model weights + metadata as JSON/pickle

## 5. Strategy Class Implementation

### 5.1 Core Methods (from base.Strategy)
[x] **Strategy interface defined**: All abstract methods specified in `base.py`
[ ] `name` property: return `"mlp"`
[ ] `description` property: return `"Question-specific MLP for misconception detection and reasoning quality"`
[ ] `fit(train_csv_path)` → `MLPStrategy`: Train the neural network
[ ] `predict(test_data)` → `list[SubmissionRow]`: Generate predictions
[ ] `save(filepath)` and `load(filepath)`: Model persistence
[ ] `display_stats()`, `display_detailed_info()`, `demonstrate_predictions()`
