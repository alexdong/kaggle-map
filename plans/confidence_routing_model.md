# Confidence Routing Model

## Purpose

Route uncertain MLP predictions to LLM for reasoning-based correction within time constraints. Process questions by highest entropy first until time runs out.

## Why?

MLP predicts 500 questions/second but lacks reasoning. LLM predicts 0.1-0.5 questions/second but uses chain-of-thought reasoning for higher accuracy. By routing only uncertain cases to LLM, we maximize MAP@3 improvement within Kaggle's 9-hour GPU limit.

## Implementation Strategy

### Sequential Processing
1. **MLP pass**: Process all questions, calculate entropy scores
2. **Sort by entropy**: Rank questions from highest to lowest uncertainty
3. **LLM pass**: Process sorted questions until time expires
4. **Fallback**: Keep MLP predictions for unprocessed questions

*Rationale*: VRAM constraints require sequential execution. No fixed threshold since test size is unknown.

### Entropy as Routing Signal

**Calculation**: `-Σ(p_i × log(p_i))` for top-3 probabilities

**Why entropy?**: High entropy indicates prediction uncertainty. Training data validation confirms entropy correlates with potential LLM improvement.

## Component Architecture

### MLP Model (`kaggle_map/mlp/`)
- **Speed**: 0.002s per prediction
- **Output**: Softmax probabilities for top-3 predictions
- **Confidence signals**: Raw logits, softmax probabilities

### LLM Model (`kaggle_map/llm/`)
- **Speed**: 2-10s per prediction (1000x slower than MLP)
- **Models**: GGUF quantized (GPT-OSS-20B, Qwen3-30B, Gemma-3-27B)
- **Advantage**: Chain-of-thought reasoning

### MAP@3 Scoring (`kaggle_map/utils/metrics.py`)
- Position 1: 1.0 points
- Position 2: 0.5 points
- Position 3: 0.333 points

*Key insight*: Fixing complete misses (0→1.0) provides 6x more value than marginal improvements (0.33→0.5).

## Implementation Plan

### Phase 1: Entropy Validation
- [ ] Calculate entropy for all training predictions
- [ ] Analyze correlation between entropy and MAP@3 scores
- [ ] Validate entropy as routing signal

### Phase 2: Core Components
- [ ] Implement MLPProcessor for batch prediction with entropy calculation
- [ ] Build adaptive router with entropy-based sorting
- [ ] Implement LLMProcessor with time tracking and fallback
- [ ] Create RoutingSession manager for pipeline orchestration

### Phase 3: Integration
- [ ] Connect to existing MLP/LLM modules
- [ ] Sequential MLP→LLM pipeline with VRAM constraints
- [ ] Time budget management (9-hour limit)
- [ ] Submission format generation

### Phase 4: Testing & Optimization
- [ ] Test end-to-end pipeline with sample data
- [ ] Measure LLM improvement over MLP baseline
- [ ] Tune for maximum MAP@3 within time constraints

## Minimal Integration Changes

### New Files (3 total)
1. **`kaggle_map/routing/entropy.py`** - Entropy calculation and routing logic
2. **`kaggle_map/routing/pipeline.py`** - Main orchestrator
3. **`kaggle_map/routing/entropy_test.py`** - Tests

### Modified Files (2 total)
1. **`kaggle_map/mlp/main.py`**
   - Add `return_probs: bool = False` parameter to `predict_single()`
   - Return tuple `(SubmissionRow, probs)` when requested

2. **`kaggle_map/core/models.py`**
   - Add `RoutedSubmissionRow` extending `SubmissionRow` with entropy field

### Core Functions

**`entropy.py`:**
```python
def calculate_entropy(probs: torch.Tensor) -> float:
    """Calculate entropy from top-3 probabilities."""

def route_predictions(
    eval_rows: list[EvaluationRow],
    mlp_model: QuestionSpecificMLP,
    llm: Llama,
    time_budget_seconds: float
) -> list[SubmissionRow]:
    """Main routing function."""
```

**`pipeline.py`:**
```python
def run_routing_pipeline(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    time_budget_seconds: float = 32400  # 9 hours
) -> None:
    """End-to-end pipeline runner."""
```

### Data Flow
1. Load evaluation data → `list[EvaluationRow]` (existing)
2. MLP batch prediction → `list[tuple[SubmissionRow, float]]` (row + entropy)
3. Sort by entropy descending → `list[tuple[SubmissionRow, float]]`
4. Process top-N through LLM until timeout → Updated `list[SubmissionRow]`
5. Write submission CSV (existing)

### Reused Components
- `EvaluationRow`, `SubmissionRow`, `Prediction` from `core/models.py`
- `predict_batch()` from `mlp/main.py`
- `get_llm_predictions()` from `utils/gguf_model.py`
- `calculate_map_at_3()` from `utils/metrics.py`
- `load_llm_model()` from `utils/gguf_model.py`
- Existing data loading utilities

### Tests
- `test_calculate_entropy()` - Verify entropy calculation
- `test_route_predictions_time_budget()` - Verify time constraint handling
- `test_route_predictions_fallback()` - Verify LLM failure fallback
- `test_entropy_correlation()` - Validate entropy vs MAP@3 correlation