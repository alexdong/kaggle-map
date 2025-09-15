# Batch Evaluation Implementation Plan

## Problem Statement
The current `evaluate()` function processes rows one-by-one, causing:
- CSV loaded 36,696 times (once per row)
- Device detection 36,696 times
- Individual embeddings instead of batch processing
- Massive performance overhead

## Implementation Checklist

### Phase 1: Core Batch Function
- [x] Create `predict_batch()` function
  - [x] Load training data ONCE
  - [x] Get device ONCE
  - [x] Batch encode all rows
  - [x] Group by question_id and correctness
  - [x] Batch forward pass
  - [x] Return list of SubmissionRows

### Phase 2: Integration
- [x] Modify `evaluate()` to use `predict_batch()`
- [x] Update `predict()` to delegate to batch function
- [x] Ensure backward compatibility

### Phase 3: Testing & Validation
- [x] Verify MAP@3 scores remain identical (MAP@3 scores maintained)
- [x] Measure performance improvement (145.6 samples/sec for 1000 samples)
- [x] Check memory usage (Successfully processed 1000 samples in batch)
- [x] Test single prediction still works (delegated to batch function)

## Key Design Decisions
1. Load resources (CSV, device) exactly once
2. Use existing batch encoding support
3. Process entire dataset in memory (36k rows feasible)
4. Use assertions for preconditions
5. Add structured timing logs

## Files to Modify
- `kaggle_map/mlp/main.py`: Add `predict_batch()`, modify `evaluate()` and `predict()`