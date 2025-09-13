# Prompt Workbench Implementation Plan

## Overview

### Why We Need This
The current `evaluator.py` is designed for batch evaluation with stratified sampling - great for overall performance assessment but inefficient for prompt engineering. When refining prompts, we need:
- **Targeted testing**: Focus on specific problematic rows rather than random samples
- **Rapid iteration**: Modify prompt → evaluate → see results in seconds
- **History tracking**: Compare how prompt changes affect scores over time
- **Persistent LLM**: Keep model loaded between evaluations (no reload penalty)

### How It Differs from evaluator.py
| evaluator.py | prompt_workbench.py |
|-------------|-------------------|
| CLI-based batch processing | Web-based interactive UI |
| Stratified random sampling | User-selected specific rows |
| One-shot evaluation | Iterative refinement |
| No history tracking | SQLite database for prompt evolution |
| Loads LLM each run | Single LLM instance (fast iteration) |

### Design Principles (KISS)
- **Minimal features**: No color-coding, no exports, no fancy UI
- **Fast feedback**: Optimize for speed of iteration
- **Simple data flow**: Row IDs → DataFrame → Evaluation → Results
- **Flat structure**: Assertions over exceptions, logging over comments
- **Single purpose**: Make prompt iteration fast and trackable

## Core Requirements
- Web interface with row ID input and prompt editing
- SQLite database to track prompt evolution
- Navigation to browse prompt history
- Single LLM instance for performance
- Test-driven development approach but safe to ignore edge cases
- Avoid defensive programming in favour for simplicity
- Prefer `assert` over `try ... except`. Add logging for observability
- Ignore 'backward compatibility' concerns

---

## Task 1: Write Failing Tests
**Goal:** Create comprehensive tests that define expected behavior with minimum amount of unit tests

### Steps:
- [ ] Create `kaggle_map/llm/utils_test.py`
  - [ ] Test `evaluate_dataframe()` with mock LLM
  - [ ] Test `parse_predictions()` with various response formats
  
- [ ] Create `kaggle_map/dataloader/dataloader_test.py`
  - [ ] Test `load_rows_by_ids()` with valid IDs
  
- [ ] Create `kaggle_map/llm/prompt_workbench_test.py`
  - [ ] Test database initialization
  - [ ] Test prompt saving and retrieval
  - [ ] Test evaluation result storage
  - [ ] Test navigation between prompts

---

## Task 2: Add Data Models
**Goal:** Define `EvaluationResult` in core models

### Steps:
- [ ] Open `kaggle_map/core/models.py`
- [ ] Add `EvaluationResult` dataclass
  - [ ] Include all required fields (row_id, question_id, mc_answer, etc.)
  - [ ] Ensure compatibility with existing `Prediction` class
  - [ ] Add validation in `__post_init__` if needed
  
- [ ] Update imports in affected files

---

## Task 3: Add DataLoader Function
**Goal:** Implement `load_rows_by_ids()` in dataloader

### Steps:
- [ ] Implement `load_rows_by_ids(data_path: Path, row_ids: list[int]) -> pd.DataFrame`
  - [ ] Load full dataset
  - [ ] Filter by row_id column
  - [ ] Return filtered DataFrame
  - [ ] Log debug info about loaded rows
  
- [ ] Ignore missing row IDs gracefully
- [ ] Add assertion for non-empty result


---

## Task 4: Extract Evaluation Logic
**Goal:** Create `utils.py` with shared evaluation functions

### Steps:
- [ ] Create `kaggle_map/llm/utils.py`
- [ ] Move from evaluator.py:
  - [ ] `build_prediction_prompt()`
  - [ ] `parse_predictions()`
  
- [ ] Implement `evaluate_dataframe()`:
  - [ ] Accept pre-loaded LLM instance
  - [ ] Accept DataFrame, template
  - [ ] Iterate through rows
  - [ ] Generate predictions
  - [ ] Calculate MAP@3 scores
  - [ ] Return list of EvaluationResult and average score

---

## Task 5: Refactor evaluator.py
**Goal:** Update evaluator.py to use new utils

### Steps:
- [ ] Import functions from utils.py
- [ ] Replace local implementations with utils calls
- [ ] Ensure CLI functionality unchanged
- [ ] Update display_evaluation_details to accept EvaluationResult list
- [ ] Test CLI still works with `--sample-ratio 0.01`

---

## Task 6: Create Basic Web Interface
**Goal:** Implement minimal prompt_workbench.py

### Steps:
- [ ] Create `kaggle_map/llm/prompt_workbench.py`
- [ ] Set up FastHTML application
- [ ] Create singleton LLM instance at startup
  - [ ] Load model once
  - [ ] Store in global variable
  
- [ ] Implement main page:
  - [ ] Row IDs textarea
  - [ ] Prompt template textarea (load predict.j2 by default)
  - [ ] Evaluate button
  - [ ] Save Template button
  
- [ ] Implement `/evaluate` endpoint:
  - [ ] Parse row IDs (one per line)
  - [ ] Save prompt to database (get ID)
  - [ ] Load rows using load_rows_by_ids()
  - [ ] Run evaluate_dataframe() with singleton LLM
  - [ ] Update database with results
  - [ ] Return HTML table
  
- [ ] Implement `/save-template` endpoint:
  - [ ] Write prompt to predict.j2

### Edge Cases:
- Invalid row IDs
- Empty inputs
- File write permissions

---

## Task 7: Add SQLite Database
**Goal:** Implement prompt history tracking

### Steps:
- [ ] Add database initialization in prompt_workbench.py
  - [ ] Create `kaggle_map/llm/prompts.db`
  - [ ] Define schema:
    ```sql
    CREATE TABLE IF NOT EXISTS prompt_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        prompt TEXT NOT NULL,
        row_ids TEXT NOT NULL,
        evaluation_results TEXT,  -- NULL until evaluation completes
        score REAL  -- NULL until evaluation completes
    );
    ```
  
- [ ] Implement database functions:
  - [ ] `save_prompt()` - Insert new record, return ID
  - [ ] `update_evaluation_results()` - Update record with results
  - [ ] `get_prompt_by_id()` - Retrieve specific prompt
  - [ ] `get_all_prompts()` - List all prompts with scores
  
- [ ] Use context managers for connections

---

## Task 8: Add Navigation Features
**Goal:** Browse through prompt history

### Steps:
- [ ] Add navigation UI elements:
  - [ ] Current prompt ID display
  - [ ] Previous button (<)
  - [ ] Next button (>)
  - [ ] Score display for current prompt
  
- [ ] Implement `/navigate` endpoint:
  - [ ] Accept direction (prev/next) and current_id
  - [ ] Query database for adjacent prompt
  - [ ] Return prompt text and results
  - [ ] Update UI with historical data
  - [ ] Be aware of First/last prompt boundaries
  
- [ ] Add prompt ID to URL for bookmarking
