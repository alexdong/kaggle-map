# Prompt Evolution System Implementation Plan

## WHY: Core Architecture Principles

### Data Structure First (from datastructure.md)
> "Show me your tables, and I won't usually need your flowcharts; they'll be obvious." - Fred Brooks

The evolution system's behavior emerges from well-designed data structures that make illegal states unrepresentable. Each generation knows its lineage, each candidate carries its hypothesis, and evaluation results are immutable facts.

### Fail Fast and Observably (from debuggability.md & observability.md)
Every evolution step must be traceable. When a prompt fails to improve performance, we need to know exactly why. Logs are our time machine for understanding what GPT-5 tried and why it didn't work.

### Simplicity Over Cleverness (from pythonic.md & readability.md)
No async complexity, no premature abstractions. Sequential execution with clear data flow. Each function does one thing well.

## Data Structures & Flow

### Core Data Models

```python
from datetime import datetime
from kaggle_map.core.models import TrainingRow, Prediction

type GenerationID = int  # e.g., 0, 1, 2, ...
type CandidateID = str   # e.g., "gen_03_candidate_2"
type MAPScore = float    # 0.0 to 1.0

class PromptCandidate(BaseModel):
    """A single prompt variation with its metadata."""
    generation: GenerationID
    candidate_id: CandidateID
    prompt: str             # Full Jinja2 template
    hypothesis: str         # Why this might work better
    parent_ids: list[CandidateID]   # For tracking lineage

class EvaluationResult(BaseModel):
    """Performance metrics for a prompt candidate."""
    candidate_id: CandidateID
    map_score: MAPScore  # Mean Average Precision @ 3 (only metric we track)
    failure_samples: list[FailureCase]  # 10 diverse failures (see selection rule below)

class FailureCase(TrainingRow):
    """A test case where the model failed, inheriting all training data fields."""
    predicted: list[Prediction]  # What the model predicted

class Generation(BaseModel):
    """A complete evolution generation."""
    generation_id: GenerationID
    candidates: list[PromptCandidate]
    evaluations: list[EvaluationResult]  # Ordered by map_score desc
    timestamp: datetime

class EvolutionContext(BaseModel):
    """Context for generating next batch of prompts."""
    current_best_prompt: CandidateID
    current_best_score: MAPScore
    parent_prompts: list[PromptCandidate]  # Top 3 across ALL generations
    failure_patterns: dict[CandidateID, list[FailureCase]]  # 10 failures per top candidate
    competition_context: str  # Content from @docs/competition.md
    next_generation_id: GenerationID
```

### Failure Selection Rule
For each candidate's 10 failure samples:
1. **Priority 1**: Cases where correct answer is NOT in top 3 predictions (complete misses)
2. **Priority 2**: Diverse sampling across (QuestionId, Category, MC_Answer) combinations
3. **Priority 3**: Different error types (wrong category vs wrong misconception)

### Data Flow

1. **Initialization** → Load baseline prompt, evaluate performance, bootstrap from error_prediction.csv
2. **Context Building** → Analyze failures, prepare GPT-5 context
3. **Generation** → GPT-5 creates 7 candidates with hypotheses
4. **Evaluation** → Benchmark each candidate on 10% balanced sample
5. **Selection** → Choose top 40% (2-3 candidates) based on MAP@3
6. **Analysis** → Extract failure patterns for next generation
7. **Iteration** → Repeat with enriched context

## Module Structure

```
kaggle_map/
├── evolution/
│   ├── __init__.py        # Data structures (all models defined here)
│   ├── generator.py       # GPT-5 prompt generation (using new Responses API)
│   ├── evaluator.py       # Candidate evaluation harness
│   ├── analysis.py        # Failure analysis and pattern extraction
│   ├── storage.py         # File I/O for prompts and results
│   └── evolve.py          # Main orchestrator with __main__ entry point
├── reranker/
│   ├── benchmark.py       # [MODIFY] Add --prompt-template parameter
│   └── prompts/
│       ├── baseline.j2    # [RENAME] from prompt.j2
│       ├── gen_00_candidate_0.j2  # Generated prompts live here for easy access
│       ├── gen_00_candidate_1.j2
│       └── generations/   # Evolution metadata and results
│           ├── context.json       # Current EvolutionContext
│           ├── gen_00/
│           │   ├── generation.json     # Generation object with all candidates
│           │   ├── candidate_0.json    # PromptCandidate + EvaluationResult
│           │   └── candidate_1.json
│           └── gen_01/...
```

## Implementation Tasks

Keep the code generated as simple and minimal as possible. Avoid over-engineering.
All code must be type-annotated and use Pydantic for data models.
Avoid `try/except` unless absolutely necessary. Instead, use `assert`. 
Also, add plenty of loguru logging for observability and debuggability.

Avoid unnecessary complexity. No async, no `| None` types, no fancy patterns.

Before you generate any code, produce the failing test cases first. Keep the tests minimum. 
We DO NOT need to be 100% coverage. This is a one-man project. We just need to catch obvious mistakes.

### Task 1: Data Models Foundation
**Goal:** Define core data structures with full type safety and validation

- [x] 1.1. Create `kaggle_map/evolution/__init__.py` with all dataclasses
- [x] 1.2. Use Pydantic for validation (MAP score 0-1, generation >= 0)
- [x] 1.3. Add assertions for data integrity
- [x] 1.4. Implement __str__ methods for readable logging

### Task 2: Storage Layer
**Goal:** Reliable persistence of prompts and evaluation results

- [x] 2.1. Create `kaggle_map/evolution/storage.py` with path management. Use Pathlib.Path.
- [x] 2.2. Save .j2 templates to `prompts/` directory (alongside baseline.j2)
- [x] 2.3. Save JSON metadata to `prompts/generations/gen_XX/`
- [x] 2.5. Create helper to load/save EvolutionContext

### Task 3: Evaluation Pipeline 
**Goal:** Benchmark candidates with consistent methodology

- [x] 3.1. Modify `benchmark.py` to accept `--prompt-template` parameter
- [x] 3.2. Implement balanced sampling: 10% stratified by (QuestionId, Category, MC_Answer). Ensure at least 3-5 samples per combination if possible.
- [x] 3.3. Create `kaggle_map/evolution/evaluator.py`
- [x] 3.4. Run benchmark for each candidate, capture MAP@3 score only
- [x] 3.5. Extract 10 failure cases per candidate using selection rule

### Task 4: Failure Analysis 
**Goal:** Extract actionable patterns from prediction failures
Summarize error_prediction.csv with top 5-10 diverse patterns, grouped by (QuestionId, Category, MC_Answer).

- [x] 4.1. Create `kaggle_map/evolution/analysis.py`
- [x] 4.2. Summarize error_prediction.csv for initial bootstrap 
- [x] 4.3. Group failures by error type (wrong category vs wrong misconception)
- [x] 4.4. Generate simple failure summary for GPT-5 context.
- [x] 4.5. Implement __main__ entry point for manual runs but also the function to be called by evolve.py to enrich context.


### Task 5: GPT-5 Prompt Generator
**Goal:** Generate diverse, hypothesis-driven prompt variations

- [x] 5.1. Create `kaggle_map/evolution/generator.py`
- [x] 5.2. Set up OpenAI client with GPT-5 configuration
- [x] 5.3. Use OpenAI's new Responses API (`client.responses.create()`)
- [x] 5.4. Design meta-prompt for GPT-5 that explains the task. Emphasize:
   - Generate 7 distinct candidates
   - Emphasize diversity and novelty
   - Each with clear hypothesis
   - Use Jinja2 syntax
   - Reference failure patterns and parent prompts
   - Maintain required template variables (Question, Category, etc.)
- [x] 5.5. Use pydantic to parse response and extract 7 candidates with hypotheses
- [x] 5.6. Validate generated prompts maintain required template variables
- [x] 5.7. Implement exponential backoff retry (max 3 attempts)
- [x] 5.8. Log all GPT-5 interactions
- [x] 5.9. Add __main__ entry point for manual runs


### Task 6: Evolution Orchestrator
**Goal:** Coordinate the complete evolution cycle. If there is a score tie, prefer the earlier candidate or the one with fewer characters in the prompt.

- [x] 6.1. Create `kaggle_map/evolution/evolve.py` with `if __name__ == "__main__"`
- [x] 6.2. Implement generation loop (max 10 generations)
- [x] 6.3. Track top 5 performers across all generations
- [x] 6.4. Select top 40% from each generation as parents
- [x] 6.5. Build enriched context for next generation
- [x] 6.6. Stop if improvement < 1% over 3 consecutive generations
- [x] 6.7. Log progress to console and file


## Features to DROP for Simplicity

Based on KISS principle, these features are NOT core and should be dropped:

1. ~~Accuracy, precision, recall, F1 metrics~~ - Only track MAP@3
2. ~~YAML metadata files~~ - Use JSON only
3. ~~Visualization of error patterns~~ - Not needed for MVP
4. ~~Systematic bias identification~~ - Too complex for initial version
5. ~~Failure statistics per QuestionId~~ - Simple grouping is enough
6. ~~Time tracking for evaluations~~ - Not critical
7. ~~Complex crossover strategies~~ - Let GPT-5 handle blending

## Validation & Testing

### Module-by-Module Validation Plan

Each module can be validated independently through its `__main__` entry point or manual testing. Run validation in this sequence to build confidence from the foundation upward.

#### Stage 1: Foundation (Data Models & Storage)
**1. Data Models (`kaggle_map/evolution/__init__.py`)**
```bash
# Run unit tests for data models
uv run pytest kaggle_map/evolution/models_test.py -v

# Expected: All validators working, illegal states rejected
```

**2. Storage Layer (`kaggle_map/evolution/storage.py`)**
```bash
# Unit tests
uv run pytest kaggle_map/evolution/storage_test.py -v

# Standalone validation (NEW - with __main__)
uv run python kaggle_map/evolution/storage.py

# Expected output:
# - Tests directory structure creation
# - Validates prompt template save/load
# - Tests generation persistence
# - Verifies context storage
# - Lists existing generations
# - Cleans up test artifacts
```

#### Stage 2: Utilities
**3. Sampling (`kaggle_map/evolution/sampling.py`)**
```bash
# Unit tests
uv run pytest kaggle_map/evolution/sampling_test.py -v

# Standalone validation (NEW - with __main__)
uv run python kaggle_map/evolution/sampling.py

# Expected output:
# - Tests different sampling ratios (1%, 5%, 10%, 20%)
# - Verifies stratification preservation
# - Tests minimum samples per stratum
# - Validates reproducibility with same seed
# - Confirms different seeds produce different samples
# - Tests edge cases (0.1%, 50%, 100% samples)
```

#### Stage 3: Core Components (with __main__ entry points)
**4. Error Analysis (`kaggle_map/evolution/analysis.py`)**
```bash
# Standalone execution - analyzes error_prediction.csv
uv run python -m kaggle_map.evolution.analysis

# Expected output:
# - Failure analysis summary
# - Top 5-10 error patterns
# - Error type distribution (wrong category vs misconception)
# - Verify summary is concise and GPT-5 ready
```

**5. GPT-5 Generator (`kaggle_map/evolution/generator.py`)**
```bash
# Standalone test generation (requires OpenAI API key)
uv run python -m kaggle_map.evolution.generator

# Expected:
# - Generates 7 diverse prompt candidates
# - Each has clear hypothesis
# - Maintains required Jinja2 variables
# - Saves to prompts/ directory
# - Check logs for GPT-5 interaction details
```

**6. Evaluator (`kaggle_map/evolution/evaluator.py`)**
```bash
# Unit tests
uv run pytest kaggle_map/evolution/evaluator_test.py -v

# Standalone validation (NEW - with __main__)
uv run python kaggle_map/evolution/evaluator.py

# Expected output:
# - Tests single candidate evaluation
# - Runs micro evaluation (0.1% sample)
# - Shows evaluation results and failure samples
# - Tests MAP score parser with various formats
# - Tests batch evaluation of multiple candidates
# - Cleans up test artifacts

# Manual test with specific candidate:
# 1. Create test prompt in prompts/test_candidate.j2
# 2. Run benchmark directly:
uv run python -m kaggle_map.reranker.benchmark \
  --model gemma-3-12b-it \
  --quantization Q4_K_XL \
  --sample-ratio 0.01 \
  --prompt-template prompts/test_candidate.j2 \
  --use-stratified
```

#### Stage 4: Integration
**7. Evolution Orchestrator (`kaggle_map/evolution/evolve.py`)**
```bash
# Full system test (WARNING: Will run for hours and cost API credits)
# Start with minimal config:
uv run python -m kaggle_map.evolution.evolve \
  --max-generations 2 \
  --candidates-per-generation 3 \
  --sample-ratio 0.01

# Monitor:
# - Generation folders created in prompts/generations/
# - Context.json updates after each generation
# - Top performers tracked across generations
# - Stops if improvement < 1% over 3 generations

```

### Quick Validation Sequence
Run all modules with __main__ entry points:
```bash
# Foundation
- [x] uv run python kaggle_map/evolution/storage.py
- [x] uv run python kaggle_map/evolution/sampling.py

# Core components  
- [ ] uv run python kaggle_map/evolution/analysis.py
- [ ] uv run python kaggle_map/evolution/evaluator.py
- [ ] uv run python kaggle_map/evolution/generator.py  # Requires API key

# Integration
- [ ] uv run python kaggle_map/evolution/evolve.py  # Full system
```

### Debugging Checklist
- [ ] All required directories exist (prompts/, prompts/generations/)
- [ ] Baseline prompt renamed to baseline.j2
- [ ] error_prediction.csv has correct columns
- [ ] OpenAI API key set in environment
- [ ] Benchmark.py accepts --prompt-template parameter
- [ ] MAP@3 scores are between 0.0 and 1.0
- [ ] Generation IDs increment properly
- [ ] Parent tracking maintains lineage
- [ ] Failure cases properly extracted

### Expected Outputs
After successful validation:
```
prompts/
├── baseline.j2                    # Original prompt
├── gen_00_candidate_0.j2          # Generated variations
├── gen_00_candidate_1.j2
└── generations/
    ├── context.json               # Current evolution state
    └── gen_00/
        ├── generation.json        # Generation metadata
        ├── candidate_0.json       # Full candidate + evaluation
        └── candidate_1.json
```



## Success Criteria

- [ ] System runs 10 generations unattended
- [ ] Each generation produces 7 candidates with clear hypotheses
- [ ] Performance improves or plateaus (no regression below baseline)
- [ ] Stops automatically when improvement < 1%
- [ ] All prompts work with benchmark.py
- [ ] Complete audit trail in JSON files

## Entry Point

Run directly: `uv run kaggle_map/evolution/evolve.py`
