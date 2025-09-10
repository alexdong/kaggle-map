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

### Task 1: Data Models Foundation
**Goal:** Define core data structures with full type safety and validation

1.1. [ ] Create `kaggle_map/evolution/__init__.py` with all dataclasses
1.2. [ ] Use Pydantic for validation (MAP score 0-1, generation >= 0)
1.3. [ ] Add assertions for data integrity
1.4. [ ] Implement __str__ methods for readable logging

**Unclear items:**
- None - all models are well defined with clear imports

### Task 2: Storage Layer
**Goal:** Reliable persistence of prompts and evaluation results

2.1. [ ] Create `kaggle_map/evolution/storage.py` with path management
2.2. [ ] Save .j2 templates to `prompts/` directory (alongside baseline.j2)
2.3. [ ] Save JSON metadata to `prompts/generations/gen_XX/`
2.4. [ ] Implement atomic writes (write to temp, then rename)
2.5. [ ] Create helper to load/save EvolutionContext

**Unclear items:**
- None - storage structure is now fully specified

### Task 3: Evaluation Pipeline (SIMPLIFIED)
**Goal:** Benchmark candidates with consistent methodology

3.1. [ ] Modify `benchmark.py` to accept `--prompt-template` parameter
3.2. [ ] Create `kaggle_map/evolution/evaluator.py`
3.3. [ ] Implement balanced sampling: 10% stratified by (QuestionId, Category, MC_Answer). Ensure at least 3-5 samples per combination if possible.
3.4. [ ] Run benchmark for each candidate, capture MAP@3 score only
3.5. [ ] Extract 10 failure cases per candidate using selection rule
3.6. [ ] Add progress logging with ETA

**Unclear items:**
- Exact stratification implementation when some combinations are rare

### Task 4: Failure Analysis (SIMPLIFIED)
**Goal:** Extract actionable patterns from prediction failures

4.1. [ ] Create `kaggle_map/evolution/analysis.py`
4.2. [ ] Summarize error_prediction.csv for initial bootstrap by grouping by (QuestionId, Category, MC_Answer) and extract top 5-10 most common error patterns
4.3. [ ] Group failures by error type (wrong category vs wrong misconception)
4.4. [ ] Generate simple failure summary for GPT-5 context

**Unclear items:**
- How to summarize error_prediction.csv effectively (top N patterns?)

### Task 5: GPT-5 Prompt Generator
**Goal:** Generate diverse, hypothesis-driven prompt variations

5.1. [ ] Create `kaggle_map/evolution/generator.py`
5.2. [ ] Set up OpenAI client with GPT-5 configuration
5.3. [ ] Use OpenAI's new Responses API (`client.responses.create()`)
5.4. [ ] Design meta-prompt for GPT-5 that explains the task
5.5. [ ] Parse response to extract 7 candidates with hypotheses
5.6. [ ] Validate generated prompts maintain required template variables
5.7. [ ] Implement exponential backoff retry (max 3 attempts)
5.8. [ ] Log all GPT-5 interactions

**Unclear items:**
- Exact meta-prompt structure for GPT-5
- How to ensure diversity in generated candidates

### Task 6: Evolution Orchestrator
**Goal:** Coordinate the complete evolution cycle

6.1. [ ] Create `kaggle_map/evolution/evolve.py` with `if __name__ == "__main__"`
6.2. [ ] Implement generation loop (max 10 generations)
6.3. [ ] Track top 5 performers across all generations
6.4. [ ] Select top 40% from each generation as parents
6.5. [ ] Build enriched context for next generation
6.6. [ ] Stop if improvement < 1% over 3 consecutive generations
6.7. [ ] Log progress to console and file

**Unclear items:**
- How to handle tie scores when selecting top performers

## Features to DROP for Simplicity

Based on KISS principle, these features are NOT core and should be dropped:

1. ~~Accuracy, precision, recall, F1 metrics~~ - Only track MAP@3
2. ~~YAML metadata files~~ - Use JSON only
3. ~~Visualization of error patterns~~ - Not needed for MVP
4. ~~Systematic bias identification~~ - Too complex for initial version
5. ~~Failure statistics per QuestionId~~ - Simple grouping is enough
6. ~~Time tracking for evaluations~~ - Not critical
7. ~~Complex crossover strategies~~ - Let GPT-5 handle blending

## Success Criteria

- [ ] System runs 10 generations unattended
- [ ] Each generation produces 7 candidates with clear hypotheses
- [ ] Performance improves or plateaus (no regression below baseline)
- [ ] Stops automatically when improvement < 1%
- [ ] All prompts work with benchmark.py
- [ ] Complete audit trail in JSON files

## Entry Point

Run directly: `uv run kaggle_map/evolution/evolve.py`
