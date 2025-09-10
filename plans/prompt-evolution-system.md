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
    map_score: MAPScore  # Mean Average Precision @ 3
    failure_samples: list[FailureCase]  # 10 distinct failures

class FailureCase(TrainingRow):
    predicted: list[Prediction]

class Generation(BaseModel):
    """A complete evolution generation."""
    generation_id: GenerationID
    candidates: list[PromptCandidate]
    evaluations: list[EvaluationResult] # order by `map_score` desc
    timestamp: datetime

class EvolutionContext(BaseModel):
    current_best_prompt: CandidateID
    current_best_score: MAPScore
    parent_prompts: list[PromptCandidate] # Top 3 so far.
    failure_patterns: dict[CandidateID, list[FailureCase]]  # 10 failures for each of the top 3 candidates. Bootstrapped from @datasets/error_predictions.csv
    competition_context: str # Content read from @docs/competition.md
    next_generation_id: GenerationID
```

### Data Flow

1. **Initialization** → Load baseline prompt, evaluate performance
2. **Context Building** → Analyze failures, prepare GPT-5 context
3. **Generation** → GPT-5 creates N candidates with hypotheses
4. **Evaluation** → Benchmark each candidate on balanced sample
5. **Selection** → Choose top performers based on MAP@3
6. **Analysis** → Extract failure patterns for next generation
7. **Iteration** → Repeat with enriched context

## Module Structure

kaggle_map/
├── evolution/
│   ├── __init__.py        # Data structures (PromptCandidate, EvaluationResult, etc.) 
│   ├── generator.py       # GPT-5 prompt generation (using new Responses API)
│   ├── analysis.py        # Failure analysis and pattern extraction
│   ├── storage.py         # File I/O for prompts and results
│   └── evolve.py          # Main evolution orchestrator. __main__ entry point.
├── reranker/
│   ├── benchmark.py       # [MODIFY] Add --prompt-template parameter
│   └── prompts/
│       ├── baseline.j2    # [RENAME] from prompt.j2
│       └── generations/   # [NEW] Evolution results
│           ├── gen-00/
│           │       ├── candidate-0.json
│           │       ├── candidate-1.json
│           ├── gen-01/
│           │       ├── ...
│           └── ...

## Implementation Tasks

Keep the code generated as simple and minimal as possible. Avoid over-engineering. 

### Task 1: Data Models Foundation
**Goal:** Define core data structures with full type safety and validation

1.1. [ ] Create `kaggle_map/evolution/models.py` with all dataclasses
1.2. [ ] Use Pydantic for validation where needed (e.g., score ranges)
1.3. [ ] Add assertions for data integrity (e.g., MAP score 0-1)
1.4. [ ] Implement __str__ methods for readable logging

### Task 2: Storage Layer
**Goal:** Reliable persistence of prompts and evaluation results

2.1. [ ] Create `kaggle_map/evolution/storage.py` with path management
2.2. [ ] Implement prompt template save/load with Jinja2 compatibility
2.3. [ ] Create JSON serialization for evaluation results
2.4. [ ] Design directory structure: `prompts/generations/gen_XX/`
2.5. [ ] Add YAML metadata files for hypotheses and lineage

### Task 3: Evaluation Pipeline
**Goal:** Benchmark candidates with consistent methodology

3.1. [ ] Create `kaggle_map/evolution/evaluator.py`
3.2. [ ] Modify `benchmark.py` to accept `--prompt-template` parameter
3.3. [ ] Implement balanced sampling (10% stratified by QuestionId)
3.4. [ ] Create evaluation harness that runs benchmark for each candidate
3.5. [ ] Capture detailed metrics (MAP@3, accuracy, precision, recall, F1)
3.6. [ ] Extract top 10 failure cases per candidate
3.7. [ ] Add progress logging with time estimates
3.8. [ ] Handle evaluation failures gracefully (return zero score)

### Task 4: Failure Analysis
**Goal:** Extract actionable patterns from prediction failures

4.1. [ ] Create `kaggle_map/evolution/analysis.py`
4.2. [ ] Implement failure categorization (wrong category vs wrong misconception)
4.3. [ ] Group failures by error type (e.g., WNB vs Incomplete confusion)
4.4. [ ] Calculate failure statistics per QuestionId
4.5. [ ] Identify systematic biases in predictions
4.6. [ ] Create human-readable failure summaries
4.7. [ ] Generate failure context for GPT-5 consumption

### Task 5: GPT-5 Prompt Generator
**Goal:** Generate diverse, hypothesis-driven prompt variations

5.1. [ ] Create `kaggle_map/evolution/generator.py`
5.2. [ ] Set up OpenAI client with GPT-5 configuration
5.3. [ ] **Use OpenAI's new Responses API** (`client.responses.create()` instead of `chat.completions.create()`)
5.4. [ ] Design meta-prompt for GPT-5 that explains the task
5.5. [ ] Implement prompt generation with structured output (JSON)
5.6. [ ] Parse GPT-5 response to extract candidates and hypotheses
5.7. [ ] Validate generated prompts maintain required template variables
5.8. [ ] Log all GPT-5 interactions for debugging

### Task 6: Evolution Orchestrator
**Goal:** Coordinate the complete evolution cycle

6.1. [ ] Create `kaggle_map/evolution/evolve.py` as main entry point
6.2. [ ] Implement generation loop with stopping criteria
6.3. [ ] Track evolution history across generations
6.4. [ ] Implement parent selection for crossover
6.5. [ ] Build context enrichment between generations
6.6. [ ] Create performance tracking and reporting
6.7. [ ] Implement early stopping on convergence (<1% over 3 gens)
6.8. [ ] Add comprehensive logging of each generation
### Task 3: Evaluation Pipeline
**Goal:** Benchmark candidates with consistent methodology

[ ] Create `kaggle_map/evolution/evaluator.py`
[ ] Modify `benchmark.py` to accept `--prompt-template` parameter
[ ] Implement balanced sampling (10% stratified by QuestionId)
[ ] Create evaluation harness that runs benchmark for each candidate
[ ] Capture detailed metrics (MAP@3, accuracy, precision, recall, F1)
[ ] Extract top 10 failure cases per candidate
[ ] Add progress logging with time estimates
[ ] Handle evaluation failures gracefully (return zero score)

### Task 4: Failure Analysis
**Goal:** Extract actionable patterns from prediction failures

[ ] Create `kaggle_map/evolution/analysis.py`
[ ] Implement failure categorization (wrong category vs wrong misconception)
[ ] Group failures by error type (e.g., WNB vs Incomplete confusion)
[ ] Calculate failure statistics per QuestionId
[ ] Identify systematic biases in predictions
[ ] Create human-readable failure summaries
[ ] Generate failure context for GPT-5 consumption

### Task 5: GPT-5 Prompt Generator
**Goal:** Generate diverse, hypothesis-driven prompt variations

[ ] Create `kaggle_map/evolution/generator.py`
[ ] Set up OpenAI client with GPT-5 configuration
[ ] **Use OpenAI's new Responses API** (`client.responses.create()` instead of `chat.completions.create()`)
[ ] Design meta-prompt for GPT-5 that explains the task
[ ] Implement prompt generation with structured output (JSON)
[ ] Parse GPT-5 response to extract candidates and hypotheses
[ ] Validate generated prompts maintain required template variables
[ ] Log all GPT-5 interactions for debugging

### Task 6: Evolution Orchestrator
**Goal:** Coordinate the complete evolution cycle

[ ] Create `kaggle_map/evolution/evolve.py` as main entry point
[ ] Implement generation loop with stopping criteria
[ ] Track evolution history across generations
[ ] Implement parent selection for crossover
[ ] Build context enrichment between generations
[ ] Create performance tracking and reporting
[ ] Implement early stopping on convergence (<1% over 3 gens)
[ ] Add comprehensive logging of each generation