# Project Overview

The **MAP – Charting Student Math Misunderstandings** competition is a featured code competition on Kaggle.

The mission is to build NLP models that predict the affinity between student open-ended responses and misconceptions. The goal is to detect and classify math misconceptions from real student explanations, helping teachers give faster, more targeted feedback and unlock new insights into how students learn.

When students answer diagnostic questions on Eedi, they sometimes explain their answer. These explanations often reveal misconceptions, but tagging explanations manually is time-consuming. The competition asks participants to create models that can automatically suggest likely misconceptions so teachers can address errors more effectively.

## Dataset and Tasks

### Data Description

The dataset consists of diagnostic questions from Eedi. After choosing a multiple-choice answer, students may provide a written explanation. Each row in [`datasets/train.csv`](datasets/train.csv) represents one student response and contains:

| Column | Description |
|--------|-------------|
| QuestionId | Unique identifier for the question |
| QuestionText | Text of the question. OCR has been applied to images, so the text is available without processing the images |
| MC_Answer | The multiple-choice answer selected by the student |
| StudentExplanation | Free-text explanation given by the student |
| Category (train only) | Relationship between the selected answer and explanation. Possible values are `True_Correct`, `True_Incorrect`, `True_Misconception` (correct answer, but explanation shows a misconception), and their `False_*` counterparts |
| Misconception (train only) | Specific math misconception tag (e.g., "Incomplete fraction simplification"); `NA` if no misconception applies |

In the [`datasets/`](datasets/) folder, you will find:
- [`train.csv`](datasets/train.csv): Training data with labels
- [`33474_full_train.csv`](datasets/33474_full_train.csv): All training data for QuestionId 33474, which is the hardest question.
- [`33474_tiny_train.csv`](datasets/33474_tiny_train.csv): 17 Questions of different difficulty level from the above. Useful for quick experiments and vibe checks.
- [`test.csv`](datasets/test.csv): Sample test data without labels
- [`sample_submission.csv`](datasets/sample_submission.csv): Example submission file

### Tasks

Models must perform three sub-tasks:

1. **Determine if the multiple-choice answer is correct**
   - Decide whether the student's chosen answer is right or wrong.
   - This is encoded in the `Category` label as `True_*` (correct) or `False_*` (incorrect).

2. **Determine whether the explanation reveals a misconception**
   - Some explanations show misunderstandings even when the answer is correct.
   - Example: Student gets the right answer but explains it using incorrect reasoning.
   - This influences whether the `Category` is labeled `*_Correct`, `*_Incorrect`, or `*_Misconception`.

3. **Identify the specific misconception tag**
   - When a misconception exists, identify the specific type.
   - Examples: "Incomplete simplification", "Confused operation", "Wrong formula".
   - There is exactly one misconception label per explanation when applicable.

### Scoring

Submissions are evaluated using **Mean Average Precision @ 3 (MAP@3)**. For each observation, participants may submit up to three predicted `Category:Misconception` pairs ranked by confidence. Given 3 test samples with ground truth and predictions:

```
Sample 1: Ground truth = "True_Misconception:Incomplete"
  Predictions: ["True_Misconception:Incomplete", "True_Incorrect:Other", "False_Correct:NA"]
  AP = 1.0 (correct at position 1)

Sample 2: Ground truth = "False_Correct:NA"
  Predictions: ["True_Incorrect:Wrong", "False_Correct:NA", "True_Misconception:Other"]
  AP = 0.5 (correct at position 2)

Sample 3: Ground truth = "True_Incorrect:Calculation"
  Predictions: ["False_Correct:NA", "True_Misconception:Other", "False_Incorrect:Wrong"]
  AP = 0.0 (correct answer not in top 3)

MAP@3 = (1.0 + 0.5 + 0.0) / 3 = 0.5
```

Please refer to [`docs/competition.md`](docs/competition.md) for more details.

## High-Level Architecture

We use a two-stage inference pipeline for each row.

1) Embedding + MLP: Convert the student's explanation into a text embedding and classify with an MLP, producing calibrated probabilities via softmax.

2) LLM augmentation (selective): For uncertain cases, call a local LLM to provide an additional classification. Uncertainty is measured by prediction entropy—higher entropy indicates lower certainty. We prioritize rows from most to least uncertain to control cost.

Latency budget: the MLP runs in ~20 ms per sample, while the local LLM takes ~10–20 s per sample. We therefore limit LLM usage and merge MLP and LLM outputs to produce the final submission.

### Core Directories

**kaggle_map/core/** – Domain models and normalization
- Pydantic data models for rows, labels, and predictions
- Category/label normalization and comparison utilities
- CSV ingestion helpers and deterministic seeds

**kaggle_map/dataloader/** – Dataset sampling utilities
- Focused dataset samplers (e.g., hardest question subsets)
- Lightweight helpers for fast experiments

**kaggle_map/embeddings/** – Text embeddings
- Wrappers for multiple embedding backends and sizes
- Canonical text construction from domain rows
- Sampling strategies for balanced training

**kaggle_map/mlp/** – MLP modeling pipeline
- Dataset preparation, label encoding, trainer, and loss
- CLI entrypoints (`python -m kaggle_map.mlp ...`) for fit/eval/predict
- Model persistence and simple deployment interface

**kaggle_map/llm/** – LLM-assisted components
- Robust parser with typo tolerance and fuzzy matching
- Prompt storage and evaluation helpers
- Structured evaluator for LLM-based predictions

**kaggle_map/optimise/** – Hyperparameter search and analysis
- Optuna-based studies for MLP and LLM settings
- Study listing, dashboard launcher, and analyses
- OpenEvolve-based LLM prompt optimization

**kaggle_map/utils/** – Cross-cutting utilities
- Logger configuration via `loguru`, metrics, device, CLI helpers
- GGUF model helpers and file utilities

**datasets/** – Input CSVs for training/evaluation
- Original, focused, and tiny subsets for quick iteration
**models/** – Saved model artifacts
**logs/** – Module-scoped structured logs


## Tech Stack

- **Language & Packaging**: Python 3.12+ with strict type modeling via Pydantic; project tooling managed by `uv` and standard `pyproject.toml` metadata. To run a module directly, use `uv run -m {module}`. 
- **Modeling & Training**: PyTorch MLP classifiers complemented by scikit-learn utilities, calibrated softmax outputs.
- **Embeddings & LLMs**: Sentence-transformers and Hugging Face `transformers` backends, local GGUF models (e.g., Qwen, Gemma) served through `llama-cpp-python`, plus Hugging Face Hub integration for artifact sync.
- **Data & Experimentation**: Pandas/numpy pipelines, Kaggle API ingestion helpers, and Optuna with `optuna-dashboard` for hyperparameter search and study visualization.
- **Interfaces & Observability**: CLI surfaces built with Click and Prompt Toolkit, templating/visualization via python-fasthtml, Textual, Jinja2, and logging/diagnostics handled by Loguru, better-exceptions, psutil, and platformdirs.
- **Developer Tooling**: Makefile targets wrap `uv run` workflows; Ruff, Ty, and Pyrefly enforce linting and static analysis; pytest, pytest-asyncio, and Hypothesis back the testing strategy. When you want to run `python` directly, use `uv run` instead to ensure the environment is correctly set up.


## Development Principles

These opinionated principles contradict much of the common wisdom. Take the time to read and understand the philosophy behind them. Follow them religiously.

### Data Before Code

  - Define types before writing code. Types are contracts between functions. They should be as tight as possible to avoid ambiguity. For example, prefer `list[str]` over `list`. Use `Pydantic` types for complex data structures to ensure validation and serialization. Think of it as test-driven development for types. Refer to [`.claude/agents/datastructure.md`](.claude/agents/datastructure.md) for more details.
  - `make dev` will run a new generation of formatters and type checkers, including `ruff`, `ty`, and `pyrefly`. Fix any issues before proceeding, even if the issues aren't in code you changed. If you disagree with a rule, discuss it with me and share your reasoning, but don't ignore or disable the rule. 
  - Leverage Python 3.13+ features for type annotations. Use algebraic data types for composite data types. Types are essentially concepts, so the fewer types we have, the better. Avoid deprecated constructs. Avoid `from __future__ import annotations`. We do not care about backward compatibility. 

### Fail Early and Noisily

  - Use `assert` statements liberally. Always include a detailed message. Make every function’s contract explicit and enforceable. Refrain from try/except blocks, union-with-None types (`| None`), or guarded early returns (`if ...: return`) unless necessary. Defensive programming is useful when interacting with external inputs, but once we are inside our code, we should aim to fail fast and loudly. Refer to [`.claude/agents/debuggability.md`](.claude/agents/debuggability.md) for more details.
  - Failing loudly makes the codebase easier to maintain in the long run by exposing internal flows and state transitions. Use `loguru` to both document and observe the code's state. Refer to [`kaggle_map/utils/logger_config.py`](kaggle_map/utils/logger_config.py) for detailed configuration and [`.claude/agents/observability.md`](.claude/agents/observability.md) for more details.
  - Add a `__main__` block at the end of each file to provide a quick entry point to key features of the module for testing and debugging purposes. Make sure the log level is set to `DEBUG`. Use the `click` package to define and organize command-line interfaces when necessary. Always include example usages in the docstring. Make sure `uv run -m {module} -h` works and is mentioned in the docstring. Use `rich` to pretty-print complex data structures and tables, but be conservative with its usage. 
 - Be mindful of the comments you leave behind. Unless they are absolutely necessary, drop them. If they can be replaced by `logging` messages, convert them. Remember that comments should explain the "why" behind complex logic, not the "what" or "how". 

### Shorter Code is Better Code

  - After you've made code changes and both `make dev` and `make test` pass, don't stop there. Instead, start rewriting the code until you can’t further improve readability. Often the third or fourth iteration gives the best code.
  - Put on a technical writer's hat and proofread code and log messages carefully. Apply Strunk & White principles to the code. Names should be well-thought-out and fit the problem domain like a glove. Use whitespace to group related code blocks together. Refer to [`.claude/agents/technicalwriter.md`](.claude/agents/technicalwriter.md) for more details.
  - Functions should be short and focused. `ruff` calculates the McCabe complexity for each function. Use that metric to guide your work. Refer to [`.claude/agents/readability.md`](.claude/agents/readability.md) for more details.

### Tests for Design and Leverage, Not Coverage

  - Before you write or change any code, write tests first. Use `uv run pytest`. Use the test code to guide the design of the implementation code. Use the test code to explore, design, and define the optimal data structures and functions for the job. Use `hypothesis` to explicitly define the properties of the code.
  - The test code should be clear, concise and unambiguous. Test function names should succinctly describe the intent. Hold them at the same high standard as the implementation code. Follow the Don't-Repeat-Yourself principle. Use `pytest.fixture` to reuse test data. Use `pytest.mark.parametrize` to "tabulate" the test data. Use real data to further improve the readability of the tests. Minimize visual noise. Use plain `assert` and functions instead of `TestCase`. Avoid mocking unless absolutely necessary.
  - Tests should be fast, independent and deterministic. Make sure all tests finish within 0.1 seconds. Be ruthless about removing tests that don't add value. Code coverage is a misleading metric. Aim for meaningful tests that validate behavior. Refer to [`.claude/agents/test.md`](.claude/agents/test.md) for more details.
