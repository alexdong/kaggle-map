# Hyperparameter Metadata Refactor Plan

## Objectives
1. Keep validation constraints on `MLPTrainingConfig` using `Annotated[...]` while deriving Optuna search metadata programmatically.
2. Use a single helper (`derive_optuna_metadata` + `attach_optuna_metadata`) to synchronise validation bounds with tuning distributions and eliminate hand-written `json_schema_extra` blocks.
3. Feed the derived descriptors into the sampling stack (`sample_hyperparameters`, CLI selectors, Optuna objective) so every consumer shares the same configuration contract.

## Data Structures & Flow
1. **MLPTrainingConfig (Pydantic BaseModel)**: retains tight validation bounds only; no embedded Optuna hints.
2. **optuna_config helpers**: `derive_optuna_metadata` inspects `model_fields` metadata and returns a mapping, while `attach_optuna_metadata` persists that mapping for downstream consumers that still expect `json_schema_extra`.
3. **HyperparamDescriptor**: materialised view (name, defaults, Optuna payload, enum weights) backing `TunableParameters`. Acts as the bridge between model metadata and the sampler.
4. **sample_hyperparameters**: builds a base dict from `MLPTrainingConfig`, applies include/exclude selectors, and delegates to each descriptor for trial suggestions. Selectors operate on immutable tuples to avoid shared state.
5. **CLI include/exclude parsing**: normalises comma-delimited strings into `tuple[TunableParameters]`, enforces mutual exclusivity, and validates names eagerly.
6. **Validation Flow**: helper asserts missing bounds, log-scale misuse, or weight-length mismatches before any Optuna trial executes, ensuring we fail loudly during configuration.

> Weighted categoricals rely on descriptor-level handling (e.g., alias sampling or duplication) rather than ad-hoc Optuna calls. We keep that logic local to the descriptor implementation so the CLI and trial objective remain agnostic.

## Modules To Update
- [x] `kaggle_map/core/models.py`: ensure config fields keep accurate validation bounds without `json_schema_extra`.
- [x] `kaggle_map/optimise/optuna_config.py`: house metadata derivation utilities (new module).
- [x] `kaggle_map/optimise/optuna_config_test.py`: regression tests for numeric bounds, enum weights, and failure cases.
- [ ] `kaggle_map/optimise/hyperparameters.py`: replace hard-coded samplers with descriptor-driven logic and selector support.
- [ ] `kaggle_map/optimise/tune.py`: swap `_FIXED_CASTERS` for include/exclude parsing built on shared helpers.
- [ ] `kaggle_map/mlp/main.py` & friends: wire new sampling results through fit/evaluate once descriptors are in place (no behaviour change expected, but signature adjustments likely).

## Task 0 – Test-First Scaffolding
- [x] (0.1) Draft targeted tests for metadata extraction, selector handling, weighted enums, and log-scale enforcement (`optuna_config_test.py`).
- [x] (0.2) Ensure tests cover malformed inputs (missing bounds, wrong weight lengths) with assertive failures.
- [ ] (0.3) Add failing tests for `sample_hyperparameters` covering include-only, exclude-only, and default-all scenarios.
- [ ] (0.4) Add failing tests for CLI selector parsing, including invalid names, duplicates, and simultaneous include+exclude usage.
- [ ] (0.5) Plug new tests into `uv run pytest` (full suite) once implementation lands.

## Task 1 – Metadata Derivation Integration
- [x] (1.1) Implement helper to walk `MLPTrainingConfig` metadata and return Optuna payloads (done via `derive_optuna_metadata`).
- [x] (1.2) Provide `attach_optuna_metadata` to slot the payload into `json_schema_extra` when legacy consumers require it.

## Task 2 – Build HyperparamDescriptor Extraction
- [ ] (2.1) Define `HyperparamDescriptor` dataclass capturing field name, type, default, Optuna sampler info, and optional weights.
- [ ] (2.2) Implement helper that iterates derived metadata to instantiate descriptors with explicit failure states for malformed payloads.
- [ ] (2.3) Provide lookup utilities: `get_descriptor(TunableParameters)`, `iter_descriptors(tunable_only=True)`.

## Task 3 – Rewrite Hyperparameter Sampling & Parsing
- [ ] (3.1) Build `TunableParameters` dynamically from descriptor keys to keep outward identifiers in sync.
- [ ] (3.2) Implement `_normalise_parameters(include/exclude)` accommodating both strings and enum members, asserting unknowns early.
- [ ] (3.3) Update `sample_hyperparameters` to consume descriptors, respect selectors, and fall back to config defaults.
- [ ] (3.4) Implement `normalise_cli_selector` for Click parsing with precise validation errors.

## Task 4 – Update CLI Integration
- [ ] (4.1) Replace `_FIXED_CASTERS` and manual parsing in `tune.py` with the new include/exclude helper.
- [ ] (4.2) Ensure Click options expose `--include`/`--exclude` usage text, showing available `TunableParameters`.
- [ ] (4.3) Maintain objective/run_search flow while threading selector results through Optuna trials.

## Task 5 – Testing & Validation
- [ ] (5.1) Iterate on failing tests from Tasks 0–4 until they pass; add regression cases for weighted categorical sampling.
- [ ] (5.3) Run `make dev` / `make test`, documenting follow-up risks if any checks fail.

## Task 6 – Documentation & Cleanup
- [ ] (6.1) Update developer docs (e.g., `docs/competition.md`, module docstrings) once CLI changes land.
- [ ] (6.2) Review final diffs for adherence to project principles (fail loudly, minimal comments, precise naming).

## Recent Embedding Updates (Sept 2025)
- `gemma.py` gained stricter assertions around batch vs. individual encoding, large-batch smoke tests, and richer logging (commit 4b49a79).
- `embedding_cache.py` and its tests now enforce explicit casting to `np.int32`, validate metadata round-trips, and tidy formatting (commit ad18539).
- `embeddings/__init__.py` exposes updated dimension constants alongside bolstered tests for Qwen/Gemma consistency (commit fabdbaa).
- Redundant negative-path tests were dropped from `embeddings_test.py` as part of the clean-up (commit acf6f74).
- Prediction parsing (`Prediction.from_string`) and Optuna fixed-parameter wiring were recently tightened (commit 74060ef), affecting downstream consumers of tuned models.
