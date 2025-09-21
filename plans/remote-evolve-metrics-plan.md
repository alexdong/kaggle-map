# Remote OpenEvolve Metrics Integration

## Task 1 – Establish Requirements and Interfaces
[ ] Re-read `.claude/agents/*.md` with emphasis on reasoning about *why* (design intent, observability, testing) and document any non-negotiable constraints that influence the implementation.
[ ] Trace current prompt evaluation flow (`kaggle_map.optimise.evolve.evaluate`) to confirm how metrics feed into OpenEvolve and where they surface in program metadata.
[ ] Inspect `kaggle_map.llm.api` to catalogue available endpoints, request contracts, and identify the minimal changes needed to expose MAP@3 plus latency for remote callers.

## Task 2 – Define Remote Evaluation Data Structures
[ ] Introduce dedicated Pydantic models representing the remote API response (e.g., `RemoteEvaluationMetrics`) capturing MAP@3 and response time; ensure these models live where both API and optimiser can reuse them.
[ ] Extend `PromptEvolutionConfig` (or companion types) to carry the remote endpoint URL plus optional auth headers to keep configuration explicit and flexible.
[ ] Update enforcement logic (model validators) so invalid URLs or missing metrics from responses fail fast with descriptive assertions.

## Task 3 – Implement Remote Evaluation Path in OpenEvolve Helper
[ ] Replace the direct `evaluate_with_llm` invocation with an HTTP POST against `API_URL`, submitting the candidate prompt and any sampling parameters, and parse the structured metrics from the response.
[ ] Capture MAP@3 and response latency from the payload, ensuring both values are added to the metrics dict returned to OpenEvolve (e.g., `{"combined_score": map_at_3, "map@3": map_at_3, "response_time": seconds}`) and logged for observability.
[ ] Handle network failures, non-2xx statuses, or malformed JSON by asserting loudly (per Fail Fast), optionally surfacing the last good prompt file for debugging before aborting evolution.

## Task 4 – Propagate Metrics and Persist Best Prompt
[ ] Ensure `_persist_best_prompt` (or equivalent) continues to write the winning template while also recording the associated metrics (MAP@3 and response time) into any saved artefacts or log summaries.
[ ] When remote metrics indicate improvement, update any local tracking (e.g., best prompt path, metadata files) so downstream tooling can inspect latency trends.
[ ] Remove vestigial local-evaluation helpers to keep the codebase lean and avoid divergence between local and remote scoring behaviour.

## Task 5 – Update Tests and Documentation
[ ] Add or amend unit tests using HTTP mocking (e.g., `pytest monkeypatch`) to verify that JSON responses with MAP@3 and latency drive the optimiser’s metric handling correctly.
[ ] Provide regression tests (or smoke tests) ensuring failure paths (HTTP errors, missing fields) raise assertions per the project philosophy.
[ ] Refresh developer documentation: note the new API contract in `docs` or module docstrings and include an example `curl` snippet showing the MAP@3/latency response.
