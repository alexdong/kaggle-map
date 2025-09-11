"""Storage layer for prompt evolution system."""

import json
from pathlib import Path

from loguru import logger

from kaggle_map.evolution import (
    MAX_DISPLAY_GENERATIONS,
    MIN_GENERATION_DIR_PARTS,
    EvolutionContext,
    Generation,
    GenerationID,
    PromptCandidate,
)


class Storage:
    """Handles persistence of prompts and evaluation results."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.cwd()

    def get_prompt_template_path(self, candidate_id: str) -> Path:
        assert "gen_" in candidate_id, f"Invalid candidate ID format (missing 'gen_'): {candidate_id}"

        return self.base_dir / "reranker" / "prompts" / f"{candidate_id}.j2"

    def get_generation_dir(self, generation_id: GenerationID) -> Path:
        assert generation_id >= 0, f"Generation ID must be non-negative, got {generation_id}"

        return self.base_dir / "reranker" / "prompts" / "generations" / f"gen_{generation_id:02d}"

    def get_context_path(self) -> Path:
        return self.base_dir / "reranker" / "prompts" / "generations" / "context.json"

    def save_prompt_template(self, candidate: PromptCandidate) -> None:
        assert candidate.prompt, f"Cannot save empty prompt for {candidate.candidate_id}"

        path = self.get_prompt_template_path(candidate.candidate_id)

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            logger.warning(f"Overwriting existing template: {candidate.candidate_id}")

        logger.info(f"Saving prompt template: {candidate.candidate_id} ({len(candidate.prompt)} chars)")

        path.write_text(candidate.prompt)

    def load_prompt_template(self, candidate_id: str) -> str:
        path = self.get_prompt_template_path(candidate_id)
        assert path.exists(), f"Template file not found at {path} for candidate {candidate_id}"
        return path.read_text()

    def save_generation(self, generation: Generation) -> None:
        assert generation.candidates, f"Cannot save generation {generation.generation_id} with no candidates"

        gen_dir = self.get_generation_dir(generation.generation_id)

        if not gen_dir.exists():
            gen_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.warning(f"Generation directory already exists, will overwrite: {gen_dir}")

        gen_path = gen_dir / "generation.json"
        logger.info(
            f"Saving generation {generation.generation_id}: "
            f"{len(generation.candidates)} candidates, "
            f"{len(generation.evaluations)} evaluations"
        )

        gen_data = generation.model_dump(mode="json")
        json_content = json.dumps(gen_data, indent=2)
        gen_path.write_text(json_content)

        saved_count = 0
        for candidate in generation.candidates:
            candidate_path = gen_dir / f"{candidate.candidate_id}.json"

            evaluation = next(
                (e for e in generation.evaluations if e.candidate_id == candidate.candidate_id),
                None,
            )

            if not evaluation:
                logger.warning(f"No evaluation found for candidate {candidate.candidate_id}")

            candidate_data = {
                "candidate": candidate.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json") if evaluation else None,
            }

            json_content = json.dumps(candidate_data, indent=2)
            candidate_path.write_text(json_content)

            saved_count += 1

        logger.info(f"Saved {saved_count}/{len(generation.candidates)} candidate files")

    def load_generation(self, generation_id: GenerationID) -> Generation:
        assert generation_id >= 0, f"Generation ID must be non-negative, got: {generation_id}"

        gen_dir = self.get_generation_dir(generation_id)
        gen_path = gen_dir / "generation.json"

        assert gen_dir.exists(), f"Generation directory not found: {gen_dir}"
        assert gen_path.exists(), f"Generation file not found: {gen_path}"

        json_content = gen_path.read_text()
        gen_data = json.loads(json_content)
        generation = Generation.model_validate(gen_data)

        assert generation.generation_id == generation_id, (
            f"ID mismatch: expected {generation_id}, got {generation.generation_id}"
        )

        logger.info(
            f"Loaded generation {generation_id}: "
            f"{len(generation.candidates)} candidates, {len(generation.evaluations)} evaluations"
        )

        if generation.evaluations:
            best = generation.evaluations[0]
            logger.info(f"  Best performer: {best.candidate_id} with MAP@3={best.map_score:.4f}")

        return generation

    def save_context(self, context: EvolutionContext) -> None:

        path = self.get_context_path()

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving evolution context for generation {context.next_generation_id}")

        context_data = context.model_dump(mode="json")
        json_content = json.dumps(context_data, indent=2)
        path.write_text(json_content)

    def load_context(self) -> EvolutionContext:
        path = self.get_context_path()
        assert path.exists(), f"Context file not found at {path} - may need to bootstrap"

        json_content = path.read_text()
        context_data = json.loads(json_content)
        context = EvolutionContext.model_validate(context_data)

        logger.info(f"Loaded context for generation {context.next_generation_id}")
        logger.info(f"  Current best: {context.current_best_prompt} (MAP@3: {context.current_best_score:.4f})")
        logger.info(f"  Parents: {len(context.parent_prompts)}, Failure patterns: {len(context.failure_patterns)}")

        return context

    def _validate_generation_dir(self, gen_dir: Path) -> GenerationID | None:
        """Validate and extract generation ID from directory."""
        parts = gen_dir.name.split("_")
        if len(parts) < MIN_GENERATION_DIR_PARTS:
            logger.warning(f"  Invalid generation directory format: '{gen_dir.name}'")
            return None

        gen_id_str = parts[1]
        if not gen_id_str.isdigit():
            logger.warning(f"  Invalid generation ID (not a number): '{gen_dir.name}'")
            return None

        gen_id = int(gen_id_str)
        if gen_id < 0:
            logger.warning(f"  Invalid generation ID (negative): {gen_id} in '{gen_dir.name}'")
            return None

        gen_file = gen_dir / "generation.json"
        if not gen_file.exists():
            logger.warning(f"  Generation {gen_id}: missing generation.json")
            return None

        return gen_id

    def list_generations(self) -> list[GenerationID]:
        generations_dir = self.base_dir / "reranker" / "prompts" / "generations"

        if not generations_dir.exists():
            logger.info("No generations directory found - this may be a fresh start")
            return []

        gen_dirs = [d for d in generations_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")]

        generation_ids = []
        invalid_count = 0

        for gen_dir in gen_dirs:
            gen_id = self._validate_generation_dir(gen_dir)
            if gen_id is not None:
                generation_ids.append(gen_id)
            else:
                invalid_count += 1

        generation_ids.sort()

        display_ids = generation_ids[:MAX_DISPLAY_GENERATIONS]
        ellipsis = "..." if len(generation_ids) > MAX_DISPLAY_GENERATIONS else ""
        logger.info(f"Found {len(generation_ids)} valid generations: {display_ids}{ellipsis}")

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid directories")

        return generation_ids


if __name__ == "__main__":
    """Standalone validation of storage operations."""
    import sys
    from datetime import datetime

    from kaggle_map.evolution import TEST_MAP_SCORE, EvaluationResult, Generation, PromptCandidate

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info("=== Storage Module Validation ===")
    storage = Storage()

    logger.info("\n1. Testing directory structure:")
    prompts_dir = storage.base_dir / "reranker" / "prompts"
    generations_dir = prompts_dir / "generations"
    logger.info(f"  Prompts directory: {prompts_dir} (exists: {prompts_dir.exists()})")
    logger.info(f"  Generations directory: {generations_dir} (exists: {generations_dir.exists()})")

    logger.info("\n2. Testing prompt template operations:")
    test_candidate = PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_test",
        prompt="Test prompt with {{question_text}}, {{category}}, {{mc_answer}}, {{student_explanation}}",
        hypothesis="Test hypothesis for validation",
        parent_ids=[],
    )

    storage.save_prompt_template(test_candidate)
    template_path = storage.get_prompt_template_path(test_candidate.candidate_id)
    logger.info(f"  Saved template to: {template_path}")

    loaded_prompt = storage.load_prompt_template(test_candidate.candidate_id)
    logger.info(f"  Loaded template: {len(loaded_prompt)} chars")
    assert loaded_prompt == test_candidate.prompt, "Template round-trip failed"
    logger.info("  ✓ Template round-trip successful")

    logger.info("\n3. Testing generation storage:")
    test_evaluation = EvaluationResult(
        candidate_id=test_candidate.candidate_id, map_score=TEST_MAP_SCORE, failure_samples=[]
    )
    test_generation = Generation(
        generation_id=999,  # Use high number to avoid conflicts
        candidates=[test_candidate],
        evaluations=[test_evaluation],
        timestamp=datetime.now(),
    )

    storage.save_generation(test_generation)
    gen_dir = storage.get_generation_dir(999)
    logger.info(f"  Saved generation to: {gen_dir}")

    loaded_gen = storage.load_generation(999)
    logger.info(f"  Loaded generation: {loaded_gen}")
    assert len(loaded_gen.candidates) == 1, "Generation round-trip failed"
    logger.info("  ✓ Generation round-trip successful")

    logger.info("\n4. Testing context storage:")
    from kaggle_map.evolution import EvolutionContext

    test_context = EvolutionContext(
        current_best_prompt="gen_00_candidate_test",
        current_best_score=TEST_MAP_SCORE,
        parent_prompts=[test_candidate],
        failure_patterns={},
        competition_context="Test competition context for validation",
        next_generation_id=1,
    )

    storage.save_context(test_context)
    context_path = storage.get_context_path()
    logger.info(f"  Saved context to: {context_path}")

    loaded_context = storage.load_context()
    logger.info(f"  Loaded context: {loaded_context}")
    assert loaded_context.current_best_score == TEST_MAP_SCORE, "Context round-trip failed"
    logger.info("  ✓ Context round-trip successful")

    logger.info("\n5. Listing generations:")
    all_generations = storage.list_generations()
    logger.info(f"  Found {len(all_generations)} generations: {all_generations[:5]}...")

    logger.info("\n6. Cleanup test artifacts:")
    if template_path.exists():
        template_path.unlink()
        logger.info(f"  Removed test template: {template_path}")

    if gen_dir.exists():
        import shutil

        shutil.rmtree(gen_dir)
        logger.info(f"  Removed test generation: {gen_dir}")

    logger.info("\n✅ Storage validation complete!")
