"""Storage layer for prompt evolution system."""

import json
from pathlib import Path

from loguru import logger

from kaggle_map.evolution import EvolutionContext, Generation, GenerationID, PromptCandidate


class Storage:
    """Handles persistence of prompts and evaluation results."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.cwd()
        assert self.base_dir.exists(), f"Base directory does not exist: {self.base_dir}"
        assert self.base_dir.is_dir(), f"Base path is not a directory: {self.base_dir}"

    def get_prompt_template_path(self, candidate_id: str) -> Path:
        assert candidate_id and candidate_id.strip(), f"Candidate ID cannot be empty: '{candidate_id}'"
        assert "gen_" in candidate_id, f"Invalid candidate ID format (missing 'gen_'): {candidate_id}"
        
        path = self.base_dir / "reranker" / "prompts" / f"{candidate_id}.j2"
        return path

    def get_generation_dir(self, generation_id: GenerationID) -> Path:
        assert isinstance(generation_id, int), f"Generation ID must be int, got {type(generation_id).__name__}"
        assert generation_id >= 0, f"Generation ID must be non-negative, got {generation_id}"
        
        path = self.base_dir / "reranker" / "prompts" / "generations" / f"gen_{generation_id:02d}"
        return path

    def get_context_path(self) -> Path:
        return self.base_dir / "reranker" / "prompts" / "generations" / "context.json"

    def save_prompt_template(self, candidate: PromptCandidate) -> None:
        assert candidate, "Cannot save None candidate"
        assert candidate.prompt, f"Cannot save empty prompt for {candidate.candidate_id}"
        
        path = self.get_prompt_template_path(candidate.candidate_id)
        
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists():
            logger.warning(f"Overwriting existing template: {candidate.candidate_id}")
        
        logger.info(f"Saving prompt template: {candidate.candidate_id} ({len(candidate.prompt)} chars)")
        
        path.write_text(candidate.prompt)
        assert path.exists(), f"Failed to create file: {path}"
        assert path.stat().st_size > 0, f"File is empty after write: {path}"

    def load_prompt_template(self, candidate_id: str) -> str:
        assert candidate_id, f"Cannot load template with empty ID: '{candidate_id}'"
        
        path = self.get_prompt_template_path(candidate_id)
        assert path.exists(), f"Template file not found at {path} for candidate {candidate_id}"
        assert path.is_file(), f"Path exists but is not a file: {path}"
        
        file_size = path.stat().st_size
        assert file_size > 0, f"Template file is empty: {path} (0 bytes)"
        
        content = path.read_text()
        assert content and content.strip(), f"Loaded empty content from {path}"
        return content

    def save_generation(self, generation: Generation) -> None:
        assert generation, "Cannot save None generation"
        assert generation.candidates, f"Cannot save generation {generation.generation_id} with no candidates"
        
        gen_dir = self.get_generation_dir(generation.generation_id)
        
        if not gen_dir.exists():
            gen_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.warning(f"Generation directory already exists, will overwrite: {gen_dir}")
        
        gen_path = gen_dir / "generation.json"
        logger.info(f"Saving generation {generation.generation_id}: {len(generation.candidates)} candidates, {len(generation.evaluations)} evaluations")
        
        gen_data = generation.model_dump(mode="json")
        json_content = json.dumps(gen_data, indent=2)
        gen_path.write_text(json_content)
        
        assert gen_path.exists(), f"Generation file not created: {gen_path}"
        assert gen_path.stat().st_size > 0, f"Generation file is empty: {gen_path}"

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
            assert candidate_path.exists(), f"Candidate file not created: {candidate_path}"
            
            saved_count += 1
                
        logger.info(f"Saved {saved_count}/{len(generation.candidates)} candidate files")

    def load_generation(self, generation_id: GenerationID) -> Generation:
        assert isinstance(generation_id, int) and generation_id >= 0, f"Invalid generation ID: {generation_id}"
        
        gen_dir = self.get_generation_dir(generation_id)
        gen_path = gen_dir / "generation.json"

        assert gen_dir.exists(), f"Generation directory not found: {gen_dir}"
        assert gen_path.exists(), f"Generation file not found: {gen_path}"
        assert gen_path.stat().st_size > 0, f"Generation file is empty: {gen_path}"
        
        json_content = gen_path.read_text()
        gen_data = json.loads(json_content)
        generation = Generation.model_validate(gen_data)
        
        assert generation.generation_id == generation_id, f"ID mismatch: expected {generation_id}, got {generation.generation_id}"
        
        logger.info(f"Loaded generation {generation_id}: {len(generation.candidates)} candidates, {len(generation.evaluations)} evaluations")
        
        if generation.evaluations:
            best = generation.evaluations[0]
            logger.info(f"  Best performer: {best.candidate_id} with MAP@3={best.map_score:.4f}")
        
        return generation

    def save_context(self, context: EvolutionContext) -> None:
        assert context, "Cannot save None context"
        
        path = self.get_context_path()
        
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving evolution context for generation {context.next_generation_id}")
        
        context_data = context.model_dump(mode="json")
        json_content = json.dumps(context_data, indent=2)
        path.write_text(json_content)
        
        assert path.exists(), f"Context file not created: {path}"
        assert path.stat().st_size > 0, f"Context file is empty: {path}"

    def load_context(self) -> EvolutionContext:
        path = self.get_context_path()
        assert path.exists(), f"Context file not found at {path} - may need to bootstrap"
        assert path.is_file(), f"Context path exists but is not a file: {path}"
        assert path.stat().st_size > 0, f"Context file is empty: {path}"
        
        json_content = path.read_text()
        context_data = json.loads(json_content)
        context = EvolutionContext.model_validate(context_data)
        
        logger.info(f"Loaded context for generation {context.next_generation_id}")
        logger.info(f"  Current best: {context.current_best_prompt} (MAP@3: {context.current_best_score:.4f})")
        logger.info(f"  Parents: {len(context.parent_prompts)}, Failure patterns: {len(context.failure_patterns)}")
        
        return context

    def list_generations(self) -> list[GenerationID]:
        generations_dir = self.base_dir / "reranker" / "prompts" / "generations"

        if not generations_dir.exists():
            logger.info("No generations directory found - this may be a fresh start")
            return []
        
        gen_dirs = [d for d in generations_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")]

        generation_ids = []
        invalid_dirs = []
        
        for gen_dir in gen_dirs:
            parts = gen_dir.name.split("_")
            if len(parts) < 2:
                logger.warning(f"  Invalid generation directory format: '{gen_dir.name}'")
                invalid_dirs.append(gen_dir.name)
                continue
            
            gen_id_str = parts[1]
            if not gen_id_str.isdigit():
                logger.warning(f"  Invalid generation ID (not a number): '{gen_dir.name}'")
                invalid_dirs.append(gen_dir.name)
                continue
                
            gen_id = int(gen_id_str)
            if gen_id < 0:
                logger.warning(f"  Invalid generation ID (negative): {gen_id} in '{gen_dir.name}'")
                invalid_dirs.append(gen_dir.name)
                continue
            
            gen_file = gen_dir / "generation.json"
            if gen_file.exists():
                generation_ids.append(gen_id)
            else:
                logger.warning(f"  Generation {gen_id}: missing generation.json")
                invalid_dirs.append(gen_dir.name)

        generation_ids.sort()
        
        logger.info(f"Found {len(generation_ids)} valid generations: {generation_ids[:5]}{'...' if len(generation_ids) > 5 else ''}")
        
        if invalid_dirs:
            logger.warning(f"Skipped {len(invalid_dirs)} invalid directories: {invalid_dirs}")

        return generation_ids
