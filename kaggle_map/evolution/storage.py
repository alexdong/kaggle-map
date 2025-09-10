"""Storage layer for prompt evolution system."""

import json
from pathlib import Path

from loguru import logger

from kaggle_map.evolution import EvolutionContext, Generation, GenerationID, PromptCandidate


class Storage:
    """Handles persistence of prompts and evaluation results."""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize storage with base directory.

        Args:
            base_dir: Base directory for all storage. Defaults to project root.
        """
        self.base_dir = base_dir or Path.cwd()
        logger.debug(f"Storage initialized with base_dir: {self.base_dir}")

    def get_prompt_template_path(self, candidate_id: str) -> Path:
        """Get path for a prompt template file.

        Args:
            candidate_id: Candidate identifier (e.g., "gen_00_candidate_0")

        Returns:
            Path to the .j2 template file
        """
        path = self.base_dir / "reranker" / "prompts" / f"{candidate_id}.j2"
        logger.debug(f"Prompt template path for {candidate_id}: {path}")
        return path

    def get_generation_dir(self, generation_id: GenerationID) -> Path:
        """Get directory for a generation's metadata.

        Args:
            generation_id: Generation number

        Returns:
            Path to generation directory
        """
        path = self.base_dir / "reranker" / "prompts" / "generations" / f"gen_{generation_id:02d}"
        logger.debug(f"Generation directory for {generation_id}: {path}")
        return path

    def get_context_path(self) -> Path:
        """Get path for evolution context file.

        Returns:
            Path to context.json
        """
        path = self.base_dir / "reranker" / "prompts" / "generations" / "context.json"
        logger.debug(f"Context path: {path}")
        return path

    def save_prompt_template(self, candidate: PromptCandidate) -> None:
        """Save a prompt template to .j2 file.

        Args:
            candidate: Prompt candidate with template content
        """
        path = self.get_prompt_template_path(candidate.candidate_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving prompt template: {candidate.candidate_id}")
        path.write_text(candidate.prompt)
        logger.debug(f"Saved {len(candidate.prompt)} chars to {path}")

    def load_prompt_template(self, candidate_id: str) -> str:
        """Load a prompt template from .j2 file.

        Args:
            candidate_id: Candidate identifier

        Returns:
            Jinja2 template content
        """
        path = self.get_prompt_template_path(candidate_id)
        assert path.exists(), f"Template not found: {path}"

        logger.debug(f"Loading prompt template: {candidate_id}")
        content = path.read_text()
        logger.debug(f"Loaded {len(content)} chars from {path}")
        return content

    def save_generation(self, generation: Generation) -> None:
        """Save generation metadata to JSON.

        Args:
            generation: Generation with candidates and evaluations
        """
        gen_dir = self.get_generation_dir(generation.generation_id)
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save main generation file
        gen_path = gen_dir / "generation.json"
        logger.info(f"Saving generation {generation.generation_id} with {len(generation.candidates)} candidates")

        gen_data = generation.model_dump(mode="json")
        gen_path.write_text(json.dumps(gen_data, indent=2))
        logger.debug(f"Saved generation metadata to {gen_path}")

        # Save individual candidate files for easier access
        for candidate in generation.candidates:
            candidate_path = gen_dir / f"{candidate.candidate_id}.json"

            # Find matching evaluation
            evaluation = next(
                (e for e in generation.evaluations if e.candidate_id == candidate.candidate_id),
                None,
            )

            candidate_data = {
                "candidate": candidate.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json") if evaluation else None,
            }

            candidate_path.write_text(json.dumps(candidate_data, indent=2))
            logger.debug(f"Saved candidate {candidate.candidate_id} to {candidate_path}")

    def load_generation(self, generation_id: GenerationID) -> Generation:
        """Load generation metadata from JSON.

        Args:
            generation_id: Generation number

        Returns:
            Generation object with all data
        """
        gen_dir = self.get_generation_dir(generation_id)
        gen_path = gen_dir / "generation.json"

        assert gen_path.exists(), f"Generation file not found: {gen_path}"

        logger.debug(f"Loading generation {generation_id}")
        gen_data = json.loads(gen_path.read_text())

        generation = Generation.model_validate(gen_data)
        logger.info(f"Loaded generation {generation_id} with {len(generation.candidates)} candidates")

        return generation

    def save_context(self, context: EvolutionContext) -> None:
        """Save evolution context to JSON.

        Args:
            context: Current evolution context
        """
        path = self.get_context_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving evolution context for generation {context.next_generation_id}")
        context_data = context.model_dump(mode="json")
        path.write_text(json.dumps(context_data, indent=2))
        logger.debug(f"Saved context to {path}")

    def load_context(self) -> EvolutionContext:
        """Load evolution context from JSON.

        Returns:
            Current evolution context
        """
        path = self.get_context_path()
        assert path.exists(), f"Context file not found: {path}"

        logger.debug("Loading evolution context")
        context_data = json.loads(path.read_text())

        context = EvolutionContext.model_validate(context_data)
        logger.info(f"Loaded context for generation {context.next_generation_id}")

        return context

    def list_generations(self) -> list[GenerationID]:
        """List all available generation IDs.

        Returns:
            Sorted list of generation IDs
        """
        generations_dir = self.base_dir / "reranker" / "prompts" / "generations"

        if not generations_dir.exists():
            logger.debug("No generations directory found")
            return []

        # Find all gen_XX directories
        gen_dirs = [d for d in generations_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")]

        # Extract generation IDs
        generation_ids = []
        for gen_dir in gen_dirs:
            try:
                gen_id = int(gen_dir.name.split("_")[1])
                generation_ids.append(gen_id)
            except (IndexError, ValueError):
                logger.warning(f"Skipping invalid generation directory: {gen_dir.name}")

        generation_ids.sort()
        logger.debug(f"Found {len(generation_ids)} generations: {generation_ids}")

        return generation_ids
