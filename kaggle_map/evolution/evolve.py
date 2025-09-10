"""Evolution orchestrator for prompt optimization."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from kaggle_map.evolution import (
    EvolutionContext,
    Generation,
    PromptCandidate,
)
from kaggle_map.evolution.analysis import load_and_analyze_errors
from kaggle_map.evolution.evaluator import evaluate_all_candidates
from kaggle_map.evolution.generator import generate_candidates
from kaggle_map.evolution.storage import Storage


def check_convergence(
    scores: list[float],
    threshold: float = 0.01,
    window: int = 3,
) -> bool:
    """Check if evolution has converged.

    Args:
        scores: List of best scores per generation
        threshold: Minimum improvement threshold
        window: Number of generations to check

    Returns:
        True if improvement < threshold over window generations
    """
    if len(scores) < window + 1:
        return False

    # Check improvement over last window generations
    recent_scores = scores[-window - 1 :]
    max_improvement = max(recent_scores[1:]) - recent_scores[0]

    logger.debug(f"Max improvement over last {window} generations: {max_improvement:.4f}")
    return max_improvement < threshold


def select_top_performers(
    generation: Generation,
    top_percentage: float = 0.4,
) -> list[PromptCandidate]:
    """Select top performing candidates from a generation.

    Args:
        generation: Generation with evaluated candidates
        top_percentage: Fraction of candidates to select

    Returns:
        Top performing candidates
    """
    # Evaluations are already sorted by MAP score descending
    num_to_select = max(1, int(len(generation.candidates) * top_percentage))
    logger.info(f"Selecting top {num_to_select} out of {len(generation.candidates)} candidates")

    # Map evaluation scores to candidates
    selected_ids = {evaluation.candidate_id for evaluation in generation.evaluations[:num_to_select]}
    selected = [c for c in generation.candidates if c.candidate_id in selected_ids]

    logger.debug(f"Selected candidates: {[c.candidate_id for c in selected]}")
    return selected


def build_evolution_context(  # noqa: C901
    all_generations: list[Generation],
    top_k: int = 3,
) -> EvolutionContext:
    """Build context for next generation.

    Args:
        all_generations: All previous generations
        top_k: Number of top performers to track

    Returns:
        Context for generating next batch
    """
    # Find top performers across all generations
    all_results = []
    for gen in all_generations:
        for evaluation in gen.evaluations:
            # Find matching candidate
            candidate = next(c for c in gen.candidates if c.candidate_id == evaluation.candidate_id)
            all_results.append((candidate, evaluation))

    # Sort by MAP score
    all_results.sort(key=lambda x: x[1].map_score, reverse=True)

    # Extract top performers
    top_candidates = [candidate for candidate, _ in all_results[:top_k]]
    best_score = all_results[0][1].map_score if all_results else 0.0
    best_id = all_results[0][0].candidate_id if all_results else "baseline"

    # Extract failure patterns from top performers
    failure_patterns = {}
    for candidate, evaluation in all_results[:top_k]:
        if evaluation.failure_samples:
            failure_patterns[candidate.candidate_id] = evaluation.failure_samples

    # Load competition context
    comp_context_path = Path("docs/competition.md")
    if comp_context_path.exists():
        competition_context = comp_context_path.read_text()
    else:
        competition_context = "Math misconception classification for Kaggle MAP competition"

    return EvolutionContext(
        current_best_prompt=best_id,
        current_best_score=best_score,
        parent_prompts=top_candidates,
        failure_patterns=failure_patterns,
        competition_context=competition_context,
        next_generation_id=len(all_generations),
    )


def run_generation(context: EvolutionContext) -> Generation:
    """Run a single evolution generation.

    Args:
        context: Evolution context

    Returns:
        Completed generation with evaluations
    """
    gen_id = context.next_generation_id
    logger.info(f"=== Starting Generation {gen_id} ===")
    logger.info(f"Current best score: {context.current_best_score:.4f}")

    # Generate candidates
    candidates = generate_candidates(context, num_candidates=7)
    if not candidates:
        logger.error("Failed to generate candidates")
        return Generation(
            generation_id=gen_id,
            candidates=[],
            evaluations=[],
            timestamp=datetime.now(),
        )

    logger.info(f"Generated {len(candidates)} candidates")

    # Evaluate candidates
    evaluations = evaluate_all_candidates(candidates, sample_ratio=0.1)

    # Create generation
    generation = Generation(
        generation_id=gen_id,
        candidates=candidates,
        evaluations=evaluations,
        timestamp=datetime.now(),
    )

    # Log results
    logger.info(f"Generation {gen_id} results:")
    for i, evaluation in enumerate(evaluations[:3], 1):
        candidate = next(c for c in candidates if c.candidate_id == evaluation.candidate_id)
        logger.info(f"  {i}. {evaluation.candidate_id}: MAP@3 = {evaluation.map_score:.4f}")
        logger.debug(f"     Hypothesis: {candidate.hypothesis}")

    return generation


def evolve_prompts(  # noqa: C901
    max_generations: int = 10,
    convergence_threshold: float = 0.01,
    convergence_window: int = 3,
) -> PromptCandidate | None:
    """Main evolution loop.

    Args:
        max_generations: Maximum number of generations
        convergence_threshold: Stop if improvement < this over window
        convergence_window: Number of generations to check for convergence

    Returns:
        Best performing prompt candidate
    """
    logger.info("Starting prompt evolution system")
    logger.info(f"Max generations: {max_generations}")
    logger.info(f"Convergence: < {convergence_threshold:.1%} over {convergence_window} generations")

    storage = Storage()
    all_generations = []
    best_scores = []

    # Bootstrap: Analyze initial errors if no context exists
    context_path = Path("reranker/prompts/generations/context.json")
    if not context_path.exists():
        logger.info("Bootstrapping from error_prediction.csv")
        _ = load_and_analyze_errors()

        # Create initial context
        initial_context = EvolutionContext(
            current_best_prompt="baseline",
            current_best_score=0.0,
            parent_prompts=[],
            failure_patterns={},
            competition_context="Math misconception classification for Kaggle MAP competition",
            next_generation_id=0,
        )
        storage.save_context(initial_context)

    # Main evolution loop
    for gen_num in range(max_generations):
        # Build context for this generation
        context = build_evolution_context(all_generations)

        # Run generation
        generation = run_generation(context)
        all_generations.append(generation)

        # Save generation
        storage.save_generation(generation)

        # Track best score
        if generation.evaluations:
            best_score = generation.evaluations[0].map_score
            best_scores.append(best_score)
            logger.info(f"Generation {gen_num} best score: {best_score:.4f}")

            # Update context with new best
            context.current_best_score = best_score
            context.current_best_prompt = generation.evaluations[0].candidate_id
            storage.save_context(context)

        # Check convergence
        if check_convergence(best_scores, convergence_threshold, convergence_window):
            logger.info(f"Converged after {gen_num + 1} generations")
            break

    # Find overall best
    best_candidate = None
    best_score = 0.0
    for gen in all_generations:
        for evaluation in gen.evaluations:
            if evaluation.map_score > best_score:
                best_score = evaluation.map_score
                # Find matching candidate
                best_candidate = next(c for c in gen.candidates if c.candidate_id == evaluation.candidate_id)

    if best_candidate:
        logger.success(f"Evolution complete! Best candidate: {best_candidate.candidate_id}")
        logger.success(f"Best MAP@3 score: {best_score:.4f}")
        logger.info(f"Hypothesis: {best_candidate.hypothesis}")
    else:
        logger.warning("No successful candidates generated")

    return best_candidate


def main() -> None:
    """Entry point for standalone execution."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/evolution.log", level="DEBUG", rotation="10 MB")

    logger.info("=" * 80)
    logger.info("PROMPT EVOLUTION SYSTEM")
    logger.info("=" * 80)

    # Run evolution
    best_prompt = evolve_prompts(
        max_generations=10,
        convergence_threshold=0.01,
        convergence_window=3,
    )

    if best_prompt:
        print("\n" + "=" * 80)
        print("EVOLUTION COMPLETE")
        print("=" * 80)
        print(f"Best prompt: {best_prompt.candidate_id}")
        print(f"Hypothesis: {best_prompt.hypothesis}")
        print(f"Template saved to: prompts/{best_prompt.candidate_id}.j2")
        print("=" * 80)


if __name__ == "__main__":
    main()
