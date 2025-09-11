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
    assert scores is not None, "Cannot check convergence on None scores"
    assert threshold > 0, f"Threshold must be positive, got {threshold}"
    assert window > 0, f"Window must be positive, got {window}"
    
    if len(scores) < window + 1:
        return False

    recent_scores = scores[-window - 1 :]
    max_improvement = max(recent_scores[1:]) - recent_scores[0]
    
    converged = max_improvement < threshold
    if converged:
        logger.info(f"Convergence detected: improvement {max_improvement:.4f} < threshold {threshold:.4f}")
    
    return converged


def select_top_performers(
    generation: Generation,
    top_percentage: float = 0.4,
) -> list[PromptCandidate]:
    assert generation, "Cannot select from None generation"
    assert generation.candidates, f"Generation {generation.generation_id} has no candidates"
    assert 0.0 < top_percentage <= 1.0, f"Top percentage must be between 0 and 1, got {top_percentage}"
    
    num_to_select = max(1, int(len(generation.candidates) * top_percentage))
    logger.info(f"Selecting top {num_to_select} out of {len(generation.candidates)} candidates ({top_percentage:.0%})")

    selected_ids = {evaluation.candidate_id for evaluation in generation.evaluations[:num_to_select]}
    selected = [c for c in generation.candidates if c.candidate_id in selected_ids]
    
    assert len(selected) == num_to_select, f"Expected {num_to_select} selected candidates, got {len(selected)}"
    
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
    assert all_generations is not None, "Cannot build context from None generations"
    assert top_k > 0, f"Top K must be positive, got {top_k}"
    
    all_results = []
    for gen in all_generations:
        assert gen.evaluations, f"Generation {gen.generation_id} has no evaluations"
        
        for evaluation in gen.evaluations:
            candidate = next((c for c in gen.candidates if c.candidate_id == evaluation.candidate_id), None)
            assert candidate, f"No candidate found for evaluation {evaluation.candidate_id} in generation {gen.generation_id}"
            all_results.append((candidate, evaluation))

    all_results.sort(key=lambda x: (-x[1].map_score, x[0].candidate_id))

    top_candidates = [candidate for candidate, _ in all_results[:top_k]]
    best_score = all_results[0][1].map_score if all_results else 0.0
    best_id = all_results[0][0].candidate_id if all_results else "baseline"
    
    logger.info(f"Current best performer: {best_id} with MAP@3={best_score:.4f}")

    failure_patterns = {}
    
    for candidate, evaluation in all_results[:top_k]:
        if evaluation.failure_samples:
            failure_patterns[candidate.candidate_id] = evaluation.failure_samples

    # Load competition context
    comp_context_path = Path("docs/competition.md")
    if comp_context_path.exists():
        competition_context = comp_context_path.read_text()
        logger.debug(f"Loaded competition context: {len(competition_context)} chars")
    else:
        competition_context = "Math misconception classification for Kaggle MAP competition"
        logger.debug("Using default competition context (docs/competition.md not found)")

    next_gen_id = len(all_generations)
    
    context = EvolutionContext(
        current_best_prompt=best_id,
        current_best_score=best_score,
        parent_prompts=top_candidates,
        failure_patterns=failure_patterns,
        competition_context=competition_context,
        next_generation_id=next_gen_id,
    )
    
    logger.success(f"Built evolution context for generation {next_gen_id}")
    logger.debug(f"  Context: {context}")
    
    return context


def run_generation(context: EvolutionContext) -> Generation:
    """Run a single evolution generation.

    Args:
        context: Evolution context

    Returns:
        Completed generation with evaluations
    """
    assert context, "Cannot run generation with None context"
    
    gen_id = context.next_generation_id
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Generation {gen_id}")
    logger.info(f"Current best: {context.current_best_prompt} (MAP@3: {context.current_best_score:.4f})")
    logger.info(f"Parent prompts: {len(context.parent_prompts)}")
    logger.info(f"Failure patterns: {sum(len(f) for f in context.failure_patterns.values())}")
    logger.info(f"{'='*60}")

    # Generate candidates
    logger.info("Phase 1: Generating candidate prompts...")
    candidates = generate_candidates(context, num_candidates=7)
    
    if not candidates:
        logger.error("❌ Failed to generate any candidates")
        return Generation(
            generation_id=gen_id,
            candidates=[],
            evaluations=[],
            timestamp=datetime.now(),
        )
    
    assert len(candidates) > 0, f"Generated {len(candidates)} candidates but expected > 0"

    logger.success(f"✅ Generated {len(candidates)} candidates")
    for i, candidate in enumerate(candidates, 1):
        logger.debug(f"  Candidate {i}: {candidate.candidate_id} - {candidate.hypothesis[:60]}...")

    # Evaluate candidates
    logger.info("Phase 2: Evaluating candidates...")
    evaluations = evaluate_all_candidates(candidates, sample_ratio=0.1)
    
    assert evaluations, f"No evaluations returned for generation {gen_id}"
    assert len(evaluations) == len(candidates), f"Evaluation count mismatch: {len(evaluations)} evaluations for {len(candidates)} candidates"

    # Create generation
    generation = Generation(
        generation_id=gen_id,
        candidates=candidates,
        evaluations=evaluations,
        timestamp=datetime.now(),
    )
    
    assert generation, "Failed to create generation object"

    # Log results
    best_score = evaluations[0].map_score if evaluations else 0.0
    worst_score = evaluations[-1].map_score if evaluations else 0.0
    avg_score = sum(e.map_score for e in evaluations) / len(evaluations) if evaluations else 0.0
    
    logger.success(f"\n🎯 Generation {gen_id} Complete!")
    logger.success(f"  Best:    {best_score:.4f} ({evaluations[0].candidate_id if evaluations else 'N/A'})")
    logger.success(f"  Worst:   {worst_score:.4f} ({evaluations[-1].candidate_id if evaluations else 'N/A'})")
    logger.success(f"  Average: {avg_score:.4f}")
    logger.success(f"  Spread:  {best_score - worst_score:.4f}")
    
    # Log top performers
    logger.info(f"\nTop 3 performers in generation {gen_id}:")
    for i, evaluation in enumerate(evaluations[:3], 1):
        candidate = next((c for c in candidates if c.candidate_id == evaluation.candidate_id), None)
        assert candidate, f"No candidate found for evaluation {evaluation.candidate_id}"
        
        logger.info(f"  {i}. {evaluation.candidate_id}: MAP@3 = {evaluation.map_score:.4f}")
        logger.debug(f"     Hypothesis: {candidate.hypothesis[:80]}{'...' if len(candidate.hypothesis) > 80 else ''}")

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
    assert max_generations > 0, f"Max generations must be positive, got {max_generations}"
    assert convergence_threshold > 0, f"Convergence threshold must be positive, got {convergence_threshold}"
    assert convergence_window > 0, f"Convergence window must be positive, got {convergence_window}"
    
    logger.info(f"\n{'='*80}")
    logger.info("STARTING PROMPT EVOLUTION SYSTEM")
    logger.info(f"{'='*80}")
    logger.info(f"Configuration:")
    logger.info(f"  Max generations: {max_generations}")
    logger.info(f"  Convergence threshold: {convergence_threshold:.1%} improvement")
    logger.info(f"  Convergence window: {convergence_window} generations")
    logger.info(f"{'='*80}")

    storage = Storage()
    all_generations = []
    best_scores = []

    # Bootstrap: Analyze initial errors if no context exists
    context_path = Path("reranker/prompts/generations/context.json")
    if not context_path.exists():
        logger.info("\n🚀 Bootstrapping evolution system...")
        logger.info("Analyzing error_prediction.csv for initial patterns")
        
        error_summary = load_and_analyze_errors()
        assert error_summary, "Failed to analyze initial errors"
        logger.success("✅ Initial error analysis complete")

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
        logger.info("✅ Initial context saved")
    else:
        logger.info("🔄 Resuming from existing evolution context")

    # Main evolution loop
    logger.info(f"\n🎯 Starting evolution with up to {max_generations} generations\n")
    
    for gen_num in range(max_generations):
        logger.info(f"\n[Generation {gen_num + 1}/{max_generations}]")
        
        # Build context for this generation
        context = build_evolution_context(all_generations)
        assert context, f"Failed to build context for generation {gen_num}"

        # Run generation
        generation = run_generation(context)
        assert generation, f"Failed to run generation {gen_num}"
        
        all_generations.append(generation)
        logger.debug(f"Added generation {gen_num} to history (total: {len(all_generations)})")

        # Save generation
        storage.save_generation(generation)
        logger.debug(f"Persisted generation {gen_num} to disk")

        # Track best score
        if generation.evaluations:
            best_score = generation.evaluations[0].map_score
            best_scores.append(best_score)
            
            # Track improvement
            improvement = best_score - (best_scores[-2] if len(best_scores) > 1 else 0.0)
            improvement_str = f" ({improvement:+.4f})" if len(best_scores) > 1 else ""
            
            logger.info(f"🏆 Generation {gen_num} best score: {best_score:.4f}{improvement_str}")
            
            if improvement > 0:
                logger.success(f"📈 Improvement detected: +{improvement:.4f}")
            elif improvement < 0:
                logger.warning(f"📉 Performance declined: {improvement:.4f}")

            # Update context with new best
            context.current_best_score = best_score
            context.current_best_prompt = generation.evaluations[0].candidate_id
            storage.save_context(context)
        else:
            logger.error(f"❌ Generation {gen_num} produced no evaluations")
            best_scores.append(0.0)

        # Check convergence
        if check_convergence(best_scores, convergence_threshold, convergence_window):
            logger.success(f"🏁 Evolution converged after {gen_num + 1} generations!")
            break
    else:
        logger.info(f"🗓️ Completed all {max_generations} generations (no convergence)")

    # Find overall best
    logger.info(f"\n🔍 Finding best candidate across all {len(all_generations)} generations...")
    
    best_candidate = None
    best_score = 0.0
    best_generation = -1
    
    for gen_idx, gen in enumerate(all_generations):
        for evaluation in gen.evaluations:
            if evaluation.map_score > best_score:
                best_score = evaluation.map_score
                best_generation = gen_idx
                # Find matching candidate
                best_candidate = next((c for c in gen.candidates if c.candidate_id == evaluation.candidate_id), None)
                assert best_candidate, f"No candidate found for best evaluation {evaluation.candidate_id}"

    if best_candidate:
        logger.success(f"\n🎆 EVOLUTION COMPLETE!")
        logger.success(f"  Best candidate: {best_candidate.candidate_id} (from generation {best_generation})")
        logger.success(f"  Best MAP@3 score: {best_score:.4f}")
        logger.success(f"  Hypothesis: {best_candidate.hypothesis[:100]}{'...' if len(best_candidate.hypothesis) > 100 else ''}")
        logger.success(f"  Template saved: reranker/prompts/{best_candidate.candidate_id}.j2")
    else:
        logger.error("❌ No successful candidates generated across all generations")

    return best_candidate


def main() -> None:
    """Entry point for standalone execution."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")
    logger.add("logs/evolution.log", level="DEBUG", rotation="10 MB", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} - {message}")
    
    logger.info("Evolution system starting up...")

    # No need for additional header - evolve_prompts will log its own

    # Run evolution
    best_prompt = evolve_prompts(
        max_generations=10,
        convergence_threshold=0.01,
        convergence_window=3,
    )

    if best_prompt:
        print("\n" + "=" * 80)
        print("🎆 EVOLUTION COMPLETE - SUCCESS! 🎆")
        print("=" * 80)
        print(f"Best prompt: {best_prompt.candidate_id}")
        print(f"Hypothesis: {best_prompt.hypothesis}")
        print(f"Template saved to: reranker/prompts/{best_prompt.candidate_id}.j2")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ EVOLUTION FAILED - NO VIABLE CANDIDATES")
        print("=" * 80)
        print("Check logs for errors and try adjusting parameters")
        print("=" * 80)


if __name__ == "__main__":
    main()
