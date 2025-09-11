"""GPT-5 prompt generator for evolution system."""

import json
import re
import sys
import time

from jinja2 import Template, TemplateSyntaxError
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from kaggle_map.evolution import EvolutionContext, PromptCandidate


class CandidateResponse(BaseModel):
    """Structured response for a single prompt candidate."""

    hypothesis: str = Field(description="Clear explanation of why this approach might work better")
    template: str = Field(description="Complete Jinja2 template")

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        required = {"question_text", "category", "mc_answer", "student_explanation"}
        var_pattern = r"{{\s*(\w+)\s*}}"
        found_vars = set(re.findall(var_pattern, v))

        missing = required - found_vars
        assert not missing, f"Template missing required variables: {missing}. Found: {found_vars}"
        
        return v


class GPT5Response(BaseModel):
    """Structured response from GPT-5 containing all candidates."""

    candidates: list[CandidateResponse] = Field(
        description="List of prompt candidates with hypotheses",
        min_length=1,
        max_length=10,
    )


def validate_template_variables(template: str) -> bool:
    required_vars = {
        "question_text",
        "category",
        "mc_answer",
        "student_explanation",
    }

    try:
        t = Template(template)
        set(t.module.__dict__.get("_body", []))
        var_pattern = r"{{\s*(\w+)\s*}}"
        found_vars = set(re.findall(var_pattern, template))

        missing = required_vars - found_vars
        if missing:
            logger.warning(f"Template missing required variables: {missing}")
            return False

        return True

    except TemplateSyntaxError as e:
        logger.error(f"Invalid template syntax: {e}")
        return False


def parse_structured_response(response_text: str, generation_id: int) -> list[PromptCandidate] | None:
    try:
        data = json.loads(response_text)
        gpt5_response = GPT5Response(**data)

        candidates = []
        for i, cand_resp in enumerate(gpt5_response.candidates):
            candidate = PromptCandidate(
                generation=generation_id,
                candidate_id=f"gen_{generation_id:02d}_candidate_{i}",
                prompt=cand_resp.template,
                hypothesis=cand_resp.hypothesis,
                parent_ids=[],
            )
            candidates.append(candidate)

        logger.info(f"Successfully parsed {len(candidates)} candidates using Pydantic")
        return candidates

    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"Structured parsing failed: {e}")
        return None


def parse_gpt5_response(response_text: str, generation_id: int) -> list[PromptCandidate]:
    candidates = parse_structured_response(response_text, generation_id)
    if candidates:
        return candidates

    logger.debug("Falling back to regex parsing")
    candidates = []

    candidate_blocks = re.split(r"CANDIDATE\s+\d+", response_text)[1:]

    for i, block in enumerate(candidate_blocks):
        hypothesis_match = re.search(r"HYPOTHESIS:\s*(.+?)(?=TEMPLATE:|$)", block, re.DOTALL)
        template_match = re.search(r"TEMPLATE:\s*(.+?)(?=CANDIDATE|$)", block, re.DOTALL)

        if hypothesis_match and template_match:
            hypothesis = hypothesis_match.group(1).strip()
            template = template_match.group(1).strip()

            if validate_template_variables(template):
                candidate = PromptCandidate(
                    generation=generation_id,
                    candidate_id=f"gen_{generation_id:02d}_candidate_{i}",
                    prompt=template,
                    hypothesis=hypothesis,
                    parent_ids=[],
                )
                candidates.append(candidate)
            else:
                logger.warning(f"Candidate {i} has invalid template, skipping")

    logger.info(f"Parsed {len(candidates)} valid candidates from GPT-5 response")
    return candidates


def build_meta_prompt(context: EvolutionContext, num_candidates: int) -> str:  # noqa: C901
    """Build meta-prompt for GPT-5 to generate candidates.

    Args:
        context: Evolution context with history and failures
        num_candidates: Number of candidates to generate

    Returns:
        Meta-prompt string
    """
    # Load competition context if available
    competition_info = context.competition_context or "No specific competition context available."

    # Format parent prompts
    parent_info = ""
    if context.parent_prompts:
        parent_info = "\n\n## Top Performing Parent Prompts:\n"
        for i, parent in enumerate(context.parent_prompts[:3], 1):
            parent_info += f"\n### Parent {i} (Generation {parent.generation}):\n"
            parent_info += f"Hypothesis: {parent.hypothesis}\n"
            parent_info += f"Template:\n```jinja2\n{parent.prompt}\n```\n"

    # Format failure patterns
    failure_info = ""
    if context.failure_patterns:
        failure_info = "\n\n## Common Failure Patterns:\n"
        for candidate_id, failures in list(context.failure_patterns.items())[:3]:
            failure_info += f"\n### From {candidate_id}:\n"
            for j, failure in enumerate(failures[:2], 1):
                failure_info += f"{j}. Question {failure.question_id}: "
                failure_info += f"Expected {failure.prediction.category.value}/"
                failure_info += f"{failure.prediction.misconception}, "
                failure_info += f"Got {failure.predicted[0].category.value}/"
                failure_info += f"{failure.predicted[0].misconception}\n"

    return f"""You are an expert prompt engineer tasked with evolving Jinja2 templates \
for a math misconception classification system.

## Competition Context:
{competition_info}

## Current Performance:
- Best MAP@3 Score: {context.current_best_score:.4f}
- Generation: {context.next_generation_id}
{parent_info}
{failure_info}

## Your Task:
Generate exactly {num_candidates} diverse prompt template candidates. Each candidate should:
1. Have a clear hypothesis about why it will work better
2. Be distinct from other candidates (explore different approaches)
3. Use Jinja2 template syntax
4. Include ALL required template variables: question_text, category, mc_answer, student_explanation

## Required Format:
You must provide your response in JSON format following this exact schema:

```json
{{
  "candidates": [
    {{
      "hypothesis": "Clear explanation of why this approach might work better",
      "template": "Your complete Jinja2 template here"
    }},
    {{
      "hypothesis": "Different hypothesis from candidate 1",
      "template": "Your complete Jinja2 template here"
    }}
  ]
}}
```

Generate exactly {num_candidates} candidates in the array.

## Strategies to Consider:
- Chain-of-thought reasoning
- Few-shot examples of common misconceptions
- Structured output formatting
- Breaking down the problem into steps
- Different ways to present the student explanation
- Mathematical notation handling
- Error pattern analysis prompts
- Confidence scoring
- Multi-step verification

Remember: Each template MUST include these variables:
- {{{{ question_text }}}}
- {{{{ category }}}}
- {{{{ mc_answer }}}}
- {{{{ student_explanation }}}}

Be creative, diverse, and hypothesis-driven in your approach!"""


def generate_candidates(  # noqa: C901, PLR0912
    context: EvolutionContext,
    num_candidates: int = 7,
    max_retries: int = 3,
) -> list[PromptCandidate]:
    """Generate prompt candidates using GPT-5.

    Args:
        context: Evolution context with history and failures
        num_candidates: Number of candidates to generate
        max_retries: Maximum retry attempts

    Returns:
        List of generated prompt candidates
    """
    logger.info(f"Generating {num_candidates} candidates for generation {context.next_generation_id}")

    # Initialize OpenAI client
    client = OpenAI()

    # Build meta-prompt
    meta_prompt = build_meta_prompt(context, num_candidates)
    logger.debug(f"Meta-prompt length: {len(meta_prompt)} characters")

    # Try to generate with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Calling GPT-5 Responses API")

            # Use the new Responses API with structured output
            # Provide Pydantic schema for structured response
            response = client.responses.create(
                model="gpt-5",
                input=meta_prompt,
                temperature=0.8,  # Some creativity but not too wild
                max_output_tokens=4000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "prompt_candidates",
                        "description": "List of prompt candidates with hypotheses",
                        "schema": GPT5Response.model_json_schema(),
                    },
                },
            )

            # Get the response text from the Responses API
            # The response object should have a direct text/content attribute
            response_text_raw = None
            if hasattr(response, "content"):
                response_text_raw = response.content
            elif hasattr(response, "text"):
                response_text_raw = response.text
            elif hasattr(response, "choices"):
                # Fallback for chat completions API format
                choices = response.choices
                if choices and len(choices) > 0:
                    first_choice = choices[0]
                    if hasattr(first_choice, "message"):
                        response_text_raw = first_choice.message.content
                    else:
                        response_text_raw = str(first_choice)
                else:
                    response_text_raw = str(response)
            else:
                response_text_raw = str(response)

            # Ensure response_text is always a string
            response_text: str = str(response_text_raw) if response_text_raw is not None else ""
            logger.info(f"Received GPT-5 response: {len(response_text)} characters")

            # Parse response into candidates
            candidates = parse_gpt5_response(response_text, context.next_generation_id)
            
            assert candidates is not None, "parse_gpt5_response returned None"

            # Set parent IDs based on context
            if context.parent_prompts:
                parent_ids = [p.candidate_id for p in context.parent_prompts[:3]]
                logger.debug(f"Setting parent IDs: {parent_ids}")
                for candidate in candidates:
                    candidate.parent_ids = parent_ids

            if len(candidates) >= num_candidates // 2:  # At least half the requested number
                logger.success(f"Successfully generated {len(candidates)} valid candidates")
                if len(candidates) > num_candidates:
                    logger.debug(f"Trimming to requested {num_candidates} candidates")
                return candidates[:num_candidates]  # Return up to requested number

            logger.warning(f"Only got {len(candidates)} valid candidates (need at least {num_candidates // 2}), retrying...")

        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries ({max_retries}) reached - unable to generate candidates")
                return []

    logger.error("Should not reach here - all attempts exhausted")
    return []


def show_json_schema() -> None:
    """Display the Pydantic JSON schema used for structured output."""
    logger.info("Generating Pydantic JSON schema for GPT-5")
    
    schema = GPT5Response.model_json_schema()
    
    assert schema, "Failed to generate JSON schema"
    assert "properties" in schema, "Schema missing properties"

    print("=" * 80)
    print("PYDANTIC JSON SCHEMA FOR GPT-5 STRUCTURED OUTPUT")
    print("=" * 80)
    print(json.dumps(schema, indent=2))
    print("=" * 80)
    
    logger.debug(f"Schema has {len(schema.get('properties', {}))} properties")


def main() -> None:
    """Manual test entry point."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    logger.info("Starting GPT-5 generator manual test")

    # Show the JSON schema
    show_json_schema()

    # Create a test context
    logger.info("Creating test evolution context")
    context = EvolutionContext(
        current_best_prompt="baseline",
        current_best_score=0.55,
        parent_prompts=[],
        failure_patterns={},
        competition_context="Math misconception classification for Kaggle MAP competition",
        next_generation_id=0,
    )
    
    assert context, "Failed to create test context"
    logger.debug(f"Test context: {context}")

    # Generate candidates
    logger.info("Generating test candidates")
    candidates = generate_candidates(context, num_candidates=3)
    
    assert candidates, "No candidates generated"
    logger.success(f"Generated {len(candidates)} test candidates")

    # Display results
    print("\n" + "=" * 80)
    print("GENERATED CANDIDATES")
    print("=" * 80)
    for i, candidate in enumerate(candidates, 1):
        print(f"\nCandidate {i}: {candidate.candidate_id}")
        print(f"Hypothesis: {candidate.hypothesis}")
        print(f"Template preview: {candidate.prompt[:200]}...")
    print("=" * 80)
    
    logger.success("Manual test completed successfully")


if __name__ == "__main__":
    main()
