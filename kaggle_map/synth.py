#!/usr/bin/env python3
"""
Generate synthetic dataset by rewriting StudentExplanation using LLM.
Creates 100x expansion: ~5.5M rows from ~55K training rows.
"""

import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
from jinja2 import Template
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

# Constants
DEBUG_ROW_LIMIT = 5

# Configuration
LM_STUDIO_URL = "http://192.168.0.15:1234/v1/chat/completions"
INPUT_FILE = Path("dataset/train.csv")
OUTPUT_FILE = Path("dataset/synth.csv")
EXPANSIONS_PER_ROW = 10
CHECKPOINT_INTERVAL = 1000  # Save progress every N rows

# Jinja2 template for rewriting
REWRITE_TEMPLATE = Template("""
Rewrite the following <original> sentence by keeping the meaning intact. A wrong answer often means they have a misconception. Make sure you identify the misconcept and keep it the same during your rewrite.

The student was presented a math problem: {{ QuestionText }}
The student's answer was: {{ MC_Answer }}

This <original> sentence was when they were asked how they solved the problem (as a way to understand how they approach the solution.)

The student doesn't know whether the answer is correct or not. They describe their thought process in the following <original> sentence.

REWRITE CONSTRAINTS:
- Student age: {{ age }} years old
- Writing style: {{ writing_style }} (casual=contractions/informal, formal=proper grammar, hesitant=uncertainty, confident=definitive, conversational=like talking to friend)
- Focus on: {{ explanation_focus }} (process=steps taken, visual=what they see, reasoning=why they think it's right, comparison=compare to similar problems)
- Detail level: {{ detail_level }} (brief=short, verbose=lots of explanation, rambling=goes off on tangents)
- Math vocabulary: {{ math_vocabulary }} (simple=basic terms, mixed=some advanced/some basic, advanced=proper mathematical terminology)
- Sentence structure: {{ sentence_structure }} (simple=short sentences, compound=longer connected sentences, fragments=incomplete thoughts like real student writing)
- Personal references: {{ personal_references }} (include=add "my teacher said", "I learned", "like when we did"; exclude=keep purely about the problem)

Make it less than {{ char_limit }} characters.

<original>{{ StudentExplanation }}</original>

PLEASE RETURN ONLY the rewritten text. NOTHING ELSE.
""".strip())

console = Console()
logger.add("logs/synth_generation.log", rotation="100 MB")


class SynthDataGenerator:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.checkpoint_file = Path("synth_checkpoint.json")
        self.progress_data = self.load_checkpoint()
        
    def load_checkpoint(self) -> dict[str, Any]:
        """Load progress from checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                with self.checkpoint_file.open() as f:
                    data = json.load(f)
                logger.info(f"Resuming from checkpoint: processed {data.get('processed_rows', 0)} rows")
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {"processed_rows": 0, "failed_requests": 0}
    
    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        with self.checkpoint_file.open("w") as f:
            json.dump(self.progress_data, f)
    
    def call_llm(self, prompt: str, retries: int = 3) -> str:
        """Call LM Studio API with retries."""
        payload = {
            "model": "google/gemma-3-12b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        
        for attempt in range(retries):
            try:
                logger.debug(f"Calling LLM API (attempt {attempt + 1})")
                response = self.session.post(LM_STUDIO_URL, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                logger.debug(f"LLM response: {content[:100]}...")
                return content
                
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.progress_data["failed_requests"] += 1
                    return ""  # Return empty string on final failure
        return None
    
    def rewrite_explanation(self, row: dict[str, str]) -> str:
        """Generate rewritten explanation for a row."""
        # Random factors for variety
        age = random.randint(5, 13)
        
        # Calculate character limit as -50% to +100% of original length
        original_length = len(row["StudentExplanation"])
        percentage_change = random.uniform(-0.50, 1.0)
        char_limit = int(original_length * (1 + percentage_change))
        # Ensure minimum of 20 characters
        char_limit = max(char_limit, 20)
        
        # Random style factors
        writing_style = random.choice(["casual", "formal", "hesitant", "confident", "conversational"])
        
        explanation_focus = random.choice(["process", "visual", "reasoning", "comparison"])
        
        detail_level = random.choice(["brief", "verbose", "rambling"])
        
        math_vocabulary = random.choice(["simple", "mixed", "advanced"])
        
        sentence_structure = random.choice(["simple", "compound", "fragments"])
        
        personal_references = random.choice(["include", "exclude"])
        
        prompt = REWRITE_TEMPLATE.render(
            QuestionText=row["QuestionText"],
            MC_Answer=row["MC_Answer"],
            StudentExplanation=row["StudentExplanation"],
            age=age,
            char_limit=char_limit,
            writing_style=writing_style,
            explanation_focus=explanation_focus,
            detail_level=detail_level,
            math_vocabulary=math_vocabulary,
            sentence_structure=sentence_structure,
            personal_references=personal_references
        )
        
        rewritten = self.call_llm(prompt)
        
        # Fallback if LLM fails
        if not rewritten:
            logger.warning(f"LLM failed for row {row['row_id']}, using original")
            return row["StudentExplanation"]
            
        # Ensure character limit
        if len(rewritten) > char_limit:
            rewritten = rewritten[:char_limit-3] + "..."
            
        return rewritten
    
    def _load_input_data(self) -> list[dict[str, str]]:
        """Load and return input training data."""
        with INPUT_FILE.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def _process_rows(self, original_rows: list[dict[str, str]], writer: csv.DictWriter,
                     progress: Progress, task: int, total_output_rows: int, f) -> None:
        """Process all rows and generate synthetic data."""
        current_row = 0
        
        for orig_row in original_rows:
            for expansion_idx in range(EXPANSIONS_PER_ROW):
                current_row += 1
                
                # Skip if already processed
                if current_row <= self.progress_data["processed_rows"]:
                    continue
                
                # Debug output for first few rows
                if current_row <= DEBUG_ROW_LIMIT:
                    logger.info(f"Processing row {current_row}: {orig_row['row_id']}_{expansion_idx}")
                
                # Create new row with rewritten explanation
                new_row = orig_row.copy()
                new_row["row_id"] = f"{orig_row['row_id']}_{expansion_idx}"
                new_row["StudentExplanation"] = self.rewrite_explanation(orig_row)
                
                writer.writerow(new_row)
                f.flush()  # Flush after each row for real-time visibility
                
                # Update progress
                self.progress_data["processed_rows"] = current_row
                progress.update(task, advance=1)
                
                # Save checkpoint periodically
                if current_row % CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint()
                    f.flush()  # Force write to disk
                    logger.info(f"Progress: {current_row:,}/{total_output_rows:,} rows ({current_row/total_output_rows*100:.1f}%)")

    def generate_synthetic_data(self) -> None:
        """Main generation function."""
        logger.info(f"Starting synthetic data generation: {EXPANSIONS_PER_ROW}x expansion")
        
        # Read input data
        original_rows = self._load_input_data()
        
        total_output_rows = len(original_rows) * EXPANSIONS_PER_ROW
        logger.info(f"Input rows: {len(original_rows):,}")
        logger.info(f"Target output rows: {total_output_rows:,}")
        
        # Always append to output file, write header only if file is new
        write_header = not OUTPUT_FILE.exists()
        
        with OUTPUT_FILE.open("a", encoding="utf-8", newline="") as f:
            fieldnames = list(original_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            # Progress tracking
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Generating synthetic data", total=total_output_rows)
                progress.update(task, advance=self.progress_data["processed_rows"])
                
                self._process_rows(original_rows, writer, progress, task, total_output_rows, f)
        
        # Final cleanup
        self.checkpoint_file.unlink(missing_ok=True)
        logger.success(f"Generation complete! Output: {OUTPUT_FILE}")
        logger.info(f"Total rows generated: {total_output_rows:,}")
        logger.info(f"Failed requests: {self.progress_data['failed_requests']}")


def cleanup() -> None:
    """Clean up generated files and progress."""
    files_to_clean = [
        OUTPUT_FILE,
        Path("synth_checkpoint.json"),
        Path("logs/synth_generation.log")
    ]
    
    console.print("[bold yellow]Cleaning up synthetic data files...[/bold yellow]")
    for file_path in files_to_clean:
        if file_path.exists():
            file_path.unlink()
            console.print(f"Removed: {file_path}")
        else:
            console.print(f"Not found: {file_path}")
    
    console.print("[bold green]Cleanup complete![/bold green]")


def main() -> None:
    """Entry point."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup()
        return
    
    console.print("[bold green]Synthetic Data Generator[/bold green]")
    console.print(f"Input: {INPUT_FILE}")
    console.print(f"Output: {OUTPUT_FILE}")
    console.print(f"Expansion factor: {EXPANSIONS_PER_ROW}x")
    console.print("[dim]Use 'python -m kaggle_map.synth cleanup' to clean files[/dim]")
    
    generator = SynthDataGenerator()
    
    try:
        generator.generate_synthetic_data()
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        console.print("[yellow]Generation stopped. Run again to resume from checkpoint.[/yellow]")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
