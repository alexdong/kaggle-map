"""Database utilities for prompt workbench."""

import sqlite3
from pathlib import Path

from loguru import logger

from kaggle_map.core.models import EvaluationResult
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

DB_PATH = Path("kaggle_map/llm/prompts.db")


def init_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database for prompt history."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT NOT NULL,
            row_ids TEXT NOT NULL,
            evaluation_results TEXT,
            score REAL
        )
    """)
    conn.commit()

    logger.info(f"Database initialized at {db_path}")
    return conn


def save_prompt(conn: sqlite3.Connection, prompt: str, row_ids: list[int]) -> int:
    """Save a prompt to the database."""
    cursor = conn.cursor()
    row_ids_str = ",".join(map(str, row_ids))

    cursor.execute(
        "INSERT INTO prompt_history (prompt, row_ids) VALUES (?, ?)",
        (prompt, row_ids_str)
    )
    conn.commit()

    return cursor.lastrowid


def update_evaluation_results(
    conn: sqlite3.Connection,
    prompt_id: int,
    results: list[EvaluationResult],
    score: float
) -> None:
    """Update evaluation results for a prompt."""
    results_str = "\n".join([
        f"{r.row_id},{r.question_id},{r.score:.3f},{' | '.join(str(p) for p in r.predictions)}"
        for r in results
    ])

    cursor = conn.cursor()
    cursor.execute(
        "UPDATE prompt_history SET evaluation_results = ?, score = ? WHERE id = ?",
        (results_str, score, prompt_id)
    )
    conn.commit()


def get_prompt_by_id(conn: sqlite3.Connection, prompt_id: int) -> dict | None:
    """Get a specific prompt by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prompt_history WHERE id = ?", (prompt_id,))
    row = cursor.fetchone()

    if row:
        return dict(row)
    return None


def get_all_prompts(conn: sqlite3.Connection) -> list[dict]:
    """Get all prompts from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prompt_history ORDER BY id DESC")
    return [dict(row) for row in cursor.fetchall()]


def get_latest_prompt(conn: sqlite3.Connection) -> dict | None:
    """Get the most recent prompt with all its data."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM prompt_history ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()

    if row:
        return dict(row)
    return None
