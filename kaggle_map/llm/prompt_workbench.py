"""Web interface for iterating on LLM prompts with evaluation tracking."""

import csv
import io
import sqlite3
from pathlib import Path

from fasthtml.common import *
from jinja2 import Template
from loguru import logger

from kaggle_map.core.models import EvaluationResult
from kaggle_map.dataloader import load_rows_by_ids
from kaggle_map.llm.utils import evaluate_dataframe
from kaggle_map.utils.gguf_model import (
    GGUFModelLoadConfig,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    get_stop_tokens,
    load_llm_model,
)
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

LLM_INSTANCE = None
LLM_STOP_TOKENS = None
DEFAULT_DATA_PATH = Path("datasets/train.csv")
DEFAULT_TEMPLATE_PATH = Path("kaggle_map/llm/prompts/predict.j2")
DB_PATH = Path("kaggle_map/llm/prompts.db")


def init_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database with prompt history table."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable column access by name

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
    """Save a prompt to the database and return its ID."""
    cursor = conn.cursor()
    row_ids_str = ",".join(map(str, row_ids))

    cursor.execute(
        "INSERT INTO prompt_history (prompt, row_ids) VALUES (?, ?)",
        (prompt, row_ids_str)
    )
    conn.commit()

    prompt_id = cursor.lastrowid
    logger.debug(f"Saved prompt {prompt_id} with {len(row_ids)} row IDs")
    return prompt_id


def update_evaluation_results(
    conn: sqlite3.Connection,
    prompt_id: int,
    results: list[EvaluationResult],
    score: float
) -> None:
    """Update a prompt record with evaluation results."""
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["row_id", "question_id", "score", "predictions"])

    for result in results:
        predictions_str = " | ".join([str(p) for p in result.predictions])
        writer.writerow([result.row_id, result.question_id, result.score, predictions_str])

    csv_data = csv_buffer.getvalue()

    cursor = conn.cursor()
    cursor.execute(
        "UPDATE prompt_history SET evaluation_results = ?, score = ? WHERE id = ?",
        (csv_data, score, prompt_id)
    )
    conn.commit()

    logger.debug(f"Updated prompt {prompt_id} with score {score:.4f}")


def get_prompt_by_id(conn: sqlite3.Connection, prompt_id: int) -> dict | None:
    """Retrieve a prompt by its ID."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM prompt_history WHERE id = ?",
        (prompt_id,)
    )
    row = cursor.fetchone()

    if row:
        return dict(row)
    return None


def get_all_prompts(conn: sqlite3.Connection) -> list[dict]:
    """Get all prompts ordered by timestamp."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, score FROM prompt_history ORDER BY timestamp DESC"
    )
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


def get_adjacent_prompt(conn: sqlite3.Connection, current_id: int, direction: str) -> dict | None:
    """Get the previous or next prompt relative to current_id."""
    if direction == "prev":
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM prompt_history WHERE id < ? ORDER BY id DESC LIMIT 1",
            (current_id,)
        )
    else:  # next
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM prompt_history WHERE id > ? ORDER BY id ASC LIMIT 1",
            (current_id,)
        )

    row = cursor.fetchone()
    if row:
        return dict(row)
    return None


def init_llm():
    """Initialize the global LLM instance."""
    global LLM_INSTANCE, LLM_STOP_TOKENS

    if LLM_INSTANCE is None:
        logger.info("Initializing LLM instance...")

        config = GGUFModelLoadConfig(
            model_name=GGUFModelName.GEMMA_3_12B_IT,
            quantization=GGUFModelQuantizationLevel.Q4_K_XL,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False,
        )

        LLM_INSTANCE = load_llm_model(config)
        LLM_STOP_TOKENS = get_stop_tokens(config.model_name)

        logger.success("LLM instance initialized successfully")

    return LLM_INSTANCE, LLM_STOP_TOKENS


def create_results_table(results: list[EvaluationResult]) -> str:
    """Generate HTML table from evaluation results."""
    if not results:
        return "<p>No results to display</p>"

    rows = []
    for r in results:
        predictions_str = " | ".join([str(p) for p in r.predictions])
        rows.append(
            Tr(
                Td(str(r.row_id)),
                Td(r.mc_answer),
                Td(r.explanation[:80] + "..." if len(r.explanation) > 80 else r.explanation),
                Td(str(r.ground_truth)),
                Td(predictions_str),
                Td(f"{r.score:.2f}"),
            )
        )

    return Table(
        Thead(
            Tr(
                Th("Row ID"),
                Th("MC Answer"),
                Th("Explanation"),
                Th("Ground Truth"),
                Th("Predictions"),
                Th("Score"),
            )
        ),
        Tbody(*rows),
        cls="results-table"
    )


app, rt = fast_app(
    live=True,
    hdrs=(
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"),
        Style("""
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .input-group { margin-bottom: 20px; }
            textarea { width: 100%; font-family: monospace; }
            .buttons { display: flex; gap: 10px; margin: 20px 0; }
            .navigation { display: flex; align-items: center; gap: 20px; margin: 20px 0; }
            .results-table { width: 100%; border-collapse: collapse; }
            .results-table th, .results-table td { padding: 8px; border: 1px solid #ddd; }
            .results-table th { background-color: #f4f4f4; }
            .score-display { font-size: 1.2em; font-weight: bold; color: #0066cc; }
        """)
    )
)

# Initialize lazily to avoid issues when running as module
conn = None
llm = None
stop_tokens = None

def ensure_initialized():
    """Ensure database and LLM are initialized."""
    global conn, llm, stop_tokens
    if conn is None:
        conn = init_database()
    if llm is None:
        llm, stop_tokens = init_llm()
    return conn, llm, stop_tokens


@rt("/")
def get():
    """Main page with prompt editor."""
    conn, _, _ = ensure_initialized()
    template_content = DEFAULT_TEMPLATE_PATH.read_text() if DEFAULT_TEMPLATE_PATH.exists() else ""

    latest_prompt = get_latest_prompt(conn)
    default_row_ids = ""
    if latest_prompt and latest_prompt.get("row_ids"):
        row_ids_list = latest_prompt["row_ids"].split(",")
        default_row_ids = "\n".join(row_ids_list)
        logger.debug(f"Restored {len(row_ids_list)} row IDs from last session")

    return Div(
        H1("Prompt Workbench"),
        Div(
            Div(
                Label("Row IDs (one per line):"),
                Textarea(
                    id="row-ids",
                    rows=5,
                    value=default_row_ids,
                    placeholder="Enter row IDs, one per line\ne.g.,\n1\n2\n3"
                ),
                cls="input-group"
            ),
            Div(
                Label("Prompt Template:"),
                Textarea(
                    id="prompt-template",
                    rows=15,
                    value=template_content,
                    placeholder="Enter Jinja2 template..."
                ),
                cls="input-group"
            ),
            Div(
                Button("Evaluate", hx_post="/evaluate", hx_target="#results", hx_include="#row-ids,#prompt-template"),
                Button("Save Template", hx_post="/save-template", hx_target="#save-status", hx_include="#prompt-template"),
                cls="buttons"
            ),
            Div(
                id="navigation",
                cls="navigation"
            ),
            Div(id="save-status"),
            Div(id="results"),
            cls="container"
        )
    )


@rt("/evaluate")
async def post(request):
    """Evaluate the prompt with selected rows."""
    conn, llm, stop_tokens = ensure_initialized()
    form = await request.form()
    row_ids_text = form.get("row-ids", "").strip()
    prompt_template = form.get("prompt-template", "").strip()

    if not row_ids_text or not prompt_template:
        return Div(P("Please provide both row IDs and prompt template", style="color: red;"))

    try:
        row_ids = [int(line.strip()) for line in row_ids_text.split("\n") if line.strip()]
    except ValueError:
        return Div(P("Invalid row IDs. Please enter numbers only.", style="color: red;"))

    if not row_ids:
        return Div(P("No valid row IDs provided", style="color: red;"))

    prompt_id = save_prompt(conn, prompt_template, row_ids)

    try:
        df = load_rows_by_ids(DEFAULT_DATA_PATH, row_ids)
        template = Template(prompt_template)
        results, avg_score = evaluate_dataframe(df, template, llm, stop_tokens)
        update_evaluation_results(conn, prompt_id, results, avg_score)
        table_html = create_results_table(results)
        
        nav_html = Div(
            Button("<", hx_get=f"/navigate?id={prompt_id}&dir=prev", hx_target="#results"),
            Span(f"Prompt #{prompt_id}"),
            Button(">", hx_get=f"/navigate?id={prompt_id}&dir=next", hx_target="#results"),
            Span(f"Score: {avg_score:.4f}", cls="score-display"),
            cls="navigation"
        )

        return Div(
            nav_html,
            H3(f"Results (Average MAP@3: {avg_score:.4f})"),
            table_html
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return Div(P(f"Evaluation failed: {e!s}", style="color: red;"))


@rt("/save-template")
async def post(request):
    """Save the current prompt template to file."""
    conn, _, _ = ensure_initialized()
    form = await request.form()
    prompt_template = form.get("prompt-template", "").strip()

    if not prompt_template:
        return Div(P("Cannot save empty template", style="color: red;"))

    try:
        DEFAULT_TEMPLATE_PATH.write_text(prompt_template)
        return Div(P("Template saved successfully!", style="color: green;"))
    except Exception as e:
        return Div(P(f"Failed to save template: {e!s}", style="color: red;"))


@rt("/navigate")
def get(request):
    """Navigate to adjacent prompts."""
    conn, _, _ = ensure_initialized()
    prompt_id = int(request.query_params.get("id", 0))
    direction = request.query_params.get("dir", "next")

    prompt_data = get_adjacent_prompt(conn, prompt_id, direction)

    if not prompt_data:
        return Div(P("No more prompts in this direction", style="color: orange;"))

    [int(x) for x in prompt_data["row_ids"].split(",")]

    if prompt_data["score"] is not None:
        nav_html = Div(
            Button("<", hx_get=f"/navigate?id={prompt_data['id']}&dir=prev", hx_target="#results"),
            Span(f"Prompt #{prompt_data['id']}"),
            Button(">", hx_get=f"/navigate?id={prompt_data['id']}&dir=next", hx_target="#results"),
            Span(f"Score: {prompt_data['score']:.4f}", cls="score-display"),
            cls="navigation"
        )

        return Div(
            nav_html,
            H3(f"Historical Results (Score: {prompt_data['score']:.4f})"),
            P(f"Timestamp: {prompt_data['timestamp']}"),
            P(f"Row IDs: {prompt_data['row_ids']}"),
            Pre(prompt_data["prompt"][:500] + "..." if len(prompt_data["prompt"]) > 500 else prompt_data["prompt"])
        )
    return Div(P("This prompt hasn't been evaluated yet", style="color: orange;"))


if __name__ == "__main__":
    import uvicorn
    # Use uvicorn directly to avoid reloader issues
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=False)
