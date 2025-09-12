#!/usr/bin/env python
"""TUI interface for iterating on LLM prompts with evaluation tracking."""

import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from jinja2 import Template
from loguru import logger
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static, TextArea

from kaggle_map.core.models import EvaluationResult
from kaggle_map.dataloader import load_rows_by_ids
from kaggle_map.llm.prompt_db import init_database as init_db
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

DEFAULT_DATA_PATH = Path("datasets/train.csv")
DEFAULT_TEMPLATE_PATH = Path("kaggle_map/llm/prompts/predict.j2")
DB_PATH = Path("kaggle_map/llm/prompts.db")


class VimTextArea(TextArea):
    """TextArea with vim-like keybindings and external editor support."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vim_mode = "insert"  # insert, normal, or visual
        self.visual_start = None
        self.yanked_text = ""

    def open_in_editor(self) -> None:
        """Open current text in external editor."""
        editor = os.environ.get("EDITOR", "vi")

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
            f.write(self.text)
            temp_path = f.name

        try:
            # Open editor and wait for it to close
            subprocess.run([editor, temp_path], check=True)

            # Read back the edited content
            with open(temp_path) as f:
                self.text = f.read()
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def on_key(self, event: events.Key) -> None:
        """Handle vim-like key bindings."""
        if self.vim_mode == "normal":
            if event.key == "i":
                self.vim_mode = "insert"
                self.border_title = "-- INSERT --"
                event.prevent_default()
            elif event.key == "a":
                self.vim_mode = "insert"
                self.border_title = "-- INSERT --"
                self.action_cursor_right()
                event.prevent_default()
            elif event.key == "o":
                # Open line below
                self.vim_mode = "insert"
                self.border_title = "-- INSERT --"
                self.action_cursor_line_end()
                self.insert("\n")
                event.prevent_default()
            elif event.key == "O":
                # Open line above
                self.vim_mode = "insert"
                self.border_title = "-- INSERT --"
                self.action_cursor_line_start()
                self.insert("\n")
                self.action_cursor_up()
                event.prevent_default()
            elif event.key == "v":
                # Visual mode
                self.vim_mode = "visual"
                self.border_title = "-- VISUAL --"
                self.visual_start = self.cursor_location
                event.prevent_default()
            elif event.key == "V":
                # Visual line mode
                self.vim_mode = "visual_line"
                self.border_title = "-- VISUAL LINE --"
                self.action_select_line()
                event.prevent_default()
            elif event.key == "y":
                # Yank current line
                self.action_select_line()
                self.action_copy()
                self.yanked_text = self.selected_text
                event.prevent_default()
            elif event.key == "p":
                # Paste after cursor
                if self.yanked_text:
                    self.insert(self.yanked_text)
                else:
                    self.action_paste()
                event.prevent_default()
            elif event.key == "d":
                # Delete current line
                self.action_select_line()
                self.action_delete_left()
                event.prevent_default()
            elif event.key == "x":
                # Delete character
                self.action_delete_right()
                event.prevent_default()
            elif event.key == "e":
                # Open in external editor (custom keybinding)
                self.open_in_editor()
                event.prevent_default()
            elif event.key == "h":
                self.action_cursor_left()
                event.prevent_default()
            elif event.key == "j":
                self.action_cursor_down()
                event.prevent_default()
            elif event.key == "k":
                self.action_cursor_up()
                event.prevent_default()
            elif event.key == "l":
                self.action_cursor_right()
                event.prevent_default()
            elif event.key == "0":
                self.action_cursor_line_start()
                event.prevent_default()
            elif event.key == "dollar":
                self.action_cursor_line_end()
                event.prevent_default()
            elif event.key == "g":
                self.action_cursor_document_start()
                event.prevent_default()
            elif event.key == "G":
                self.action_cursor_document_end()
                event.prevent_default()
        elif self.vim_mode == "visual":
            if event.key == "escape":
                self.vim_mode = "normal"
                self.border_title = "-- NORMAL --"
                event.prevent_default()
            elif event.key == "y":
                # Yank selection
                self.action_copy()
                self.yanked_text = self.selected_text
                self.vim_mode = "normal"
                self.border_title = "-- NORMAL --"
                event.prevent_default()
            elif event.key == "d":
                # Delete selection
                self.action_delete_left()
                self.vim_mode = "normal"
                self.border_title = "-- NORMAL --"
                event.prevent_default()
            elif event.key in "hjkl":
                # Extend selection
                if event.key == "h":
                    self.action_cursor_left(select=True)
                elif event.key == "j":
                    self.action_cursor_down(select=True)
                elif event.key == "k":
                    self.action_cursor_up(select=True)
                elif event.key == "l":
                    self.action_cursor_right(select=True)
                event.prevent_default()
        elif self.vim_mode == "visual_line":
            if event.key == "escape":
                self.vim_mode = "normal"
                self.border_title = "-- NORMAL --"
                event.prevent_default()
            elif event.key == "y":
                # Yank lines
                self.action_copy()
                self.yanked_text = self.selected_text
                self.vim_mode = "normal"
                self.border_title = "-- NORMAL --"
                event.prevent_default()
            elif event.key == "d":
                # Delete lines
                self.action_delete_left()
                self.vim_mode = "normal"
                self.border_title = "-- NORMAL --"
                event.prevent_default()
        elif self.vim_mode == "insert" and event.key == "escape":
            self.vim_mode = "normal"
            self.border_title = "-- NORMAL --"
            event.prevent_default()


class LoadingScreen(ModalScreen):
    """Modal screen shown while LLM is loading."""

    DEFAULT_CSS = """
    LoadingScreen {
        align: center middle;
    }

    LoadingScreen > Static {
        width: 60;
        height: 7;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Loading LLM model...\nThis may take a moment on first run.")


class PromptWorkbenchApp(App):
    """Textual TUI for prompt workbench with wide-screen layout."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 0.6fr 2fr 2fr;
    }

    .left-panel {
        border: solid $primary;
        margin: 1;
        padding: 1;
    }

    .center-panel {
        border: solid $primary;
        margin: 1;
        padding: 1;
    }

    .right-panel {
        border: solid $primary;
        margin: 1;
        padding: 1;
    }

    .row-ids-area {
        height: 20;
        margin-bottom: 1;
        width: 100%;
    }

    .prompt-area {
        height: 100%;
    }

    .buttons-container {
        height: 8;
        padding: 1;
    }

    .status-bar {
        height: 3;
        background: $boost;
        padding: 1;
        margin-top: 1;
    }

    DataTable {
        height: 100%;
    }

    Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+e", "evaluate", "Evaluate"),
        ("ctrl+s", "save_template", "Save Template"),
        ("ctrl+p", "prev_prompt", "Previous Prompt"),
        ("ctrl+n", "next_prompt", "Next Prompt"),
        ("ctrl+h", "show_help", "Show Help"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.llm = None
        self.stop_tokens = None
        self.conn = None
        self.current_prompt_id = None

    def compose(self) -> ComposeResult:
        """Create the UI layout - left/center/right panels."""
        yield Header()

        # Left panel: Row IDs and controls
        with Vertical(classes="left-panel"):
            yield Label("Row IDs [Ctrl+H for help]:")
            yield VimTextArea(id="row_ids", classes="row-ids-area")
            yield Vertical(
                Button("Evaluate [Ctrl+E]", id="btn_evaluate", variant="primary"),
                Button("Save Template [Ctrl+S]", id="btn_save", variant="success"),
                Button("◀ Previous [Ctrl+P]", id="btn_prev"),
                Button("Next ▶ [Ctrl+N]", id="btn_next"),
                classes="buttons-container"
            )
            yield Static("Ready", id="status", classes="status-bar")

        # Center panel: Prompt template editor
        with Vertical(classes="center-panel"):
            yield Label("Prompt Template [Ctrl+H for vim help]:")
            yield VimTextArea(id="prompt_template", classes="prompt-area")

        # Right panel: Results table
        with ScrollableContainer(classes="right-panel"):
            yield Label("Evaluation Results:")
            yield DataTable(id="results_table")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app after mounting."""
        self.push_screen(LoadingScreen())
        await self.init_resources()
        self.pop_screen()

        # Load default template
        if DEFAULT_TEMPLATE_PATH.exists():
            template_area = self.query_one("#prompt_template", VimTextArea)
            template_area.text = DEFAULT_TEMPLATE_PATH.read_text()

        # Load last used row IDs if available
        if self.conn:
            latest = self._get_latest_prompt()
            if latest and latest.get("row_ids"):
                row_ids_area = self.query_one("#row_ids", VimTextArea)
                row_ids_list = latest["row_ids"].split(",")
                row_ids_area.text = "\n".join(row_ids_list)

        # Initialize results table
        table = self.query_one("#results_table", DataTable)
        table.add_columns("Row", "Answer", "Explanation", "Truth", "Predictions", "Score")
        table.zebra_stripes = True

    async def init_resources(self) -> None:
        """Initialize database and LLM."""
        # Initialize database
        self.conn = self._init_database()

        # Initialize LLM
        logger.info("Initializing LLM instance...")
        config = GGUFModelLoadConfig(
            model_name=GGUFModelName.GEMMA_3_12B_IT,
            quantization=GGUFModelQuantizationLevel.Q4_K_XL,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False,
        )
        self.llm = load_llm_model(config)
        self.stop_tokens = get_stop_tokens(config.model_name)
        logger.success("LLM instance initialized")

    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database."""
        return init_db(DB_PATH)

    def _save_prompt(self, prompt: str, row_ids: list[int]) -> int:
        """Save prompt to database."""
        cursor = self.conn.cursor()
        row_ids_str = ",".join(map(str, row_ids))
        cursor.execute(
            "INSERT INTO prompt_history (prompt, row_ids) VALUES (?, ?)",
            (prompt, row_ids_str)
        )
        self.conn.commit()
        return cursor.lastrowid

    def _update_evaluation_results(self, prompt_id: int, results: list[EvaluationResult], score: float) -> None:
        """Update evaluation results in database."""
        results_str = "\n".join([
            f"{r.row_id},{r.question_id},{r.score:.3f},{' | '.join(str(p) for p in r.predictions)}"
            for r in results
        ])

        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE prompt_history SET evaluation_results = ?, score = ? WHERE id = ?",
            (results_str, score, prompt_id)
        )
        self.conn.commit()

    def _get_latest_prompt(self) -> dict | None:
        """Get most recent prompt."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM prompt_history ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return dict(row) if row else None

    def _get_adjacent_prompt(self, current_id: int, direction: str) -> dict | None:
        """Get adjacent prompt."""
        cursor = self.conn.cursor()
        if direction == "prev":
            cursor.execute(
                "SELECT * FROM prompt_history WHERE id < ? ORDER BY id DESC LIMIT 1",
                (current_id,)
            )
        else:
            cursor.execute(
                "SELECT * FROM prompt_history WHERE id > ? ORDER BY id ASC LIMIT 1",
                (current_id,)
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def _update_status(self, message: str, error: bool = False) -> None:
        """Update status bar."""
        status = self.query_one("#status", Static)
        if error:
            status.update(f"❌ {message}")
        else:
            status.update(message)

    def _update_results_table(self, results: list[EvaluationResult]) -> None:
        """Update the results table with evaluation results."""
        table = self.query_one("#results_table", DataTable)
        table.clear()

        explanation_max_len = 40
        for r in results:
            predictions_str = " | ".join(str(p) for p in r.predictions[:3])
            explanation = (
                r.explanation[:explanation_max_len] + "..."
                if len(r.explanation) > explanation_max_len
                else r.explanation
            )

            table.add_row(
                str(r.row_id),
                r.mc_answer[:20],
                explanation,
                str(r.ground_truth)[:30],
                predictions_str,
                f"{r.score:.2f}"
            )

    @work(thread=True)
    async def action_evaluate(self) -> None:
        """Handle evaluate action."""
        await self._evaluate()

    @on(Button.Pressed, "#btn_evaluate")
    async def handle_evaluate_button(self) -> None:
        """Handle evaluate button press."""
        await self._evaluate()

    async def _evaluate(self) -> None:
        """Run evaluation with current inputs."""
        row_ids_area = self.query_one("#row_ids", VimTextArea)
        prompt_area = self.query_one("#prompt_template", VimTextArea)

        row_ids_text = row_ids_area.text.strip()
        prompt_text = prompt_area.text.strip()

        if not row_ids_text or not prompt_text:
            self._update_status("Need both row IDs and prompt template", error=True)
            return

        try:
            row_ids = [int(line.strip()) for line in row_ids_text.split("\n") if line.strip()]
        except ValueError:
            self._update_status("Invalid row IDs (numbers only)", error=True)
            return

        if not row_ids:
            self._update_status("No valid row IDs", error=True)
            return

        self._update_status(f"⏳ Evaluating {len(row_ids)} rows...")

        # Save prompt
        prompt_id = self._save_prompt(prompt_text, row_ids)
        self.current_prompt_id = prompt_id

        try:
            # Load data and evaluate
            df = load_rows_by_ids(DEFAULT_DATA_PATH, row_ids)
            template = Template(prompt_text)
            results, avg_score = evaluate_dataframe(df, template, self.llm, self.stop_tokens)

            # Update database
            self._update_evaluation_results(prompt_id, results, avg_score)

            # Update UI
            self._update_results_table(results)
            self._update_status(f"✓ Prompt #{prompt_id} | MAP@3: {avg_score:.4f} | {len(results)} rows")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self._update_status(f"Failed: {e!s}", error=True)

    @on(Button.Pressed, "#btn_save")
    async def handle_save_button(self) -> None:
        """Handle save template button press."""
        await self._save_template()

    async def action_save_template(self) -> None:
        """Save current template to file."""
        await self._save_template()

    async def _save_template(self) -> None:
        """Save template to file."""
        prompt_area = self.query_one("#prompt_template", VimTextArea)
        prompt_text = prompt_area.text.strip()

        if not prompt_text:
            self._update_status("Cannot save empty template", error=True)
            return

        try:
            DEFAULT_TEMPLATE_PATH.write_text(prompt_text)
            self._update_status("✓ Template saved to predict.j2")
        except Exception as e:
            self._update_status(f"Save failed: {e!s}", error=True)

    @on(Button.Pressed, "#btn_prev")
    async def handle_prev_button(self) -> None:
        """Handle previous button press."""
        await self._navigate("prev")

    async def action_prev_prompt(self) -> None:
        """Navigate to previous prompt."""
        await self._navigate("prev")

    @on(Button.Pressed, "#btn_next")
    async def handle_next_button(self) -> None:
        """Handle next button press."""
        await self._navigate("next")

    async def action_next_prompt(self) -> None:
        """Navigate to next prompt."""
        await self._navigate("next")

    async def _navigate(self, direction: str) -> None:
        """Navigate to adjacent prompt."""
        if not self.current_prompt_id:
            latest = self._get_latest_prompt()
            if latest:
                self.current_prompt_id = latest["id"]
            else:
                self._update_status("No prompt history", error=True)
                return

        prompt_data = self._get_adjacent_prompt(self.current_prompt_id, direction)

        if not prompt_data:
            self._update_status(f"No more prompts ({direction})", error=True)
            return

        # Update UI with historical data
        self.current_prompt_id = prompt_data["id"]

        # Update row IDs
        row_ids_area = self.query_one("#row_ids", VimTextArea)
        row_ids_list = prompt_data["row_ids"].split(",")
        row_ids_area.text = "\n".join(row_ids_list)

        # Update prompt template
        prompt_area = self.query_one("#prompt_template", VimTextArea)
        prompt_area.text = prompt_data["prompt"]

        # Update status
        if prompt_data["score"] is not None:
            self._update_status(
                f"📝 Prompt #{prompt_data['id']} | MAP@3: {prompt_data['score']:.4f} | "
                f"{prompt_data['timestamp'][:19]}"
            )

            # Clear table for historical prompt
            table = self.query_one("#results_table", DataTable)
            table.clear()
        else:
            self._update_status(f"📝 Prompt #{prompt_data['id']} (not evaluated)")

    async def action_show_help(self) -> None:
        """Show vim keybindings help."""
        self._update_status("Vim Help: " + " | ".join([
            "ESC=normal", "i/a=insert", "v=visual", "y/p=copy/paste",
            "d=delete", "e=$EDITOR", "hjkl=move"
        ]))


def main() -> None:
    """Entry point for the TUI application."""
    app = PromptWorkbenchApp()
    app.run()


if __name__ == "__main__":
    main()

