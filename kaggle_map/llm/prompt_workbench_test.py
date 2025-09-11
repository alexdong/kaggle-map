"""Tests for prompt workbench web interface."""



from kaggle_map.core.models import Category, EvaluationResult, Prediction
from kaggle_map.llm.prompt_workbench import (
    get_all_prompts,
    get_latest_prompt,
    get_prompt_by_id,
    init_database,
    save_prompt,
    update_evaluation_results,
)


def test_database_initialization(tmp_path):
    """Test database is created with correct schema."""
    db_path = tmp_path / "test_prompts.db"
    conn = init_database(db_path)

    # Check table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_history'")
    assert cursor.fetchone() is not None

    # Check columns
    cursor.execute("PRAGMA table_info(prompt_history)")
    columns = [col[1] for col in cursor.fetchall()]
    assert "id" in columns
    assert "timestamp" in columns
    assert "prompt" in columns
    assert "row_ids" in columns
    assert "evaluation_results" in columns
    assert "score" in columns

    conn.close()


def test_save_prompt(tmp_path):
    """Test saving a prompt to database."""
    db_path = tmp_path / "test_prompts.db"
    conn = init_database(db_path)

    prompt_text = "Question: {{ question_text }}\nAnswer: {{ mc_answer }}"
    row_ids = [1, 2, 3]

    prompt_id = save_prompt(conn, prompt_text, row_ids)

    assert prompt_id > 0

    # Verify saved data
    cursor = conn.cursor()
    cursor.execute("SELECT prompt, row_ids FROM prompt_history WHERE id = ?", (prompt_id,))
    result = cursor.fetchone()
    assert result[0] == prompt_text
    assert result[1] == "1,2,3"

    conn.close()


def test_update_evaluation_results(tmp_path):
    """Test updating evaluation results for a prompt."""
    db_path = tmp_path / "test_prompts.db"
    conn = init_database(db_path)

    # Save initial prompt
    prompt_id = save_prompt(conn, "Test prompt", [1, 2])

    # Create evaluation results
    results = [
        EvaluationResult(
            row_id=1,
            question_id=100,
            mc_answer="A",
            explanation="Test explanation",
            ground_truth=Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
            predictions=[Prediction(category=Category.TRUE_CORRECT, misconception="NA")],
            score=1.0
        ),
        EvaluationResult(
            row_id=2,
            question_id=101,
            mc_answer="B",
            explanation="Another test",
            ground_truth=Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Division"),
            predictions=[Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Division")],
            score=1.0
        )
    ]

    update_evaluation_results(conn, prompt_id, results, 1.0)

    # Verify update
    cursor = conn.cursor()
    cursor.execute("SELECT evaluation_results, score FROM prompt_history WHERE id = ?", (prompt_id,))
    result = cursor.fetchone()
    assert result[0] is not None  # CSV results stored
    assert result[1] == 1.0

    conn.close()


def test_get_prompt_by_id(tmp_path):
    """Test retrieving a prompt by ID."""
    db_path = tmp_path / "test_prompts.db"
    conn = init_database(db_path)

    # Save prompt
    prompt_text = "Test prompt template"
    row_ids = [5, 10, 15]
    prompt_id = save_prompt(conn, prompt_text, row_ids)

    # Retrieve prompt
    prompt_data = get_prompt_by_id(conn, prompt_id)

    assert prompt_data["id"] == prompt_id
    assert prompt_data["prompt"] == prompt_text
    assert prompt_data["row_ids"] == "5,10,15"

    conn.close()


def test_navigation_between_prompts(tmp_path):
    """Test navigating through prompt history."""
    db_path = tmp_path / "test_prompts.db"
    conn = init_database(db_path)

    # Save multiple prompts
    save_prompt(conn, "Prompt 1", [1])
    id2 = save_prompt(conn, "Prompt 2", [2])
    save_prompt(conn, "Prompt 3", [3])

    # Get all prompts
    all_prompts = get_all_prompts(conn)
    assert len(all_prompts) == 3

    # Test navigation
    prompt2 = get_prompt_by_id(conn, id2)
    assert prompt2["prompt"] == "Prompt 2"

    conn.close()


def test_get_latest_prompt(tmp_path):
    """Test retrieving the most recent prompt."""
    db_path = tmp_path / "test_prompts.db"
    conn = init_database(db_path)

    # Initially no prompts
    latest = get_latest_prompt(conn)
    assert latest is None

    # Save multiple prompts with different row_ids
    save_prompt(conn, "First prompt", [1, 2, 3])
    save_prompt(conn, "Second prompt", [4, 5, 6])
    save_prompt(conn, "Latest prompt", [7, 8, 9, 10])

    # Get latest prompt
    latest = get_latest_prompt(conn)
    assert latest is not None
    assert latest["prompt"] == "Latest prompt"
    assert latest["row_ids"] == "7,8,9,10"

    conn.close()


def test_row_ids_persistence(tmp_path):
    """Test that row_ids are correctly saved and restored."""
    db_path = tmp_path / "test_prompts.db"
    conn = init_database(db_path)

    # Save prompt with specific row_ids
    row_ids = [100, 200, 300, 400, 500]
    save_prompt(conn, "Test prompt", row_ids)

    # Retrieve latest prompt and verify row_ids
    latest = get_latest_prompt(conn)
    assert latest["row_ids"] == "100,200,300,400,500"

    # Verify row_ids can be parsed back to list
    restored_ids = [int(x) for x in latest["row_ids"].split(",")]
    assert restored_ids == row_ids

    conn.close()
