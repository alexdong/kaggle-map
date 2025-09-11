"""Run the prompt workbench web interface."""

from kaggle_map.llm.prompt_workbench import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)