from datetime import datetime
from pathlib import Path


def create_timestamped_filename(base_name: str, extension: str) -> str:
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    return f"{timestamp}_{base_name}{extension}"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
