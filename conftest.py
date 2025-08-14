import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from ingest import memory


@pytest.fixture(scope="session", autouse=True)
def _in_memory_db() -> Iterator[None]:
    """
    Share one SQLite in-memory database across the whole test run.
    The URI form + `cache=shared` ensures every new sqlite3 connection
    opened by the code under test points to *the same* DB instance.
    """
    # Create a temporary file that will be deleted when the session ends
    temp_fd, temp_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_fd)  # Close the file descriptor, we only need the path

    shared_memory_db_uri = f"file:{temp_path}?cache=shared"
    with memory.use_db(shared_memory_db_uri):
        yield

    if Path(temp_path).exists():
        Path(temp_path).unlink()
