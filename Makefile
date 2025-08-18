.PHONY: dev test

dev:
	uv run ruff check . --fix --unsafe-fixes
	uv run ruff format .
	uv run ty check .

test:
	# Run tests fast: only dev deps (not full project), with parallelism via addopts
	uv run --only-dev -m pytest
