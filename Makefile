.PHONY: dev test

dev:
	uv run ruff check . --fix --unsafe-fixes
	uv run ruff format .
	uv run ty check .

test:
	uv run python -m pytest --lf
