.PHONY: dev test test-coverage web serve

dev:
	uv run ruff check . --fix --unsafe-fixes
	uv run ruff format .
	uv run ty check .

test:
	uv run pytest --lf

test-coverage:
	uv run pytest --cov=. --cov-report=html --cov-report=term --duration=5 

web:
	uv run python -m kaggle_map.web

serve:
	./tools/run_with_tunnel.sh
