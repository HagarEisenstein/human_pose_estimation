.PHONY: install install-dev lint test eval download-mini clean

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# ── Quality ──────────────────────────────────────────────────────────────────
lint:
	ruff check . --fix
	ruff format .

typecheck:
	mypy data pose eval segmentation

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

# ── Data ─────────────────────────────────────────────────────────────────────
# Download a tiny COCO val subset (500 images + annotations) to data/raw/
download-mini:
	python -m data.download --subset 500

# ── Evaluation ───────────────────────────────────────────────────────────────
eval:
	python -m eval.runner --data-root data/raw --split val2017 --subset 100

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage mlruns
