.PHONY: install install-dev lint test eval eval-save \
        figures pipeline-figures graphs comparison clean download-mini

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
# Download COCO val subset + DensePose annotations + images
download-mini:
	python -m data.download --subset 500 --densepose

# ── Evaluation ───────────────────────────────────────────────────────────────
# Quick check (prints to stdout only)
eval:
	python -m eval.runner --data-root data/raw --split val2017 --subset 100

# Full evaluation: run on all available DensePose samples, save JSON for reuse
eval-save:
	python -m eval.runner --data-root data/raw --split val2017 \
		--out results/eval_oracle.json

# ── Figures & Graphs (M2 deliverables) ───────────────────────────────────────
# 50 gallery figures: image | part-mask | GT skeleton
figures:
	python notebooks/visualize_samples.py --n 50 --out outputs/figures

# 50 pipeline figures: image | part-mask | raw joints | assembled skeleton
pipeline-figures:
	python notebooks/visualize_pipeline.py --n 50 --out outputs/pipeline_figures

# Per-joint PCK bar chart (re-runs evaluation unless results JSON is present)
graphs:
	python notebooks/plot_per_joint_pck.py \
		--results-json results/eval_oracle.json \
		--out outputs/graphs

# Comparison table + grouped bar chart vs HRNet-W32
comparison:
	python notebooks/comparison_table.py \
		--results-json results/eval_oracle.json \
		--out outputs/graphs

# Run ALL M2 output steps in sequence
m2-outputs: eval-save figures pipeline-figures graphs comparison

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage mlruns
