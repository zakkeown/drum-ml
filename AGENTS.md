# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package for automatic drum transcription. Core code lives in `src/drumml/`: taxonomy, event I/O, tokenization, evaluation, checkpoints, training, and transcription. Dataset adapters are in `src/drumml/data/`, baseline integrations in `src/drumml/baselines/`, and seq2seq code in `src/drumml/model/`.

Command-line and experiment helpers live in `scripts/`; design notes live in `docs/` and `ADT_PIPELINE_2026.md`. Tests are in `tests/` and are fast and dataset-free unless noted. Large local artifacts belong in `datasets/`, `checkpoints/`, or `runs/`.

## Build, Test, and Development Commands

- `uv sync`: create the virtual environment and install dev dependencies, including the optional model stack.
- `uv run pytest`: run the full configured pytest suite.
- `uv run pytest tests/test_eval.py`: run one focused test module.
- `uv run python scripts/train.py --dataset egmd --root datasets/e-gmd --limit 50 --epochs 2 --out checkpoints/smoke.pt`: run a small training smoke test.
- `uv run python scripts/evaluate.py --checkpoint checkpoints/seq2seq.pt --dataset egmd --root datasets/e-gmd --split test`: evaluate a saved checkpoint.

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, concise docstrings where useful, and type hints for public interfaces. Prefer small dataclasses and pure functions for the torch-free measurement core. Keep heavyweight dependencies behind extras and lazy imports.

Use lowercase module names, `snake_case` functions and variables, `PascalCase` classes, and `test_*` test functions. Keep canonical drum vocabulary and reduction behavior centralized in `taxonomy.py`.

## Testing Guidelines

Pytest is configured in `pyproject.toml` with `tests/` as the test root and quiet output. Add tests beside related coverage, for example `tests/test_tokenize.py` for tokenizer changes or `tests/test_torch_dataset.py` for batching. Prefer synthetic fixtures over real datasets. For model or audio changes, include a smoke-level assertion that verifies shapes, decoding, or end-to-end behavior.

## Commit & Pull Request Guidelines

Git history uses short, imperative, result-oriented commit messages, such as `Add --shuffle-seed for diverse track sampling` or `Correct no-aug OOD baseline: measure density/P/R on all 23 tracks`. Follow that style and keep commits focused.

Pull requests should describe the behavioral change, list validation commands run, note any required datasets or checkpoints, and link related docs or issues. Include metrics or before/after numbers for training, evaluation, or OOD changes.

## Security & Configuration Tips

Do not commit local datasets, checkpoints, run outputs, or secrets. Treat third-party dataset paths and external baseline repositories as local configuration, and document reproducibility assumptions in `docs/`.
