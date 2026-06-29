# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` covers contribution conventions (style, commits, PRs); this file covers architecture and commands. `README.md` has the user-facing tour and `ADT_PIPELINE_2026.md` the design thesis.

## What this project is

An automatic drum transcription (ADT) system. The guiding thesis is that the
bottleneck is **data + evaluation, not architecture** — so the codebase is built
measurement-first. The headline metric is **cross-dataset / OOD macro-F at ±50 ms**,
not in-domain F1 (which is saturated). Recent work is about closing the
synthetic-EGMD → real-full-mix domain gap (accompaniment-mixing augmentation).

## Commands

```bash
uv sync                      # create .venv (Python 3.12) + install everything (incl. torch)
uv run pytest                # full suite (dataset-free; MERT test skips without transformers)
uv run pytest tests/test_eval.py            # one module
uv run pytest tests/test_eval.py::test_name # one test

# end-to-end smoke (real audio, ~50 tracks) — validates train→checkpoint→transcribe→score
uv run python scripts/train.py --dataset egmd --root datasets/e-gmd \
    --split train --limit 50 --epochs 2 --d-model 128 \
    --out checkpoints/smoke.pt --eval-after --eval-split test --eval-limit 20

# score a saved checkpoint (self-describing — only needs the .pt)
uv run python scripts/evaluate.py --checkpoint checkpoints/seq2seq.pt \
    --dataset egmd --root datasets/e-gmd --split test
```

Other entry points in `scripts/`: `run_eval.py` (score a prediction dir),
`run_adtof_baseline.py` (the ADTOF floor baseline), `eval_ood_sweep.py`
(cross-dataset transfer curve), `prep_accompaniment.py` + `probe_snr_robustness.py`
(augmentation data prep / probes).

## Architecture

Two layers, split by dependency weight. The split is load-bearing — don't pull torch
into the core.

**Measurement core** (`numpy` + `mir_eval` + `pretty_midi` only):
- `taxonomy.py` — the single source of truth for drum vocabulary. One canonical
  14-class enum keyed on the General MIDI percussion map, with deterministic
  reductions to `3`/`5`/`8`/`canonical` schemes (`SCHEMES`, `reduce()`,
  `scheme_classes()`). **Train rich, evaluate collapsed** — never retrain per
  benchmark. Any new drum-class logic belongs here, centralized.
- `events.py` — `DrumEvent` / `DrumAnnotation`, the common representation passed
  between every component, plus TSV I/O. Velocity is first-class.
- `eval.py` — mir_eval onset-F at `DEFAULT_WINDOW` (±50 ms); per-class →
  `TrackScore` → `DatasetScore` (`score_track`, `aggregate`,
  `cross_dataset_macro_f`, `format_report`).
- `tokenize.py` — MT3-style event ↔ token codec (vocab 214 @ scheme=5).
- `data/*.py` adapters all yield the same `DrumAnnotation` via the `DatasetAdapter`
  ABC (`data/base.py`): `egmd.py` (exact, GM MIDI notes), `mdb.py` (MDB-Drums label
  table), `adtof.py` (ADTOF label/prediction parser).
- `baselines/adtof_port.py` — runs the external ADTOF baseline into canonical output.

**Training stack** (needs the `model` extra: torch/torchaudio/soundfile — keep
behind lazy imports):
- `features.py` — `LogMel` and `MERTFrontend` (lazy) audio front-ends.
- `model/seq2seq.py` — MT3/T5-style encoder-decoder (`Seq2SeqADT`, `Seq2SeqConfig`,
  ~32M params at default config); `greedy_decode` for inference.
- `data/torch_dataset.py` — segmenting `Dataset` + collate.
- `data/augment.py` — `AccompanimentMixer` / `MixingFrontend`: mixes drum stems with
  MUSDB accompaniment to simulate full-mix audio (the OOD-gap-closing augmentation).
- `train.py` — teacher-forced training loop. `transcribe.py` — model + audio →
  `DrumAnnotation`, the bridge from inference back into `eval`.
- `checkpoint.py` — checkpoints are **self-describing**: they carry model config +
  tokenizer params (`FORMAT_VERSION`, `save_checkpoint`/`load_checkpoint`), so
  `evaluate.py` reconstructs everything from the `.pt`.

The canonical data flow: **adapter → `DrumAnnotation` → (tokenizer ↔ model) →
`transcribe` → `eval`**. Everything funnels through `DrumAnnotation` and the
taxonomy, which is what lets one trained model be scored against any benchmark.

## Conventions specific here

- Apple Silicon GPU (MPS) is the default training device; data loading is
  multi-worker.
- Keep tests synthetic and dataset-free; large artifacts live in `datasets/`,
  `checkpoints/`, `runs/` (git-ignored).
- Heavy/conflicting deps (torch, ADTOF runtime, audio libs) stay optional or in
  their own environments behind lazy imports — the eval core must stay importable
  with only the three core deps.
