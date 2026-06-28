# drum-ml

Toward a competitive 2026-era **automatic drum transcription (ADT)** system.
Design rationale lives in [`ADT_PIPELINE_2026.md`](ADT_PIPELINE_2026.md); its
thesis is that the bottleneck is **data + evaluation, not architecture**.

This repo starts at **step 1 of the build order**: a verifiable core for
*measuring* ADT before training one.

## What's here

```
src/drumml/
  # --- measurement core (torch-free) ---
  taxonomy.py        canonical 14-class vocab + deterministic 3/5/8-class reductions
  events.py          DrumEvent / DrumAnnotation — the common representation + TSV I/O
  eval.py            mir_eval onset-F (per-class macro/micro) + cross-dataset (OOD) aggregation
  tokenize.py        MT3-style event <-> token codec (vocab 214 @ scheme=5)
  data/
    egmd.py            E-GMD (exact, via GM MIDI notes)
    mdb.py             MDB-Drums (label-table based; VERIFY tokens vs repo)
    adtof.py           ADTOF-style label/prediction parser
  baselines/
    adtof_port.py    run the ADTOF floor baseline -> canonical output
  # --- training stack (needs the `model` extra: torch) ---
  features.py        LogMel + MERT (lazy) audio front-ends
  model/seq2seq.py   MT3/T5-style encoder-decoder (~32M @ default config)
  data/torch_dataset.py  segmenting torch Dataset + collate
  train.py           teacher-forced training loop
  transcribe.py      model + audio -> DrumAnnotation (inference bridge → eval)
scripts/
  run_eval.py        CLI: score a prediction dir against a dataset
  train.py           CLI: train a seq2seq transcriber from a dataset adapter
docs/                reproduce_adtof_baseline.md, step3_contracts.md
tests/               fast, dataset-free unit tests (synthetic fixtures)
```

## Setup

```bash
uv sync                 # creates .venv (Python 3.12), installs everything + the package
uv run pytest           # full suite (55 tests; MERT test skips without `transformers`)
```

`uv sync` installs the full dev environment (incl. torch) so all tests run. The
*runtime* deps are split so eval-only users stay light:

```bash
pip install drumml            # torch-free core: taxonomy, events, eval, tokenizer
pip install drumml[model]     # + training stack (torch, torchaudio, soundfile)
pip install drumml[model,mert]  # + MERT foundation-model front-end
```

## Running a training / eval experiment

Once a dataset is on disk (e.g. E-GMD at `datasets/e-gmd`), the full
train → checkpoint → transcribe → score loop is one command. Start with a tiny
**smoke run** to validate end-to-end on real audio:

```bash
# smoke: 50 tracks, small model, then score 20 held-out tracks
uv run python scripts/train.py --dataset egmd --root datasets/e-gmd \
    --split train --limit 50 --epochs 2 --d-model 128 \
    --out checkpoints/smoke.pt --eval-after --eval-split test --eval-limit 20
```

Then a real run, and score a saved checkpoint separately:

```bash
uv run python scripts/train.py --dataset egmd --root datasets/e-gmd --epochs 20
uv run python scripts/evaluate.py --checkpoint checkpoints/seq2seq.pt \
    --dataset egmd --root datasets/e-gmd --split test
```

Checkpoints are self-describing (carry model config + tokenizer params), so
`evaluate.py` needs only the `.pt` file. Cross-dataset (OOD) scoring = train on
one corpus, evaluate against another (`drumml.eval.cross_dataset_macro_f`).

## Design decisions baked in

- **One canonical taxonomy, collapsed per benchmark.** Train rich; evaluate at 3,
  5, or 8 classes against whichever dataset — no per-benchmark retraining.
- **Headline metric is cross-dataset / OOD macro-F at ±50 ms.** In-domain F1 is
  saturated; generalization is what distinguishes a 2026 system from a 2019 CRNN.
- **Velocity is first-class** in the event representation (kept for E-GMD).
- **Heavy/conflicting deps stay out.** `torch`, the ADTOF runtime, and audio libs
  are optional or run in their own environments; the eval core needs only
  `numpy` + `mir_eval` + `pretty_midi`.

## Next steps (from the build order)

1. ✅ Taxonomy + eval harness + adapters.
2. ⬜ Reproduce the **ADTOF baseline** floor → `docs/reproduce_adtof_baseline.md` (needs external repo + data).
3. ✅ MT3-style seq2seq **stack scaffolded & unit-tested end-to-end** (tokenizer, LogMel/MERT front-ends, model, dataset, train loop, **inference bridge** `transcribe` → eval). Encoder-uses-audio verified by an overfit-then-greedy-decode test. ⬜ Still to do: train on real E-GMD + STAR and confirm it matches/beats ADTOF cross-dataset.
4. ◑ **MERT front-end** wrapper present (`features.MERTFrontend`); ⬜ learned layer-weighting + fusion + OOD-delta measurement.
5. ⬜ Velocity tokens (tokenizer supports them) + CLAP-curated synthetic into real backing tracks (DTM).
