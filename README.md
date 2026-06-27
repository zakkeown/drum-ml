# drum-ml

Toward a competitive 2026-era **automatic drum transcription (ADT)** system.
Design rationale lives in [`ADT_PIPELINE_2026.md`](ADT_PIPELINE_2026.md); its
thesis is that the bottleneck is **data + evaluation, not architecture**.

This repo starts at **step 1 of the build order**: a verifiable core for
*measuring* ADT before training one.

## What's here

```
src/drumml/
  taxonomy.py        canonical 14-class vocab + deterministic 3/5/8-class reductions
  events.py          DrumEvent / DrumAnnotation — the common representation + TSV I/O
  eval.py            mir_eval onset-F (per-class macro/micro) + cross-dataset (OOD) aggregation
  data/              dataset adapters that normalize each corpus onto the taxonomy
    egmd.py            E-GMD (exact, via GM MIDI notes)
    mdb.py             MDB-Drums (label-table based; VERIFY tokens vs repo)
    adtof.py           ADTOF-style label/prediction parser
  baselines/
    adtof_port.py    run the ADTOF floor baseline -> canonical output
scripts/run_eval.py  CLI: score a prediction dir against a dataset
docs/                reproduce_adtof_baseline.md
tests/               fast, dataset-free unit tests (synthetic fixtures)
```

## Setup

```bash
uv sync                 # creates .venv (Python 3.12), installs deps + package
uv run pytest           # run the test suite
```

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

1. ✅ Taxonomy + eval harness + adapters (this scaffold).
2. Reproduce the **ADTOF baseline** floor → `docs/reproduce_adtof_baseline.md`.
3. MT3-style seq2seq on E-GMD + STAR; confirm it matches/beats ADTOF cross-dataset.
4. Add the **MERT front-end** with learned layer-weighting; measure the OOD delta.
5. Velocity tokens + CLAP-curated synthetic into real backing tracks (DTM).
