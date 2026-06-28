# Reproducing the ADTOF baseline (the floor)

ADTOF (Zehren et al.) is our reproducible floor: a real-music, 5-class CRNN with
released weights. We run the **PyTorch port** — `github.com/xavriley/ADTOF-pytorch`,
the *same model* as the original, within ~0.2% F — because it depends only on
`torch`/`librosa`/`pretty_midi` and ships its weights in-tree, avoiding the
original's TensorFlow + git-`madmom` + Cython stack. Its predictions are piped
into `drumml`'s eval harness via `drumml.baselines.adtof_port` /
`drumml.data.adtof`.

## 1. Install the port (isolated environment)

Keep its deps out of drumml's env; install it as a standalone tool so the
`adtof` command lands on PATH:

```bash
uv tool install "git+https://github.com/xavriley/ADTOF-pytorch"
adtof --help     # confirm it's on PATH
```

Its CLI is `adtof --audio IN.wav --out OUT.mid --device cpu`. **Note the default
device is `cuda`** — pass `--device cpu` on a Mac. Output is a MIDI file with
onsets at the 5-class pitches `LABELS_5 = [35, 38, 47, 42, 49]` =
kick / snare / tom / hihat / cymbal (the cymbal class fuses crash+ride).
`drumml.data.adtof.annotation_from_adtof_midi` parses that into our taxonomy.

## 2. Test set: MDB-Drums (public, full-mix audio bundled)

```bash
git clone --depth 1 https://github.com/CarlSouthall/MDBDrums datasets/MDBDrums
```

23 tracks, audio included (44.1 kHz). The checkout nests everything under a
top-level `MDB Drums/` folder (literal space); `MDBDrumsAdapter` handles that and
the `_class`/`_MIX` filename-suffix join. Verified reference: **7994 onsets**
across 6 class tokens `{KD 1539, SD 2654, HH 2639, TT 90, CY 1002, OT 70}`; at
scheme 5 the OT/percussion drops (matching ADTOF's no-OT output), leaving 7924.

ENST and RBMA (the other published ADTOF test sets) are gated, so MDB is the one
clean, openly reproducible floor. E-GMD is a *different regime* (isolated-kit, no
published ADTOF number) — not part of reproducing this baseline.

## 3. Run + score (one command)

```bash
uv run python scripts/run_adtof_baseline.py --root datasets/MDBDrums
```

This transcribes MDB full-mix with ADTOF, writes MIDI + canonical TSV under
`runs/adtof/mdb/`, scores at scheme 5, prints a per-class spot-check (predicted
vs reference onset counts for one track) and the aggregate report, and compares
to the published band.

## 4. The bar to reproduce

Published 5-class onset F at ±50 ms on **standard** MDB-Drums (the "SUM"/global F
— TP/FP/FN pooled across classes and tracks): **~0.76–0.81** depending on which
bundled checkpoint (ADTOF-RGW-only ≈0.76; trained-on-all-five ≈0.81). Treat the
whole band as success and use the port's **default per-class thresholds** —
don't tune.

> **Which metric:** that SUM convention is our report's **micro-F**, *not* the
> macro headline. MDB has only ~90 tom onsets, so macro-F (equal class weighting)
> reads much lower than micro and would falsely look like a failed reproduction.
> Compare **micro-F** to 0.76–0.81; the **macro-F** is the separate floor our own
> model must later clear.

> **Don't confuse with MDBDrums++:** the port's own README quotes ~0.88, but that
> is on `xavriley/MDBDrumsPlusPlus`, a *corrected* re-annotation. Against the
> standard MDB-Drums annotations here, ~0.76–0.81 is the right target.

That ADTOF micro-F is the floor every later iteration (our MT3-style seq2seq,
then + MERT) must clear.

## Result (reproduced 2026-06-28)

ADTOF-pytorch on standard MDB-Drums full-mix, 23 tracks, scheme 5, ±50 ms:

```
class     ref    est     tp      F
KD       1539   1609   1335   0.848
SD       2654   1861   1582   0.701
TT         90    238     57   0.348
HH       2639   2327   2110   0.850
CY       1002   1101    876   0.833
--------------------------------------
micro-F (SUM, the comparable metric)   0.792   -> WITHIN 0.76-0.81
macro-F (pooled)                       0.716
macro-F (per-track mean)               0.775
```

**micro-F 0.792 reproduces the published baseline.** Per-class is sensible:
kick/hihat/cymbal ~0.83–0.85, snare 0.70; toms are the weak spot (F 0.35, only
90 reference onsets and the model over-predicts them — e.g. 62 false toms on the
sparse ride-driven 80sRock track). No systematic class swap: dataset-wide HH est
(2327) tracks HH ref (2639), confirming the tom/hihat pitch assignment is
correct. **This 0.792 micro-F is our floor.**
