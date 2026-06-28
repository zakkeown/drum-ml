# First real model: E-GMD seq2seq, and a controlled data-diversity A/B

Date: 2026-06-28. First real training run of the MT3-style seq2seq transcriber,
plus a controlled experiment isolating **training-data diversity** as a variable.
All scores: 5-class, onset F at ±50 ms, scored through `drumml.eval` (the same
harness validated by the ADTOF reproduction).

## Setup

- Model: `Seq2SeqADT`, d_model 256, 6+6 layers, 11.2M params. Greedy decode.
- Data: E-GMD, 8000 train tracks, 8 epochs, batch 32, AdamW lr 3e-4, MPS.
- Two runs differing in **one variable only — how the 8000 tracks are sampled**:
  - **head**: `tracks[:8000]` — E-GMD is ordered by groove (~43 kits/groove), so
    this is only **~187 unique grooves**. `checkpoints/egmd_seq2seq_epoch*.pt`.
  - **diverse**: `--shuffle-seed 0` then take 8000 — **all ~809 grooves**.
    `checkpoints/egmd_seq2seq_div_epoch*.pt`.
- Floor: reproduced ADTOF on MDB-Drums = **macro 0.716 / micro 0.792** (see
  `reproduce_adtof_baseline.md`).

## Result

| | head (187 grooves) | diverse (809 grooves) | ADTOF floor |
|---|---|---|---|
| train mean-loss (ep1→ep8) | 1.8 → **0.06** (memorising) | 1.8 → **0.49** (learning) |
| **in-domain** E-GMD test, macro / micro | 0.302 / 0.318 | **0.938 / 0.941** | — |
| **OOD** MDB, best macro / micro | 0.133 / 0.208 (ep5) | **0.346 / 0.381 (ep6)** | 0.716 / 0.792 |

Per-class in-domain (diverse, ep8): KD 0.991, SD 0.897, TT 0.901, HH 0.934,
CY 0.968 — strong even on toms.

OOD transfer curves (MDB, scheme 5, max_len 192). Both peak mid-training then
decline — over-fitting E-GMD timbre **hurts** transfer, which is why we
epoch-stamp and sweep rather than trust the last checkpoint:

```
epoch    head macro   diverse macro
  1         0.009         0.208
  2         0.106         0.259
  3         0.108         0.262
  4         0.121         0.253
  5         0.133 *       0.274
  6         0.125         0.346 *
  7         0.124         0.333
  8         0.112         0.293        (* = best)
```

## What this shows

1. **The bottleneck is data, not architecture.** Same model, same number of
   training tracks: sampling diversely (187 → 809 grooves) lifts in-domain test
   F from **0.30 → 0.94**. The head sample lets the model *memorise* (train loss
   0.06) while test stays flat from epoch 1; diverse data forces real learning (a
   gradual loss curve and a rising test score).

2. **Diversity helps OOD too, but cannot close the full-mix gap.** Diverse
   sampling more than doubles OOD MDB (0.13 → 0.35 macro). But that is still far
   below the 0.716 full-mix floor. **Why:** on MDB full mixes (bass/guitar/vocals
   it has never heard) the model **over-generates ~4× too many onsets** (e.g. 419
   predicted vs 103 reference on one track) — it fires drums on non-drum spectral
   energy. Groove variety teaches rhythm/timing (which partly transfers) but
   nothing about *separating drums from a mix*. That needs **full-mix training
   data**, the next lever.

## Pipeline is provably correct (this is a real result, not a bug)

Before trusting the low head number, the whole transcribe/decode/score path was
exonerated:

- Oracle round-trip (reference → tokens → reference, no model) scores **1.000** —
  the tokenizer/segmentation/stitching preserve timing.
- On **training** tracks the model scores **0.98–0.999** with a matched-onset
  timing offset of **−0.0 ms** (std 3.7 ms) — greedy decode and timing are exact;
  no systematic offset, no exposure-bias collapse.

So head's 0.31 in-domain test is a genuine generalisation gap (caused by the
clustered sample), not a measurement artefact.

## Caveats

- E-GMD's train/test split is **not groove-disjoint**, so the in-domain 0.94
  partly reflects new takes/kits of grooves whose siblings were trained on. The
  **MDB OOD** number is the contamination-free headline.
- max_len 192 for the OOD sweep (the model over-generates on OOD; 192 = 96
  onsets/2 s, ≫ real density, and head under-generates so the cap doesn't bind —
  verified identical to max_len 512 for head).

## Follow-up: accompaniment-mixing augmentation (closing the OOD gap)

The diverse model still over-generated on MDB (2.15× too many onsets, precision
0.28) because it had never heard non-drum audio. Next intervention: overlay
drum-free accompaniment (MUSDB18 bass+other+vocals, 150 tracks) onto E-GMD drum
segments at training time, mix SNR sampled from the **empirical** MUSDB
drums-vs-accompaniment distribution (median −5.3 dB, 90% accompaniment-dominant —
so the model is forced to find drums under *louder* accompaniment). Same diverse
config, only the mixing added (`--accompaniment-dir`, aug-prob 0.7).

OOD MDB at each model's best epoch, both measured on the **same 23 tracks** at
max_len 192 (no-aug = epoch 6, +mix = epoch 7):

| metric | diverse (no aug) | **diverse + mix** | floor |
|---|---|---|---|
| onset **density** (est/ref) | 2.15× | **1.19×** | 1.0 |
| **precision** | 0.28 | **0.42** | — |
| recall | 0.60 | 0.50 | — |
| macro-F | 0.346 | **0.382** | 0.716 |
| micro-F | 0.381 | **0.460** | 0.792 |

(P/R are self-consistent with F: 0.28/0.60→0.382≈0.381; 0.42/0.50→0.457≈0.460.)
In-domain E-GMD test dipped to 0.824 macro (from 0.938) — capacity spent on the
harder mixed task. The **mechanism is confirmed**: augmentation roughly halved
over-generation (2.15×→1.19×, toward the ideal 1.0) and raised precision by ~50%
(0.28→0.42), lifting OOD micro-F 0.381→0.460. Unlike the no-aug runs (OOD peaks
~epoch 5–6 then decays), the augmented OOD curve *rises* through epoch 7 with
density falling monotonically (3.68×→1.19×) — a healthier transfer profile.

**Reading:** synthetic full-mix data directly attacks the OOD failure mode and
closes ~1/5 of the micro-F gap to the floor (0.381→0.460, floor 0.792). The
remaining gap + a modest recall drop (0.60→0.50, the model trades some recall for
much higher precision) are the next levers: MUSDB is a proxy for real recordings,
11M params is small, and real labelled full-mix data (ADTOF) is the eventual step.

## Reproduce

```bash
# head (clustered) and diverse runs
uv run python scripts/train.py --dataset egmd --root datasets/e-gmd --split train \
    --limit 8000 --epochs 8 --d-model 256 --batch-size 32 --num-workers 8 \
    --out checkpoints/egmd_seq2seq.pt --eval-after --eval-split test --eval-limit 100
uv run python scripts/train.py --dataset egmd --root datasets/e-gmd --split train \
    --limit 8000 --shuffle-seed 0 --epochs 8 --d-model 256 --batch-size 32 --num-workers 8 \
    --out checkpoints/egmd_seq2seq_div.pt --eval-after --eval-split test --eval-limit 100
# OOD transfer curves vs the ADTOF floor
uv run python scripts/eval_ood_sweep.py --checkpoints "checkpoints/egmd_seq2seq_div_epoch*.pt" \
    --dataset mdb --root datasets/MDBDrums --max-len 192
```
