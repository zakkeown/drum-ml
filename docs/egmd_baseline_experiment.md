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

## Follow-up 2: the recall drop is a *timbre/domain* gap, not a masking gap (negative result)

Per-class decomposition of the recall drop (MDB, no-aug ep6 → +mix ep7) localizes
it: hihat recall fell **0.567→0.382 with no precision gain** (HH precision flat at
~0.71), and kick fell 0.797→0.699 (also no precision gain) — whereas snare's loss
(0.718→0.554) *bought* precision (0.20→0.41, killing a 3.5× false-positive flood).
So the **recoverable** recall is HH (and some KD); SD's loss is the intended effect.
More epochs does **not** recover HH (ep7→ep8 pooled recall rises 0.503→0.527 but
F falls 0.460→0.448 — it just slides back along the P/R trade; HH recall stays
~0.37). So HH suppression is structural to the augmentation, not undertraining.

The natural hypothesis was **over-suppression under loud accompaniment** (MUSDB's
empirical SNR reaches −20 dB; clip the inaudible tail to recover recall). A
fixed-SNR probe on E-GMD **test** drums (exact labels, so masking is isolated from
domain shift — `scripts/probe_snr_robustness.py`) **falsifies** it:

```
              HH recall vs fixed mix-SNR (60 E-GMD test tracks)
  SNR     clean   +10    +5     0     -5    -10    -15
  +mix    0.677  0.668  0.661  0.665  0.652  0.628  0.589   <- ~FLAT (robust)
  no-aug  0.836  0.775  0.725  0.658  0.593  0.538   --     <- collapses
```

The +mix model's hihat recall is **nearly flat across a 25 dB SNR range**
(0.677→0.589): the augmentation *succeeded* at making detection masking-robust
(the no-aug model collapses 0.836→0.538 over the same range).

**The decisive comparison — synthetic vs real transfer — is what names the gap.**
Line each model's real-MDB recall up against its *matched-SNR synthetic* recall:

```
                MDB (real)   E-GMD-test mixed @ -5/-10 dB   transfers?
  no-aug HH R      0.567            0.593 / 0.538            YES  (real == synthetic)
  +mix   HH R      0.382       >=0.589 at EVERY SNR (-15!)   NO   (real << worst synthetic)
```

The no-aug model **transfers**: its MDB hihat recall (0.567) is exactly what its
synthetic performance at MDB's SNR regime predicts (0.54–0.59). The +mix model
**does not**: on real audio (0.382) it is *worse than its worst synthetic SNR*
(0.589 at −15 dB). So the deficit is **+mix-specific** — the augmentation's
robustness is keyed to the exact synthetic distribution (E-GMD drums *digitally
summed* with MUSDB stems) and evaporates on a real produced mix. This is not a
generic drum-timbre gap; it is **sim-to-real overfitting of the augmentation
itself** — the model learned to find drums under MUSDB, not under a real mix.
(The "timbre" intuition is one bundled factor among digital-sum-vs-produced-mix,
accompaniment realism, and groove novelty.) It even cost a little in-domain hihat
recall (clean E-GMD-test HH 0.84→0.68) by tightening the detector to the synthetic
distribution.

**Consequence:** the recall drop **cannot be recovered by tuning the mix SNR**
(floor-clip, distribution shift, or aug-prob — the latter only slides the same P/R
trade, as ep8 showed). More to the point, the data implicate the *synthetic-ness*
of the augmentation, so the levers split by confidence:

* **Real labelled full-mix data (ADTOF)** — high-confidence. It is the only lever
  that attacks what the evidence actually implicates (the sim-to-real gap).
* **Drum-audio augmentation** (EQ / pitch-shift / reverb / codec on isolated drums,
  digitally summed) — cheap, but it is *the same class of synthetic intervention
  that just failed to transfer*. Worth a bet, but gate it behind the same
  `probe_snr_robustness`-style E-GMD-test-vs-real check *before* committing a
  ~100-min run; do not assume it transfers.

(Methodological note: the per-class diagnosis was read off MDB, so MDB has acted as
a dev set here. The floor was *not* tuned on MDB — the probe is on E-GMD test — but
a clean cross-dataset headline for any intervention needs a fresh full-mix set,
i.e. ADTOF, not MDB.)

## Follow-up 3: the *fine-tune regime* is the lever; real ≈ synthetic data in it

Follow-up 2 showed the from-scratch synthetic-mixing model overfit its own mix
distribution (sim-to-real gap). The test here: bring in **real** labelled full-mix
data and measure transfer — but with a regime control that turned out to flip the
conclusion. Real source = **A2MD** (1565 internet songs with drum labels from
DTW-aligned Lakh MIDI), tightest buckets `dist<=0.10` (197 clips, ~4.4h). A2MD
labels are *weak* (a separate arrangement aligned to the recording), so we first
**bounded their quality**: the ADTOF model scores micro **0.782** against `dist0p00`
labels and **0.887** against `dist0p10` labels, vs **0.796** against clean MDB —
i.e. tight-bucket A2MD labels are ~as trustworthy as MDB hand labels, so a weak
result can't be blamed on label noise.

Method: **fine-tune the diverse E-GMD model** (`div_epoch6`), lr 1e-4, epoch-stamped,
bare front-end, scored at **scheme 3** (A2MD is 3-class; aux percussion maps
PERC→drop) vs a re-derived scheme-3 ADTOF floor (**micro 0.796 / macro 0.800**).
Crucially we run **two** fine-tunes from the *same* checkpoint — one on real A2MD,
one on the *synthetic* MUSDB mixing — so "real vs synthetic" is isolated from
"fine-tune vs from-scratch".

| MDB held-out, scheme 3, best epoch | micro-F | macro-F | density | P | R |
|---|---|---|---|---|---|
| diverse E-GMD (no full-mix data) | 0.442 | 0.513 | 2.07× | 0.33 | 0.68 |
| + synthetic MUSDB mixing, **from scratch** | 0.505 | 0.514 | 1.06× | 0.49 | 0.52 |
| + synthetic MUSDB mixing, **fine-tune** (control) | 0.620 | **0.635** | 1.07× | 0.60 | 0.64 |
| + **real** A2MD data, **fine-tune** | **0.624** | 0.620 | 1.21× | 0.57 | **0.69** |
| ADTOF floor (scheme 3) | 0.796 | 0.800 | 1.0 | — | — |

**The dominant lever is the fine-tune regime, not the data source.** Gentle
adaptation from the converged `div_epoch6` (low LR, few epochs) lifts micro-F
**0.505 → ~0.62** for *both* data sources — a +0.115 jump. Head-to-head in that
regime, **real A2MD (0.624) and synthetic mixing (0.620) are a tie** (synthetic is
even marginally higher on macro). An earlier draft of this section claimed real
data "beats synthetic ~3:1 (+0.119)" — that was a **regime confound** (real
*fine-tune* vs synthetic *from-scratch*) and is **retracted**; the real-vs-synthetic
effect, properly controlled, is ≈0.

Why does the *same* synthetic mixing that overfit from-scratch (0.505, Follow-up 2)
now reach 0.620? Because the gentle fine-tune regime never over-trains on the
synthetic distribution — it adapts the already-general model a little, rather than
letting it memorize MUSDB seams over 8 from-scratch epochs. The regime *fixes* the
sim-to-real overfitting that Follow-up 2 diagnosed.

Two real (if modest) edges for real data remain: it reaches the tie with **~4.5×
fewer segments** (7.8k vs 35k — more data-efficient), and it holds **higher recall**
(0.69 vs 0.64) at slightly lower precision. Both models push the P/R frontier well
past the from-scratch +mix (P 0.49 / R 0.52). The OOD optimum is early for both
(real ep2, synthetic ep3) then declines.

Caveats: (1) **scheme 3 only** — fine-tuning on 3-class A2MD made the model forget
toms/cymbals (scheme-5 TT 0.000, CY 0.083, macro 0.389); recoverable by co-training
with E-GMD's 5-class labels. (2) Still **~0.17 micro below the floor**. (3) The
synthetic fine-tune used more (E-GMD+MUSDB) segments than A2MD; matched-data and
co-training comparisons are the natural follow-ups.

**Revised reading for the "do we pivot to synthesis?" question:** *more* favorable
to synthesis, not less. Synthetic mixing, used in the right (fine-tune) regime,
**equals** our scarce real data on transfer — and synthetic is infinitely scalable.
So synthesis is re-validated as a co-equal lever, and the likely best path is
**hybrid** (real to anchor + for efficiency, synthetic for scale), both via gentle
fine-tuning, with realism improvements (Zehren-style) on the synthetic side.

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
# Falsification probe: HH recall vs fixed mix-SNR (isolates masking from domain shift)
uv run python scripts/probe_snr_robustness.py \
    --checkpoints checkpoints/egmd_seq2seq_mix_epoch7.pt checkpoints/egmd_seq2seq_div_epoch6.pt \
    --accompaniment-dir datasets/musdb_accompaniment --egmd-root datasets/e-gmd --n-tracks 60
# Follow-up 3: fine-tune the diverse model on REAL A2MD full-mix data, sweep on MDB (scheme 3)
uv run python scripts/train.py --dataset a2md --root datasets/a2md/a2md_public --a2md-max-dist 0.10 \
    --init-from checkpoints/egmd_seq2seq_div_epoch6.pt --epochs 8 --batch-size 32 \
    --num-workers 8 --lr 1e-4 --out checkpoints/egmd_a2md_ft.pt
uv run python scripts/eval_ood_sweep.py --checkpoints "checkpoints/egmd_a2md_ft_epoch*.pt" \
    --dataset mdb --root datasets/MDBDrums --scheme 3 --max-len 192
# Regime control: fine-tune the SAME checkpoint on synthetic mixing (isolates real-vs-synthetic)
uv run python scripts/train.py --dataset egmd --root datasets/e-gmd --split train \
    --shuffle-seed 0 --limit 2000 --init-from checkpoints/egmd_seq2seq_div_epoch6.pt \
    --accompaniment-dir datasets/musdb_accompaniment --aug-prob 0.7 \
    --epochs 4 --batch-size 32 --num-workers 8 --lr 1e-4 --out checkpoints/egmd_synth_ft.pt
uv run python scripts/eval_ood_sweep.py --checkpoints "checkpoints/egmd_synth_ft_epoch*.pt" \
    --dataset mdb --root datasets/MDBDrums --scheme 3 --max-len 192
```
