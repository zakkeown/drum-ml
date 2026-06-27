# A Competitive 2026-Era Automatic Drum Transcription Pipeline

*Design sketch — scoped 2026-06-27. Opinionated single spine; variants flagged inline.*

## 0. The one thing that matters

ADT's bottleneck is **not architecture** — the model family is settled (encoder–decoder transformer, MIDI-token output). The bottleneck is **labeled real-music data with fine drum classes and velocity**. So the competitive edge of this design is the **training-data union** and the **out-of-domain evaluation protocol**, not the network. Everything below leads from that.

## 1. Scope: three constraints the task doesn't state, resolved

| Constraint | Decision | Why |
|---|---|---|
| **DTM vs DTD** (drums-in-mix vs isolated drums) | **Target DTM** (full mix) | Real use is full-mix. The synthetic-data SOTA paper's DTD-only eval was a *flagged limitation*; DTM is what justifies the separation front-end and real-mix data. |
| **Commercial vs research** | **Default research-grade**, with a one-line commercial swap (§8) | The strongest stack (MERT, LarsNet, ADTOF weights) is all CC-BY-NC. Repo is empty → assume research scoping until told otherwise. |
| **Offline vs real-time** | **Default offline** | "Competitive" = accuracy. Real-time/few-shot (Fraunhofer 2024) is a separate design point, noted but not pursued. |

## 2. The spine

```
                          full-mix audio (44.1k mono/stereo)
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                        │
        [A] log-mel             [B] MERT-v1-330M        [C] HT-Demucs ── drums stem
        128 bins, 10ms          intermediate layers      (optional branch)
              │                  + learned layer-weights         │
              └───────────┬───────────┘                          │
                          ▼                                LarsNet → {kick,snare,
              ┌───────────────────────┐                    toms,hihat,cymbals}
              │  Transformer ENCODER  │                          │
              │  (mel ⊕ MERT fusion)  │                   post-hoc velocity +
              └───────────┬───────────┘                   crash/ride split
                          ▼                                       │
              ┌───────────────────────┐                          │
              │  Transformer DECODER  │◄── teacher forcing        │
              │  autoregressive       │                           │
              │  MIDI-token vocab     │                           │
              │  (time · class · vel) │                           │
              └───────────┬───────────┘                          │
                          ▼                                       │
                 drum events (onset, class, velocity) ◄───────────┘
                          │                      (stem branch refines
                          ▼                       cymbal class + velocity)
        canonical 14-class internal taxonomy
                          │
        deterministic collapse maps → {8 | 5 | 3}-class for eval
```

**Primary track (build this):** A + B → encoder → decoder. Reproducible today (code exists: `magenta/mt3`, `mimbres/YourMT3`, `pier-maker92/ADT_STR`).

**R&D track (higher ceiling, later):** swap the decoder for an **N2N-style diffusion head** (audio-conditioned generative refinement). No released code/weights and delicate loss engineering — treat as a research bet, not the baseline. But **port its two transferable wins into the primary track now**: (1) MERT intermediate features for OOD robustness, (2) joint onset+velocity modeling.

## 3. Component decisions (one-line rationale each)

| Component | Choice | Rationale |
|---|---|---|
| **Front-end 1** | Log-mel, 128 bins, 10 ms hop | Standard; cheap; onset-precise grid. |
| **Front-end 2** | **MERT-v1-330M**, learned weighting over hidden layers (not last-only) | Finest SSL grid (75 Hz/13 ms); N2N ablation showed MERT features buy large OOD gains (MDB 82 vs 71, IDMT 90 vs 81 F1). Lower layers = timbre, higher = semantics → learn the mix. |
| **Backbone** | Encoder–decoder transformer, T5/MT3-style, RoPE | Field consensus (MT3, YourMT3+, 2025 AMT Challenge winner all use it). Reproducible. |
| **Output** | MIDI-token seq2seq: `[time-shift (10ms grid) · drum-class · velocity]` triplets, ~2 s segments | MT3/YourMT3+ paradigm + **per-onset velocity** (re-added, as the 2026 drum work does). Velocity is now table-stakes (E-GMD, OaF, N2N all model it). |
| **Stem branch** *(optional)* | HT-Demucs (drums stem) → **LarsNet** (per-piece) → post-hoc velocity + crash/ride split | Only way to expand cymbal granularity + get reliable velocity. **Caveats:** LarsNet is CC-BY-NC, validated only on *synthetic* StemGMD, and its hi-hat/cymbal SDR is weak (4–6 dB) → cymbal velocity will be noisy. Riley & Dixon's simpler post-hoc RMS-from-stem velocity is the validated fallback. |
| **Taxonomy** | Train on a **rich 14-class** internal vocab; collapse deterministically to 8/5/3 at eval | Train rich, eval at any granularity — exactly what STAR/N2N/ADT_STR do. Avoids re-training per benchmark. |

## 4. The data plan (the actual differentiator)

No single dataset is enough. Real-music + fine-class + velocity exists only in fragments. **Union them, unify the label spaces, and balance with on-the-fly synthesis.**

| Source | ~Hours | Real? | Classes | Velocity | Role in the union | License |
|---|---|---|---|---|---|---|
| **E-GMD** | 444 | Electronic kit (TD-17) | ~9 | **Yes** | Velocity supervision + volume | CC BY 4.0 ✅ |
| **STAR Drums** | 124 | Hybrid (synth drums + real mix) | 18→8/5/3 | from MIDI | DTM realism, fine classes, 48 kHz | CC BY 4.0 ✅ |
| **ADTOF** (RGW+YT) | ~359 | **Real commercial music** | 5 | No | Real-world DTM robustness | CC BY-NC-SA, request-gated ⚠ |
| **CLAP-curated synthetic** | ∞ (on-the-fly) | Synthetic | up to 26 | controllable | Class balance, rare-instrument coverage | depends on samples |
| **ENST / MDB / IDMT** | ~3 total | Real | 3–21 | partial | **Held-out test only** (never train) | mixed ⚠ |

**Taxonomy unification** is the load-bearing engineering: map every source's labels into one canonical 14-class set via explicit many-to-fewer tables, keyed on the **GM percussion map** (notes 35–81). Then deterministic collapse maps produce 8/5/3-class targets for whichever benchmark you're scoring.

**On-the-fly synthesis** (per the 2026 SOTA recipe): curate a one-shot drum-sample library by auto-labeling unlabeled samples with **CLAP audio-embedding centroids** (cosine similarity to ~1.4k hand-seeded samples), then render drum tracks from MIDI (Lakh) at train time with per-class interpolation augmentation. Critically — and where that paper fell short — **render into real backing tracks** (use STAR/MUSDB non-drum stems) so the model trains for **DTM**, not drums-only.

**Sample-rate hygiene:** sources span 16/44.1/48 kHz. Resample everything to a single rate (44.1 kHz) at ingestion.

## 5. Training recipe

- **Loss:** cross-entropy on tokens (primary). If you adopt the diffusion head, use N2N's **Annealed Pseudo-Huber loss** — it interpolates MSE→MAE so onset errors stop swamping velocity regression.
- **Augmentation:** time-stretch / tempo jitter (the 2026 paper's *omission* of this was a flagged weakness), pitch/EQ, additive backing-track mixing, sample-library interpolation.
- **Curriculum:** pretrain on the large synthetic + E-GMD union → fine-tune on ADTOF + STAR (real/hybrid) to close the synthetic-to-real gap.
- **Scale reference:** N2N hit SOTA with a ~50M-param model, 4×A100, ~1 day. This is not a frontier-scale problem — iteration speed matters more than size.

## 6. Evaluation (where "competitive" is won)

In-domain F1 is **saturated and misleading** — a 2019 CRNN already scores high there. The headline number must be generalization:

- **Headline metric: cross-dataset / OOD onset-F at ±50 ms** (`mir_eval.onset`, per-class P/R/F, macro-averaged). Train on one corpus, test on the held-out others (e.g. train E-GMD+STAR+synthetic → test ENST, MDB, IDMT, RBMA-if-recoverable). This is exactly the axis N2N's MERT ablation improved — it's what separates this design from a CRNN.
- **Velocity:** report velocity-aware F (E-GMD has the labels) — the differentiator most systems skip.
- **Granularity:** report at 3 / 5 / 8 classes so numbers are comparable to *every* prior paper despite their inconsistent taxonomies.
- **DTM vs DTD split:** report both, but lead with **DTM** (full-mix) since that's the target.

## 7. Build order

1. **Reproduce a baseline.** Stand up ADTOF (PyTorch port `xavriley/ADTOF-pytorch`) — real-music, 5-class, weights available. This is your floor.
2. **MT3-style seq2seq + log-mel**, trained on E-GMD+STAR, eval cross-dataset. Confirm you match/beat ADTOF.
3. **Add MERT front-end** with learned layer-weighting. Measure the OOD delta (this is the thesis).
4. **Add velocity tokens + CLAP-curated synthetic into real backing tracks.** Push DTM numbers.
5. **(Optional) stem branch** for cymbal granularity. **(R&D) diffusion head.**

## 8. Proven vs. novel — the honest line

- **Proven individually:** MT3 tokenization; MERT for music tasks; CLAP for sample curation (the 2026 paper); LarsNet for *synthetic* drum separation; velocity modeling (E-GMD/N2N).
- **Novel / unproven:** *No published system demonstrates a foundation-model front-end as a drum-transcription backbone with reported per-drum F-measures, in DTM, with velocity.* That composition is the **risk and the opportunity** simultaneously. N2N validated MERT-for-drums in a diffusion setting; nobody has shown it in a clean reproducible seq2seq DTM system. That gap is the contribution.

**Commercial swap (if needed):** replace MERT → **MusicFM** (MIT weights, the only permissive music-SSL, and the only one with *transcription* evidence — 2025 AMT Challenge winner, though pitched-only); drop LarsNet; train your own separator/velocity on permissive data (E-GMD/STAR/MUSDB). Note MusicFM's 25 Hz/40 ms grid needs the decoder to recover onset precision, and its *training-data* license is a separate question from its MIT weights.

## 9. Key references

- **Noise-to-Notes** (diffusion + MERT, ICASSP 2026) — arXiv 2509.21739
- **Towards Realistic Synthetic Data for ADT** (seq2seq + CLAP curation, Jan 2026) — arXiv 2601.09520, code `github.com/pier-maker92/ADT_STR`
- **STAR Drums** (hybrid dataset, TISMIR 2025) — `10.5334/tismir.244`, Zenodo `10.5281/zenodo.15690078`
- **Enhanced ADT via Drum Stem Separation** (Riley & Dixon, 2025) — arXiv 2509.24853
- **Synthetic-to-real gap in ADT** (Zehren et al., 2024) — arXiv 2407.19823
- **ADTOF** — `github.com/MZehren/ADTOF`; PyTorch port `github.com/xavriley/ADTOF-pytorch`
- **E-GMD / OaF-Drums** — arXiv 2004.00188; `magenta.tensorflow.org/oaf-drums`
- **MT3 / YourMT3+** — arXiv 2111.03017 / 2407.04822
- **MERT** — arXiv 2306.00107 (`m-a-p/MERT-v1-330M`); **MusicFM** — arXiv 2311.03318 (`minzwon/MusicFM`, MIT)
- **HT-Demucs** — arXiv 2211.08553; **LarsNet** — arXiv 2312.09663 (`github.com/polimi-ispl/larsnet`)
- **Eval** — `mir_eval` (onset ±50 ms); GM percussion map (notes 35–81)
