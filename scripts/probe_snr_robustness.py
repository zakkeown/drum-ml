#!/usr/bin/env python
"""Probe a checkpoint's detection-vs-mix-SNR curve on E-GMD test (exact labels).

Mix isolated E-GMD **test** drums with MUSDB accompaniment at a sweep of *fixed*
SNRs and score. Because the drum labels are exact, this isolates the model's
detection-under-masking behavior from any held-out *domain* shift -- it answers
"does this model lose onsets purely because accompaniment is loud?" separately
from "does it lose onsets because the test drums sound different?".

Used to falsify the SNR-floor-clip hypothesis for the accompaniment-augmented
model: its hihat recall is ~flat across SNR (robust), so the residual recall
deficit on real MDB full-mixes is a *domain/timbre* gap, not a masking gap --
tuning the mix SNR cannot recover it. See docs/egmd_baseline_experiment.md.

Example:
    uv run python scripts/probe_snr_robustness.py \\
        --checkpoints checkpoints/egmd_seq2seq_mix_epoch7.pt \\
                      checkpoints/egmd_seq2seq_div_epoch6.pt \\
        --accompaniment-dir datasets/musdb_accompaniment \\
        --egmd-root datasets/e-gmd --n-tracks 60
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _perclass(scores, scheme):
    counts: dict[str, list[int]] = {}
    for t in scores:
        for cls, cs in t.per_class.items():
            a = counts.setdefault(cls, [0, 0, 0])
            a[0] += cs.n_ref
            a[1] += cs.n_est
            a[2] += cs.tp
    out = {}
    for cls in scheme:
        if cls in counts:
            ref, est, tp = counts[cls]
            out[cls] = (tp / est if est else 0.0, tp / ref if ref else 0.0, est, ref)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--accompaniment-dir", type=Path, default=Path("datasets/musdb_accompaniment"))
    ap.add_argument("--egmd-root", type=Path, default=Path("datasets/e-gmd"))
    ap.add_argument("--n-tracks", type=int, default=60)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--snrs", nargs="+", type=float, default=[10, 5, 0, -5, -10, -15],
                    help="fixed mix SNRs (dB); a 'clean' no-mix row is always included")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    from drumml.checkpoint import load_checkpoint
    from drumml.data.augment import AccompanimentMixer, MixingFrontend
    from drumml.data.egmd import EGMDAdapter
    from drumml.eval import score_track
    from drumml.features import LogMelFrontend
    from drumml.taxonomy import scheme_classes
    from drumml.train import pick_device
    from drumml.transcribe import transcribe_dataset

    device = pick_device(args.device)
    tracks = list(EGMDAdapter(args.egmd_root, split="test").tracks())
    random.Random(args.shuffle_seed).shuffle(tracks)
    tracks = tracks[: args.n_tracks]
    print(f"device {device} | {len(tracks)} E-GMD test tracks | floor probe")
    base = LogMelFrontend()
    snr_rows = [None, *args.snrs]

    for path in args.checkpoints:
        model, tok, _ = load_checkpoint(path, device=device)
        scheme = scheme_classes(tok.scheme)
        print(f"\n#### {path}")
        print(f"{'SNR':>6} | {'HH_R':>5} {'HH_P':>5} | {'KD_R':>5} {'SD_R':>5} "
              f"{'SD_P':>5} {'CY_R':>5} | {'allR':>5} {'allP':>5}")
        for snr in snr_rows:
            if snr is None:
                fe = base
            else:
                mixer = AccompanimentMixer(args.accompaniment_dir, snr_db_choices=[snr],
                                           prob=1.0, seed=args.shuffle_seed)
                fe = MixingFrontend(base, mixer)
            preds = transcribe_dataset(model, tracks, tok, fe,
                                       max_len=args.max_len, batch_size=64, device=device)
            scores = [score_track(t.annotation, preds[t.track_id], tok.scheme)
                      for t in tracks if t.track_id in preds]
            pc = _perclass(scores, scheme)
            tot = [0, 0, 0]
            for p, r, est, ref in pc.values():
                tot[0] += ref
                tot[1] += est
                tot[2] += int(round(p * est))
            all_p = tot[2] / tot[1] if tot[1] else 0.0
            all_r = tot[2] / tot[0] if tot[0] else 0.0
            hh = pc.get("HH", (0, 0, 0, 0))
            kd = pc.get("KD", (0, 0, 0, 0))
            sd = pc.get("SD", (0, 0, 0, 0))
            cy = pc.get("CY", (0, 0, 0, 0))
            tag = "clean" if snr is None else f"{snr:+.0f}"
            print(f"{tag:>6} | {hh[1]:>5.3f} {hh[0]:>5.3f} | {kd[1]:>5.3f} "
                  f"{sd[1]:>5.3f} {sd[0]:>5.3f} {cy[1]:>5.3f} | {all_r:>5.3f} {all_p:>5.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
