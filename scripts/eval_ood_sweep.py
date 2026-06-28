#!/usr/bin/env python
"""Sweep epoch checkpoints over a held-out dataset -> the OOD transfer curve.

The cross-dataset (out-of-distribution) optimum usually arrives *before* the
in-domain optimum, so a single last-epoch number under-reports transfer. This
loads each ``*_epoch{n}.pt`` checkpoint, transcribes the target dataset, scores
it through the same harness, and prints macro-/micro-F vs epoch — the curve that,
for a data-first thesis, *is* the result.

Example (our E-GMD model on the MDB full-mix floor):
    uv run python scripts/eval_ood_sweep.py \\
        --checkpoints "checkpoints/egmd_seq2seq_epoch*.pt" \\
        --dataset mdb --root datasets/MDBDrums
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

# ADTOF reproduced floor on standard MDB-Drums full-mix (scheme 5, +/-50 ms).
ADTOF_FLOOR = {"macro": 0.716, "micro": 0.792}


def _epoch_of(path: str) -> int:
    m = re.search(r"epoch(\d+)", Path(path).stem)
    return int(m.group(1)) if m else -1


def build_adapter(name: str, root: Path):
    if name == "mdb":
        from drumml.data.mdb import MDBDrumsAdapter

        return MDBDrumsAdapter(root)
    if name == "egmd":
        from drumml.data.egmd import EGMDAdapter

        return EGMDAdapter(root, split="test")
    raise SystemExit(f"unknown dataset adapter {name!r} (have: mdb, egmd)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", required=True,
                    help="glob of epoch checkpoints, e.g. 'checkpoints/m_epoch*.pt'")
    ap.add_argument("--dataset", default="mdb", help="held-out adapter: mdb | egmd")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--scheme", default=None, help="override scoring scheme (default: tokenizer's)")
    ap.add_argument("--limit", type=int, default=None, help="cap #tracks")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    from drumml.checkpoint import load_checkpoint
    from drumml.eval import aggregate, score_track
    from drumml.features import LogMelFrontend
    from drumml.train import pick_device
    from drumml.transcribe import transcribe_dataset

    paths = sorted(glob.glob(args.checkpoints), key=_epoch_of)
    if not paths:
        print(f"no checkpoints match {args.checkpoints!r}", flush=True)
        return 1
    device = pick_device(args.device)
    print(f"device: {device} | {len(paths)} checkpoints | {args.dataset}/{args.root}")

    # Parse the target tracks once; reuse across checkpoints.
    adapter = build_adapter(args.dataset, args.root)
    tracks = list(adapter.tracks())
    if args.limit:
        tracks = tracks[: args.limit]
    frontend = LogMelFrontend()

    rows = []
    for path in paths:
        model, tokenizer, _ = load_checkpoint(path, device=device)
        scheme = args.scheme or tokenizer.scheme
        preds = transcribe_dataset(
            model, tracks, tokenizer, frontend,
            max_len=args.max_len, batch_size=args.batch_size, device=device,
        )
        scores = [score_track(t.annotation, preds[t.track_id], scheme)
                  for t in tracks if t.track_id in preds]
        ds = aggregate(scores, name=f"{adapter.name}")
        rows.append((_epoch_of(path), ds.macro_f_pooled, ds.micro_f, scheme))
        print(f"  epoch {_epoch_of(path):>2}: macro-F {ds.macro_f_pooled:.3f}  "
              f"micro-F {ds.micro_f:.3f}", flush=True)

    # --- transfer curve + comparison to the ADTOF floor ----------------------
    scheme = rows[0][3]
    best_macro = max(rows, key=lambda r: r[1])
    best_micro = max(rows, key=lambda r: r[2])
    print(f"\nOOD transfer curve ({adapter.name}, scheme {scheme}, {len(tracks)} tracks):")
    print(f"  {'epoch':>5} {'macro-F':>8} {'micro-F':>8}")
    for ep, ma, mi, _ in rows:
        mark = "  <- best macro" if (ep, ma) == (best_macro[0], best_macro[1]) else ""
        print(f"  {ep:>5} {ma:>8.3f} {mi:>8.3f}{mark}")
    print(f"\nbest macro-F {best_macro[1]:.3f} @ epoch {best_macro[0]}  "
          f"(ADTOF floor {ADTOF_FLOOR['macro']:.3f})")
    print(f"best micro-F {best_micro[2]:.3f} @ epoch {best_micro[0]}  "
          f"(ADTOF floor {ADTOF_FLOOR['micro']:.3f})")
    gap = ADTOF_FLOOR["macro"] - best_macro[1]
    print(f"gap to floor (macro): {gap:+.3f}  "
          f"-- expected large for E-GMD->MDB (isolated-kit -> full-mix domain shift)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
