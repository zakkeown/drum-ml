#!/usr/bin/env python
"""Evaluate a trained checkpoint on a dataset split (transcribe -> score).

Loads a checkpoint, transcribes the chosen split with the model, and scores the
predictions against the references with the onset-F harness. This is the
end-to-end measurement the design's headline metric needs.

Example:
    uv run python scripts/evaluate.py \\
        --checkpoint checkpoints/egmd_seq2seq.pt \\
        --dataset egmd --root datasets/e-gmd --split test --limit 200
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_adapter(name: str, root: Path, split: str | None):
    if name == "egmd":
        from drumml.data.egmd import EGMDAdapter

        return EGMDAdapter(root, split=split)
    if name == "mdb":
        from drumml.data.mdb import MDBDrumsAdapter

        if split:
            print(f"(note: MDB has no splits; ignoring --split {split})")
        return MDBDrumsAdapter(root)
    raise SystemExit(f"unknown dataset adapter {name!r} (have: egmd, mdb)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset", required=True, help="adapter: egmd | mdb")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="cap #tracks scored")
    ap.add_argument("--scheme", default=None, help="override eval scheme (default: tokenizer's)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from drumml.checkpoint import load_checkpoint
    from drumml.eval import aggregate, format_report, score_track
    from drumml.features import LogMelFrontend
    from drumml.transcribe import transcribe_dataset

    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=args.device)
    scheme = args.scheme or tokenizer.scheme

    adapter = build_adapter(args.dataset, args.root, args.split)
    tracks = list(adapter.tracks())
    if args.limit:
        tracks = tracks[: args.limit]
    print(f"scoring {len(tracks)} tracks from {adapter.name}/{args.split} @ scheme {scheme}")

    frontend = LogMelFrontend()
    preds = transcribe_dataset(
        model, tracks, tokenizer, frontend,
        max_len=args.max_len, device=args.device,
        on_track=lambda i, tid: (i % 50 == 0) and print(f"  transcribed {i} ..."),
    )

    scores = [
        score_track(t.annotation, preds[t.track_id], scheme)
        for t in tracks
        if t.track_id in preds
    ]
    if not scores:
        print("no tracks scored", flush=True)
        return 1
    print(format_report(aggregate(scores, name=f"{adapter.name}/{args.split}")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
