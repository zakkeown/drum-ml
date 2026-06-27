#!/usr/bin/env python
"""Evaluate predictions against a dataset's references.

Predictions live in a directory as ``<track_id>.tsv`` (the format written by
``DrumAnnotation.to_tsv``). References come from a dataset adapter.

Example:
    uv run python scripts/run_eval.py \\
        --dataset mdb --root datasets/MDBDrums \\
        --pred-dir runs/adtof/mdb --scheme 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drumml.eval import aggregate, format_report, score_track
from drumml.events import DrumAnnotation


def build_adapter(name: str, root: Path):
    if name == "egmd":
        from drumml.data.egmd import EGMDAdapter

        return EGMDAdapter(root)
    if name == "mdb":
        from drumml.data.mdb import MDBDrumsAdapter

        return MDBDrumsAdapter(root)
    raise SystemExit(f"unknown dataset adapter {name!r} (have: egmd, mdb)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="adapter name: egmd | mdb")
    ap.add_argument("--root", required=True, type=Path, help="dataset root dir")
    ap.add_argument("--pred-dir", required=True, type=Path, help="dir of <track_id>.tsv predictions")
    ap.add_argument("--scheme", default="5", choices=["3", "5", "8", "canonical"])
    ap.add_argument("--window", type=float, default=0.05, help="onset tolerance (s)")
    args = ap.parse_args(argv)

    adapter = build_adapter(args.dataset, args.root)

    track_scores = []
    missing = 0
    for track in adapter.tracks():
        pred_path = args.pred_dir / f"{track.track_id}.tsv"
        if not pred_path.exists():
            missing += 1
            continue
        est = DrumAnnotation.from_tsv(pred_path, track.track_id)
        track_scores.append(score_track(track.annotation, est, args.scheme, args.window))

    if not track_scores:
        print("no matched predictions found — nothing scored", file=sys.stderr)
        return 1

    ds = aggregate(track_scores, name=adapter.name)
    print(format_report(ds))
    if missing:
        print(f"\n(note: {missing} reference tracks had no matching prediction)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
