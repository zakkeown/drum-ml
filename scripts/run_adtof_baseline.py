#!/usr/bin/env python
"""Reproduce the ADTOF baseline on MDB-Drums and score it with our harness.

This is the project's reproducible *floor*: transcribe MDB-Drums full-mix audio
with the ADTOF-pytorch CRNN (github.com/xavriley/ADTOF-pytorch — the same model
as Zehren et al., within ~0.2% F of the original), then score the predictions
against MDB's references at the 5-class scheme.

The number to match is the published 5-class onset F at +/-50 ms on standard
MDB-Drums: ~0.76-0.81 ("SUM"/global F, i.e. TP/FP/FN pooled across classes and
tracks). In our report that convention is **micro-F**, NOT the macro headline:
MDB has only ~90 tom onsets, so macro (which weights toms equally with kick)
reads much lower than micro and is the wrong number for this comparison.

Prereq: install the port in its own env, e.g.
    uv tool install "git+https://github.com/xavriley/ADTOF-pytorch"
so the ``adtof`` command is on PATH. See docs/reproduce_adtof_baseline.md.

Example:
    uv run python scripts/run_adtof_baseline.py --root datasets/MDBDrums
"""

from __future__ import annotations

import argparse
from pathlib import Path

PUBLISHED_MDB_F = (0.76, 0.81)  # standard MDB-Drums, 5-class, +/-50 ms, SUM F


def _count_by_class(ann, scheme: str) -> dict[str, int]:
    return {c: len(v) for c, v in ann.onsets_by_class(scheme).items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets/MDBDrums"),
                    help="MDB-Drums checkout root")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/adtof/mdb"),
                    help="where ADTOF MIDI + canonical TSV predictions are written")
    ap.add_argument("--scheme", default="5", help="scoring scheme (default 5-class)")
    ap.add_argument("--mix", default="full_mix", help="full_mix (default) | drum_only")
    ap.add_argument("--limit", type=int, default=None, help="cap #tracks (smoke)")
    ap.add_argument("--device", default="cpu", help="adtof device: cpu | cuda")
    ap.add_argument("--command", default="adtof", help="ADTOF CLI name on PATH")
    args = ap.parse_args(argv)

    from drumml.baselines.adtof_port import run_adtof
    from drumml.data.mdb import MDBDrumsAdapter
    from drumml.eval import aggregate, format_report, score_track

    adapter = MDBDrumsAdapter(args.root, mix=args.mix)
    tracks = [t for t in adapter.tracks() if t.audio_path is not None]
    if args.limit:
        tracks = tracks[: args.limit]
    if not tracks:
        print(f"no tracks with audio under {args.root}", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"transcribing {len(tracks)} MDB tracks with ADTOF ({args.mix}, device={args.device})")

    scores = []
    first_spotcheck = None
    for i, track in enumerate(tracks):
        try:
            pred = run_adtof(
                track.audio_path, args.out_dir,
                command=(args.command,), device=args.device,
            )
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  [{i}] {track.track_id}: ADTOF failed: {exc}", flush=True)
            continue
        pred.to_tsv(args.out_dir / f"{track.track_id}.tsv")
        s = score_track(track.annotation, pred, args.scheme)
        scores.append(s)
        if first_spotcheck is None:
            first_spotcheck = (track, pred, s)
        print(f"  [{i}] {track.track_id}: micro-F {s.micro_f:.3f}  macro-F {s.macro_f:.3f}",
              flush=True)

    if not scores:
        print("no tracks scored (is the `adtof` command installed?)", flush=True)
        return 1

    # --- per-track spot-check: predicted vs reference onset counts ------------
    # A wrong pitch map / dropped-events bug shows up as an absurd count here,
    # which an aggregate F that just looks "a bit low" would hide.
    track, pred, s = first_spotcheck
    ref_counts = _count_by_class(track.annotation, args.scheme)
    est_counts = _count_by_class(pred, args.scheme)
    print(f"\nspot-check {track.track_id} (onset counts, scheme {args.scheme}):")
    print(f"  {'class':<6} {'ref':>5} {'est':>5}")
    for c in sorted(set(ref_counts) | set(est_counts)):
        print(f"  {c:<6} {ref_counts.get(c, 0):>5} {est_counts.get(c, 0):>5}")

    # --- aggregate report + comparison to the published band -----------------
    ds = aggregate(scores, name=f"adtof/mdb-{args.mix}")

    # Dataset-wide per-class ref/est/tp counts — the decisive sanity check.
    # A systematic class swap (e.g. tom<->hihat pitch assignment) shows up as
    # est counts that don't track the reference's class balance.
    from drumml.taxonomy import scheme_classes
    pooled: dict[str, list[int]] = {c: [0, 0, 0] for c in scheme_classes(args.scheme)}
    for t in scores:
        for c, cs in t.per_class.items():
            pooled[c][0] += cs.n_ref
            pooled[c][1] += cs.n_est
            pooled[c][2] += cs.tp
    print(f"\ndataset-wide onset counts (scheme {args.scheme}):")
    print(f"  {'class':<6} {'ref':>6} {'est':>6} {'tp':>6}")
    for c in scheme_classes(args.scheme):
        ref, est, tp = pooled[c]
        print(f"  {c:<6} {ref:>6} {est:>6} {tp:>6}")

    print("\n" + format_report(ds))
    lo, hi = PUBLISHED_MDB_F
    verdict = "WITHIN" if lo - 0.03 <= ds.micro_f <= hi + 0.03 else "OUTSIDE"
    print(
        f"\nreproduction check: micro-F {ds.micro_f:.3f} vs published {lo:.2f}-{hi:.2f} "
        f"(5-class, +/-50 ms, SUM) -> {verdict} band"
    )
    print("note: micro-F is the comparable metric; macro-F is the floor our model must clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
