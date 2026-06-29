#!/usr/bin/env python
"""Lower-bound A2MD's weak-label quality with a strong reference transcriber.

A2MD's drum labels come from DTW-aligning a *separate* Lakh-MIDI arrangement to a
recording, so the dominant error is presence noise, and the alignment-distance
buckets measure global time-warp, NOT note-level agreement. To turn "weak labels
(caveat)" into a number, score the **ADTOF** model's predictions *against* A2MD's
labels: ADTOF scores ~0.79 micro against CLEAN MDB hand labels, so its audio-side
ability is ~constant; whatever it scores against A2MD labels is a direct lower
bound on A2MD label quality. A small gap from 0.79 => the tight-bucket labels are
trustworthy (a weak downstream result can't then be blamed on label noise).

Requires the `adtof` CLI on PATH (see docs/reproduce_adtof_baseline.md).

Example:
    uv run python scripts/probe_a2md_label_quality.py --n 12 --max-dist 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ADTOF reproduced on clean MDB full-mix (scheme 3): the comparison anchor.
ADTOF_ON_MDB = {"micro": 0.796, "macro": 0.800}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets/a2md/a2md_public"))
    ap.add_argument("--n", type=int, default=12, help="#tracks to probe")
    ap.add_argument("--max-dist", type=float, default=0.0, help="A2MD bucket cap (0.0 = tightest)")
    ap.add_argument("--scheme", default="3")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/adtof/a2md"))
    ap.add_argument("--command", default="adtof")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from drumml.baselines.adtof_port import run_adtof
    from drumml.data.a2md import A2MDAdapter
    from drumml.eval import aggregate, score_track
    from drumml.taxonomy import scheme_classes

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks = list(A2MDAdapter(args.root, max_dist=args.max_dist).tracks())[: args.n]
    print(f"scoring ADTOF vs A2MD labels on {len(tracks)} tracks "
          f"(dist<={args.max_dist}, scheme {args.scheme})", flush=True)

    scores = []
    for i, t in enumerate(tracks):
        try:
            pred = run_adtof(t.audio_path, args.out_dir, command=(args.command,), device=args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {t.track_id}: ADTOF failed: {exc}", flush=True)
            continue
        s = score_track(t.annotation, pred, args.scheme)
        scores.append(s)
        n_ref = sum(len(v) for v in t.annotation.onsets_by_class(args.scheme).values())
        print(f"  [{i}] {t.track_id}: micro {s.micro_f:.3f}  (ref {n_ref} onsets)", flush=True)

    if not scores:
        print("no tracks scored (is the `adtof` command installed?)", flush=True)
        return 1
    ds = aggregate(scores, name="adtof-vs-a2md")
    pc = " ".join(f"{c}:{ds.per_class_f.get(c, 0):.3f}" for c in scheme_classes(args.scheme))
    print(f"\nADTOF vs A2MD labels (scheme {args.scheme}, {len(scores)} trk): "
          f"micro {ds.micro_f:.3f}  macro {ds.macro_f_pooled:.3f} | {pc}")
    print(f"reference: ADTOF vs CLEAN MDB labels = micro {ADTOF_ON_MDB['micro']:.3f} "
          f"/ macro {ADTOF_ON_MDB['macro']:.3f}")
    print(f"=> A2MD label-quality lower bound: micro {ds.micro_f:.3f} "
          f"(gap {ADTOF_ON_MDB['micro'] - ds.micro_f:+.3f} attributable to A2MD label noise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
