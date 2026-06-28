#!/usr/bin/env python
"""Precompute drum-free accompaniment + empirical SNRs from MUSDB18(-HQ) stems.

For each MUSDB track, sum the non-drum stems (bass + other + vocals) into a mono
accompaniment wav, and record the real drums-vs-accompaniment loudness ratio
(``20*log10(rms_drums/rms_accomp)`` in dB). The augmentation
(:mod:`drumml.data.augment`) samples its mix SNR from these empirical ratios, so
training mixes span the real range -- including accompaniment-as-loud-as-drums,
which is what teaches the model to ignore non-drum energy.

Expects the MUSDB18-HQ wav layout: ``<root>/**/<track>/{drums,bass,other,vocals}.wav``.

Example:
    uv run python scripts/prep_accompaniment.py \\
        --musdb-root datasets/musdb18hq --out-dir datasets/musdb_accompaniment
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musdb-root", required=True, type=Path,
                    help="MUSDB18-HQ root (contains track dirs with per-stem wavs)")
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/musdb_accompaniment"))
    ap.add_argument("--stems", nargs="+", default=["bass", "other", "vocals"],
                    help="non-drum stems to sum into accompaniment")
    args = ap.parse_args(argv)

    import soundfile as sf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    drum_paths = sorted(args.musdb_root.rglob("drums.wav"))
    if not drum_paths:
        print(f"no drums.wav under {args.musdb_root} (is it the HQ wav layout?)", flush=True)
        return 1
    print(f"{len(drum_paths)} MUSDB tracks under {args.musdb_root}")

    snr_db: list[float] = []
    written = 0
    for dp in drum_paths:
        track_dir = dp.parent
        name = track_dir.name.replace("/", "_")
        try:
            drums, sr = _load_mono(dp)
            acc = None
            for stem in args.stems:
                sp = track_dir / f"{stem}.wav"
                if not sp.exists():
                    continue
                s, _ = _load_mono(sp)
                acc = s if acc is None else acc[: len(s)] + s[: len(acc)]
            if acc is None or _rms(acc) < 1e-6 or _rms(drums) < 1e-6:
                print(f"  skip {name}: empty drums/accompaniment", flush=True)
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {name}: {exc}", flush=True)
            continue

        ratio = 20.0 * math.log10(_rms(drums) / _rms(acc))
        snr_db.append(round(ratio, 3))
        sf.write(str(args.out_dir / f"{name}.wav"), acc, sr)
        written += 1
        if written % 25 == 0:
            print(f"  {written} done ...", flush=True)

    (args.out_dir / "snr_db.json").write_text(json.dumps(snr_db))
    arr = np.asarray(snr_db)
    print(f"\nwrote {written} accompaniment wavs -> {args.out_dir}")
    print(f"empirical drums-vs-accompaniment SNR (dB): "
          f"min {arr.min():.1f}  median {np.median(arr):.1f}  max {arr.max():.1f}  "
          f"| {(arr < 0).mean()*100:.0f}% accompaniment-dominant (SNR<0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
