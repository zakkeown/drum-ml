#!/usr/bin/env python
"""Precompute drum-free accompaniment + empirical SNRs from MUSDB18 stems.

For each MUSDB track, sum the non-drum stems (bass + other + vocals) into a mono
accompaniment wav, and record the real drums-vs-accompaniment loudness ratio
(``20*log10(rms_drums/rms_accomp)`` in dB). The augmentation
(:mod:`drumml.data.augment`) samples its mix SNR from these empirical ratios, so
training mixes span the real range -- including accompaniment-as-loud-as-drums,
which is what teaches the model to ignore non-drum energy.

Reads either layout, auto-detected:
* MUSDB18-HQ wav: ``<root>/**/<track>/{drums,bass,other,vocals}.wav`` (soundfile).
* MUSDB18 mp4 STEMS: ``<root>/{train,test}/*.stem.mp4`` via the ``musdb`` package
  (needs ``musdb``/``stempeg``/ffmpeg).

Example:
    uv run --with musdb python scripts/prep_accompaniment.py \\
        --musdb-root datasets/musdb18 --out-dir datasets/musdb_accompaniment
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim > 1 else x


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return np.ascontiguousarray(_mono(wav), dtype=np.float32), int(sr)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _iter_wav(root: Path, stems: list[str]) -> Iterator[tuple[str, np.ndarray, np.ndarray, int]]:
    """Yield (name, drums_mono, accompaniment_mono, sr) for the HQ wav layout."""
    for dp in sorted(root.rglob("drums.wav")):
        d = dp.parent
        drums, sr = _load_mono(dp)
        acc = None
        for stem in stems:
            sp = d / f"{stem}.wav"
            if sp.exists():
                s, _ = _load_mono(sp)
                acc = s if acc is None else acc[: len(s)] + s[: len(acc)]
        if acc is not None:
            yield d.name.replace("/", "_"), drums, acc, sr


def _iter_musdb(root: Path, stems: list[str]) -> Iterator[tuple[str, np.ndarray, np.ndarray, int]]:
    """Yield (name, drums_mono, accompaniment_mono, sr) via the musdb mp4 reader."""
    import musdb  # lazy: needs stempeg + ffmpeg

    db = musdb.DB(root=str(root), is_wav=False)
    if not db.tracks:
        raise SystemExit(f"musdb found no tracks under {root}")
    for track in db.tracks:
        sr = int(track.rate)
        drums = _mono(track.targets["drums"].audio)
        acc = None
        for stem in stems:
            s = _mono(track.targets[stem].audio)
            acc = s if acc is None else acc[: len(s)] + s[: len(acc)]
        yield track.name.replace("/", "_"), drums, acc, sr


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
    # Auto-detect layout: HQ wav stems on disk, else the musdb mp4 reader.
    if sorted(args.musdb_root.rglob("drums.wav")):
        source = _iter_wav(args.musdb_root, args.stems)
        print(f"reading HQ wav stems under {args.musdb_root}")
    else:
        source = _iter_musdb(args.musdb_root, args.stems)
        print(f"reading mp4 STEMS under {args.musdb_root} via musdb")

    snr_db: list[float] = []
    written = 0
    for name, drums, acc, sr in source:
        if acc is None or _rms(acc) < 1e-6 or _rms(drums) < 1e-6:
            print(f"  skip {name}: empty drums/accompaniment", flush=True)
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
