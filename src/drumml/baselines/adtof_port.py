"""Run the ADTOF baseline and load its predictions into our format.

ADTOF is our floor: a real-music, 5-class CRNN with released weights. We run the
PyTorch port (github.com/xavriley/ADTOF-pytorch) — the same model, ~-0.2% F vs
the original — which depends only on torch/librosa/pretty_midi. It runs in *its
own* isolated environment (e.g. ``uv tool install``), so this module shells out
to its ``adtof`` CLI rather than importing it, keeping the port's deps out of
drumml's environment.

The port's CLI is ``adtof --audio IN.wav --out OUT.mid --device cpu`` (its
default device is ``cuda``; pass ``cpu`` on a Mac). Output is a MIDI file with
onsets at the LABELS_5 pitches, which :func:`annotation_from_adtof_midi` maps
into our canonical taxonomy.

Workflow:
    1. Install the port in its own env (see docs/reproduce_adtof_baseline.md).
    2. ``run_adtof(audio, out_dir)`` invokes it and parses the resulting MIDI
       into a canonical ``DrumAnnotation``.

The MIDI parser is unit-tested against synthetic MIDI; the subprocess call
itself is environment-dependent and not unit-tested.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

from drumml.data.adtof import annotation_from_adtof_midi
from drumml.events import DrumAnnotation


def run_adtof(
    audio_path: str | Path,
    out_dir: str | Path,
    *,
    command: Sequence[str] = ("adtof",),
    device: str = "cpu",
    extra_args: Sequence[str] = (),
    timeout: Optional[float] = None,
) -> DrumAnnotation:
    """Transcribe ``audio_path`` with the installed ADTOF-pytorch CLI.

    Invokes ``adtof --audio <wav> --out <mid> --device <device>`` and parses the
    MIDI it writes. ``device`` defaults to ``cpu`` (the CLI defaults to ``cuda``).
    Pass per-class thresholds etc. via ``extra_args``. Raises if the CLI is
    missing or fails, or if no MIDI file is produced.
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{audio_path.stem}.mid"

    cmd = [
        *command,
        "--audio", str(audio_path),
        "--out", str(out_file),
        "--device", device,
        *extra_args,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
    except FileNotFoundError as exc:  # CLI not on PATH
        raise FileNotFoundError(
            f"ADTOF command {command[0]!r} not found. See "
            "docs/reproduce_adtof_baseline.md for setup."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ADTOF failed:\n{exc.stderr}") from exc

    if not out_file.exists():
        raise FileNotFoundError(
            f"ADTOF produced no MIDI at {out_file}; check the CLI's output "
            "convention and adjust `command`/`extra_args`."
        )
    return annotation_from_adtof_midi(out_file, track_id=audio_path.stem)
