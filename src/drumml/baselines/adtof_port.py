"""Run the ADTOF baseline and load its predictions into our format.

ADTOF (and the PyTorch port, github.com/xavriley/ADTOF-pytorch) is our floor: a
real-music, 5-class CRNN with released weights. It runs in *its own* environment
(TensorFlow/Keras or the port's PyTorch + madmom), so this module shells out to
its CLI rather than importing it — keeping ADTOF's heavy, version-pinned deps out
of drumml's environment.

Workflow:
    1. Install ADTOF separately (see docs/reproduce_adtof_baseline.md).
    2. ``run_adtof(audio, out_dir)`` invokes its transcriptor and parses the
       resulting label file into a canonical ``DrumAnnotation``.

The prediction parser is exercised by tests against a synthetic ADTOF-style
file; the subprocess call itself is environment-dependent and not unit-tested.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

from drumml.data.adtof import annotation_from_adtof_labels
from drumml.events import DrumAnnotation


def run_adtof(
    audio_path: str | Path,
    out_dir: str | Path,
    *,
    command: Sequence[str] = ("drumTranscriptor",),
    extra_args: Sequence[str] = (),
    timeout: Optional[float] = None,
) -> DrumAnnotation:
    """Transcribe ``audio_path`` with the installed ADTOF CLI; return its output.

    Adjust ``command``/``extra_args`` to match your ADTOF install (the official
    repo exposes ``drumTranscriptor``; the port differs). Raises if the CLI is
    missing or fails, or if no label file is produced.
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{audio_path.stem}.txt"

    cmd = [*command, str(audio_path), str(out_file), *extra_args]
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
            f"ADTOF produced no label file at {out_file}; check the CLI's output "
            "convention and adjust `command`/`extra_args`."
        )
    return annotation_from_adtof_labels(out_file, track_id=audio_path.stem)
