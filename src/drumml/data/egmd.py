"""E-GMD adapter (Expanded Groove MIDI Dataset, Google Magenta).

E-GMD ships paired audio (Roland TD-17 renders) and MIDI; the MIDI uses standard
GM percussion notes, so parsing is exact — no guessed label table. The dataset's
``e-gmd-v1.0.0.csv`` index lists, per row, the split and the audio/MIDI paths.

License: CC BY 4.0. Download: https://magenta.tensorflow.org/datasets/e-gmd
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional

from drumml.data.base import DatasetAdapter, Track
from drumml.events import DrumAnnotation


def annotation_from_midi(midi_path: str | Path, track_id: Optional[str] = None) -> DrumAnnotation:
    """Parse a (drum) MIDI file into a canonical annotation via GM note numbers."""
    import pretty_midi  # local import: only needed when actually parsing MIDI

    midi_path = Path(midi_path)
    track_id = track_id or midi_path.stem
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    rows: list[tuple[float, int, Optional[int]]] = []
    for inst in pm.instruments:
        # GM drums live on channel 10; pretty_midi exposes that as is_drum.
        if not inst.is_drum:
            continue
        for note in inst.notes:
            rows.append((note.start, note.pitch, int(note.velocity)))
    return DrumAnnotation.from_gm_notes(track_id, rows)


class EGMDAdapter(DatasetAdapter):
    name = "egmd"

    def __init__(self, root: str | Path, split: Optional[str] = None):
        """``root`` is the extracted E-GMD directory.

        Two on-disk layouts are auto-detected:

        * **HF mirror** (``schism-audio/e-gmd``): a ``metadata.csv`` whose
          ``midi_path`` / ``audio_path`` columns hold full in-repo paths under
          ``audio/{split}/{kit}/...`` and ``midi/{split}/{kit}/...``.
        * **Original Google release**: an ``e-gmd-v*.csv`` whose
          ``midi_filename`` / ``audio_filename`` columns are relative to ``root``.

        ``split`` optionally filters to "train" / "validation" / "test".
        """
        self.root = Path(root)
        self.split = split

        hf_meta = self.root / "metadata.csv"
        original = sorted(self.root.glob("e-gmd-v*.csv"))
        if hf_meta.exists():
            self.index_csv = hf_meta
            self._midi_col, self._audio_col = "midi_path", "audio_path"
        elif original:
            self.index_csv = original[0]
            self._midi_col, self._audio_col = "midi_filename", "audio_filename"
        else:
            raise FileNotFoundError(
                f"no E-GMD index under {self.root} (expected metadata.csv or e-gmd-v*.csv)"
            )

    def tracks(self) -> Iterator[Track]:
        with self.index_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if self.split and row.get("split") != self.split:
                    continue
                midi_rel = row[self._midi_col]
                audio_rel = row.get(self._audio_col)
                track_id = Path(midi_rel).stem
                ann = annotation_from_midi(self.root / midi_rel, track_id)
                audio = self.root / audio_rel if audio_rel else None
                yield Track(track_id=track_id, annotation=ann, audio_path=audio)
