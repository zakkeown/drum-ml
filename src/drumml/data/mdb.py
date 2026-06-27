"""MDB-Drums adapter (Southall et al., ISMIR-LBD 2017).

MDB-Drums annotations are tab-separated ``<onset_seconds>\\t<label>`` files. The
repo provides both a coarse ``class`` level and a fine ``subclass`` level.

NOTE / VERIFY: the label tokens below are the standard MDB-Drums *class*-level
tokens, but token spelling differs slightly between releases. Before trusting
numbers, diff ``MDB_CLASS_LABELS`` against the ``annotations/`` folder of the
checkout (https://github.com/CarlSouthall/MDBDrums) — any unmapped token is
skipped (or raises, with ``strict=True``), so a mismatch shows up as missing
events rather than silent corruption.

License: dataset CC BY-NC-SA 4.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from drumml.data.base import DatasetAdapter, Track
from drumml.events import DrumAnnotation
from drumml.taxonomy import Canonical as C

# MDB-Drums class-level tokens -> canonical. VERIFY against the repo (see module docstring).
MDB_CLASS_LABELS: dict[str, C] = {
    "KD": C.KICK,
    "SD": C.SNARE,
    "SDB": C.SNARE,   # snare brush
    "SDD": C.SNARE,   # snare drag/ghost
    "SST": C.SIDESTICK,
    "TT": C.TOM_MID,  # MDB class level does not split toms; map to a tom class
    "HH": C.HH_CLOSED,
    "CHH": C.HH_CLOSED,
    "OHH": C.HH_OPEN,
    "PHH": C.HH_PEDAL,
    "RD": C.RIDE,
    "RDB": C.RIDE_BELL,
    "CRC": C.CRASH,
    "SPC": C.CRASH,   # splash
    "CHC": C.CRASH,   # china
    "CY": C.CRASH,    # generic cymbal -> crash bucket
    "TMB": C.PERC,    # tambourine
    "OT": C.PERC,     # other
}


def _read_label_file(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.replace(",", "\t").split()
        rows.append((float(parts[0]), parts[1]))
    return rows


class MDBDrumsAdapter(DatasetAdapter):
    name = "mdb"

    def __init__(self, root: str | Path, *, strict: bool = False):
        """``root`` is the MDBDrums checkout (with ``annotations/`` and ``audio/``)."""
        self.root = Path(root)
        self.strict = strict
        self.ann_dir = self.root / "annotations" / "class"
        if not self.ann_dir.is_dir():
            # Some checkouts flatten annotations; fall back to a recursive search.
            self.ann_dir = self.root / "annotations"

    def tracks(self) -> Iterator[Track]:
        for ann_path in sorted(self.ann_dir.rglob("*.txt")):
            track_id = ann_path.stem
            rows = _read_label_file(ann_path)
            ann = DrumAnnotation.from_label_rows(
                track_id, rows, MDB_CLASS_LABELS, strict=self.strict
            )
            audio = self._find_audio(track_id)
            yield Track(track_id=track_id, annotation=ann, audio_path=audio)

    def _find_audio(self, track_id: str) -> Path | None:
        for sub in ("audio/full_mix", "audio/drum_only", "audio"):
            for ext in (".wav", ".flac"):
                cand = self.root / sub / f"{track_id}{ext}"
                if cand.exists():
                    return cand
        return None
