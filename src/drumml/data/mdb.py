"""MDB-Drums adapter (Southall et al., ISMIR-LBD 2017).

MDB-Drums annotations are whitespace-separated ``<onset_seconds> <label>`` files
(the official files are ``<onset> \\t <label>`` with stray surrounding spaces, so
we split on any whitespace). The repo provides a coarse ``class`` level and a
fine ``subclass`` level.

Verified against the official checkout (github.com/CarlSouthall/MDBDrums): the
content lives under a top-level ``MDB Drums/`` folder (literal space); the
class-level vocabulary is exactly six tokens — ``KD SD HH TT CY OT`` — totalling
7994 onsets across the 23 tracks (KD 1539, SD 2654, HH 2639, TT 90, CY 1002,
OT 70). Annotation stems carry a ``_class`` suffix while the matching audio
carries ``_MIX`` (full mix) / ``_Drum`` (drums only), so tracks join on the bare
``MusicDelta_<Genre>`` stem, not the filename. ``MDB_CLASS_LABELS`` also lists the
subclass tokens (harmless at class level) for callers that parse subclass files;
any unmapped token is skipped, or raises with ``strict=True``.

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
    "RDC": C.RIDE,   # ride cymbal (the real subclass token; "RD" does not exist)
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

    # Audio-filename suffix per mix kind (full_mix -> *_MIX.wav, drum_only -> *_Drum.wav).
    _MIX_SUFFIX = {"full_mix": "_MIX", "drum_only": "_Drum"}

    def __init__(
        self,
        root: str | Path,
        *,
        strict: bool = False,
        level: str = "class",
        mix: str = "full_mix",
    ):
        """Adapt an MDB-Drums checkout.

        ``root`` may point at the repository root (which nests everything under
        ``MDB Drums/``) or directly at that inner folder. ``level`` selects
        ``class`` (default; the 5-class target) or ``subclass`` annotations;
        ``mix`` selects ``full_mix`` (default; what full-mix transcribers expect)
        or ``drum_only`` audio.
        """
        self.root = Path(root)
        self.strict = strict
        self.level = level
        self.mix = mix

        # The official checkout nests content under "MDB Drums/" (literal space);
        # accept a root pointing at either the repo or that inner folder.
        inner = self.root / "MDB Drums"
        self.base = inner if inner.is_dir() else self.root

        self.ann_dir = self.base / "annotations" / level
        if not self.ann_dir.is_dir():
            # Flattened checkouts: fall back to a recursive search.
            self.ann_dir = self.base / "annotations"

    def tracks(self) -> Iterator[Track]:
        suffix = f"_{self.level}"  # strip "_class"/"_subclass" to get the join stem
        ann_paths = sorted(self.ann_dir.rglob(f"*{suffix}.txt"))
        if not ann_paths:  # synthetic/flat layouts may omit the level suffix
            ann_paths = sorted(self.ann_dir.rglob("*.txt"))
        for ann_path in ann_paths:
            stem = ann_path.stem
            track_id = stem[: -len(suffix)] if stem.endswith(suffix) else stem
            rows = _read_label_file(ann_path)
            ann = DrumAnnotation.from_label_rows(
                track_id, rows, MDB_CLASS_LABELS, strict=self.strict
            )
            audio = self._find_audio(track_id)
            yield Track(track_id=track_id, annotation=ann, audio_path=audio)

    def _find_audio(self, track_id: str) -> Path | None:
        mix_suffix = self._MIX_SUFFIX.get(self.mix, "")
        # Prefer the chosen mix with its suffix; tolerate suffix-less flat layouts.
        for sub, sfx in ((self.mix, mix_suffix), (self.mix, ""), ("audio", "")):
            for ext in (".wav", ".flac"):
                base = self.base / "audio" / sub if sub != "audio" else self.base / "audio"
                cand = base / f"{track_id}{sfx}{ext}"
                if cand.exists():
                    return cand
        return None
