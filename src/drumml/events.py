"""Drum events and annotations — the common in-memory representation.

Every dataset adapter and every model prediction is normalized to a
``DrumAnnotation`` (a track id plus a list of canonical-class onsets with
optional velocity). The evaluator consumes only this type, so adding a new
dataset or a new model means writing one adapter, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from drumml.taxonomy import Canonical, gm_note_to_canonical, reduce


@dataclass(frozen=True, order=True)
class DrumEvent:
    """A single drum onset. ``time`` (seconds) is first so events sort by time."""

    time: float
    canonical: Canonical
    velocity: Optional[int] = None  # MIDI 1..127, or None if unknown

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"event time must be >= 0, got {self.time}")
        if self.velocity is not None and not (0 <= self.velocity <= 127):
            raise ValueError(f"velocity must be in 0..127, got {self.velocity}")


@dataclass
class DrumAnnotation:
    """An ordered set of drum events for one track."""

    track_id: str
    events: list[DrumEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.events = sorted(self.events)

    def __len__(self) -> int:
        return len(self.events)

    # --- construction -------------------------------------------------------
    @classmethod
    def from_gm_notes(
        cls,
        track_id: str,
        rows: Iterable[tuple[float, int, Optional[int]]],
    ) -> "DrumAnnotation":
        """Build from ``(time_seconds, gm_note, velocity)`` rows (e.g. parsed MIDI).

        Notes outside the GM percussion range are skipped.
        """
        events: list[DrumEvent] = []
        for time, note, vel in rows:
            canonical = gm_note_to_canonical(note)
            if canonical is None:
                continue
            events.append(DrumEvent(float(time), canonical, vel))
        return cls(track_id, events)

    @classmethod
    def from_label_rows(
        cls,
        track_id: str,
        rows: Iterable[tuple[float, str]],
        label_to_canonical: dict[str, Canonical],
        *,
        strict: bool = False,
    ) -> "DrumAnnotation":
        """Build from ``(time, label)`` rows using a dataset-specific label map.

        Unknown labels are skipped unless ``strict`` (then they raise). This is
        the generic path for text-annotated datasets (MDB, ENST, ...).
        """
        events: list[DrumEvent] = []
        for time, label in rows:
            canonical = label_to_canonical.get(label)
            if canonical is None:
                if strict:
                    raise KeyError(f"unmapped label {label!r} in track {track_id!r}")
                continue
            events.append(DrumEvent(float(time), canonical))
        return cls(track_id, events)

    # --- views for evaluation ----------------------------------------------
    def onsets_by_class(self, scheme: str) -> dict[str, np.ndarray]:
        """Group onset times (sorted, seconds) by reduced-class for a scheme.

        Canonical classes that the scheme does not score are dropped.
        """
        buckets: dict[str, list[float]] = {}
        for ev in self.events:
            name = reduce(ev.canonical, scheme)
            if name is None:
                continue
            buckets.setdefault(name, []).append(ev.time)
        return {name: np.array(sorted(times), dtype=float) for name, times in buckets.items()}

    # --- interchange (model predictions <-> disk) --------------------------
    def to_tsv(self, path: str | Path) -> None:
        """Write ``time<TAB>canonical<TAB>velocity`` rows (velocity blank if None)."""
        path = Path(path)
        lines = [
            f"{ev.time:.6f}\t{ev.canonical.value}\t{'' if ev.velocity is None else ev.velocity}"
            for ev in self.events
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))

    @classmethod
    def from_tsv(cls, path: str | Path, track_id: Optional[str] = None) -> "DrumAnnotation":
        path = Path(path)
        track_id = track_id or path.stem
        events: list[DrumEvent] = []
        for raw in path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split("\t")
            time = float(parts[0])
            canonical = Canonical(parts[1])
            vel = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
            events.append(DrumEvent(time, canonical, vel))
        return cls(track_id, events)
