"""ADTOF adapter / parser (Zehren et al.).

ADTOF is primarily a *training pipeline + 5-class model* over crowdsourced
rhythm-game charts; for copyright reasons it distributes mel-spectrograms and
labels rather than raw audio. Its labels use a 5-class scheme encoded as GM-ish
MIDI pitches. We provide a parser for ADTOF-style label files so ADTOF output
(or its references) can be scored with this toolkit.

VERIFY: confirm the exact pitch->class encoding of the ADTOF release you use;
the map below is the documented 5-class convention (KD/SD/TT/HH/CY).
License: code CC BY-NC-SA 4.0; dataset has custom non-commercial terms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from drumml.events import DrumAnnotation
from drumml.taxonomy import Canonical as C

# ADTOF 5-class output pitches -> canonical. VERIFY per release.
ADTOF_PITCH_TO_CANONICAL: dict[int, C] = {
    35: C.KICK, 36: C.KICK,
    38: C.SNARE, 40: C.SNARE,
    47: C.TOM_MID, 45: C.TOM_MID, 43: C.TOM_LO, 50: C.TOM_HI,
    42: C.HH_CLOSED, 44: C.HH_PEDAL, 46: C.HH_OPEN,
    49: C.CRASH, 51: C.RIDE, 52: C.CRASH, 53: C.RIDE_BELL, 55: C.CRASH, 57: C.CRASH, 59: C.RIDE,
}


def annotation_from_adtof_labels(path: str | Path, track_id: Optional[str] = None) -> DrumAnnotation:
    """Parse an ADTOF-style ``<time>\\t<midi_pitch>`` label/prediction file."""
    path = Path(path)
    track_id = track_id or path.stem
    rows: list[tuple[float, int, Optional[int]]] = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.replace(",", "\t").split()
        time = float(parts[0])
        pitch = int(float(parts[1]))
        rows.append((time, pitch, None))
    # Route through the ADTOF pitch map rather than the generic GM map.
    events = []
    from drumml.events import DrumEvent

    for time, pitch, _ in rows:
        canonical = ADTOF_PITCH_TO_CANONICAL.get(pitch)
        if canonical is not None:
            events.append(DrumEvent(time, canonical))
    return DrumAnnotation(track_id, events)
