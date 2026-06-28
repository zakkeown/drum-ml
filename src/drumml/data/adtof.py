"""ADTOF adapter / parser (Zehren et al.).

ADTOF is a *training pipeline + 5-class CRNN* over crowdsourced rhythm-game
charts. We run inference with the PyTorch port (github.com/xavriley/ADTOF-pytorch,
same model, ~-0.2% F vs original), which emits a **MIDI file** with onsets at the
5-class pitches ``LABELS_5 = [35, 38, 47, 42, 49]`` = kick/snare/tom/hihat/cymbal
(the cymbal class fuses crash+ride). :func:`annotation_from_adtof_midi` parses
that output; :func:`annotation_from_adtof_labels` parses the original notebook's
``<time> <pitch>`` text dumps. Both route pitches through the ADTOF map below so
predictions land in our canonical taxonomy and score with this toolkit.

License: code GPLv3 (setup.py) / CC BY-NC-SA 4.0 (README, inconsistent); dataset
has custom non-commercial terms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from drumml.events import DrumAnnotation, DrumEvent
from drumml.taxonomy import Canonical as C

# ADTOF output pitches -> canonical. The port emits only the 5 LABELS_5 pitches
# (35/38/47/42/49); the extra GM-neighbour pitches are tolerated for robustness
# across releases. At scheme "5" all cymbals (49 + ride/bell neighbours) fold to CY.
ADTOF_PITCH_TO_CANONICAL: dict[int, C] = {
    35: C.KICK, 36: C.KICK,
    38: C.SNARE, 40: C.SNARE,
    47: C.TOM_MID, 45: C.TOM_MID, 43: C.TOM_LO, 50: C.TOM_HI,
    42: C.HH_CLOSED, 44: C.HH_PEDAL, 46: C.HH_OPEN,
    49: C.CRASH, 51: C.RIDE, 52: C.CRASH, 53: C.RIDE_BELL, 55: C.CRASH, 57: C.CRASH, 59: C.RIDE,
}


def annotation_from_adtof_midi(path: str | Path, track_id: Optional[str] = None) -> DrumAnnotation:
    """Parse an ADTOF-pytorch MIDI transcription into a canonical annotation.

    The port writes a single-instrument MIDI with notes at the LABELS_5 pitches;
    each note's start time becomes a :class:`DrumEvent`. Pitches outside the ADTOF
    map are skipped (the port emits none, but be defensive).
    """
    import pretty_midi  # lazy: pretty_midi is part of the [model] extra

    path = Path(path)
    track_id = track_id or path.stem
    pm = pretty_midi.PrettyMIDI(str(path))
    events = []
    for inst in pm.instruments:
        for note in inst.notes:
            canonical = ADTOF_PITCH_TO_CANONICAL.get(note.pitch)
            if canonical is not None:
                events.append(DrumEvent(float(note.start), canonical))
    return DrumAnnotation(track_id, events)


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
    for time, pitch, _ in rows:
        canonical = ADTOF_PITCH_TO_CANONICAL.get(pitch)
        if canonical is not None:
            events.append(DrumEvent(time, canonical))
    return DrumAnnotation(track_id, events)
