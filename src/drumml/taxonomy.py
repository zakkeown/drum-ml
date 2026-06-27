"""Canonical drum taxonomy and deterministic reductions.

The whole pipeline normalizes every data source onto one canonical, fine-grained
class set (keyed on the General MIDI percussion map), then collapses
deterministically to the benchmark taxonomies (3 / 5 / 8 class). This is the
"train rich, evaluate at any granularity" principle from the design doc, and it
is the load-bearing piece that lets a single model be scored against ENST, MDB,
ADTOF, etc. despite their inconsistent label sets.

Reductions return ``None`` for canonical classes that a given scheme does *not*
score (e.g. cymbals are not evaluated in the 3-class MIREX setting), so the
evaluator can drop them rather than mislabel them.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class Canonical(str, Enum):
    """The 14-class canonical (fine) drum vocabulary.

    Declaration order is intentional — it is conventional drum-kit order (kick,
    snare, auxiliary, toms, hats, cymbals) and drives the column order of
    evaluation reports via :func:`scheme_classes`.
    """

    KICK = "kick"
    SNARE = "snare"
    SIDESTICK = "sidestick"
    CLAP = "clap"
    TOM_HI = "tom_hi"
    TOM_MID = "tom_mid"
    TOM_LO = "tom_lo"
    HH_CLOSED = "hh_closed"
    HH_PEDAL = "hh_pedal"
    HH_OPEN = "hh_open"
    CRASH = "crash"
    RIDE = "ride"
    RIDE_BELL = "ride_bell"
    PERC = "perc"  # tambourine, cowbell, bongos, congas, claves, ... (auxiliary)


# General MIDI percussion key map (channel 10), notes 35-81. Verified against the
# GM spec; this is the anchor every dataset's labels are translated through.
GM_PERCUSSION: dict[int, str] = {
    35: "Acoustic Bass Drum", 36: "Bass Drum 1", 37: "Side Stick",
    38: "Acoustic Snare", 39: "Hand Clap", 40: "Electric Snare",
    41: "Low Floor Tom", 42: "Closed Hi-Hat", 43: "High Floor Tom",
    44: "Pedal Hi-Hat", 45: "Low Tom", 46: "Open Hi-Hat",
    47: "Low-Mid Tom", 48: "Hi-Mid Tom", 49: "Crash Cymbal 1",
    50: "High Tom", 51: "Ride Cymbal 1", 52: "Chinese Cymbal",
    53: "Ride Bell", 54: "Tambourine", 55: "Splash Cymbal",
    56: "Cowbell", 57: "Crash Cymbal 2", 58: "Vibraslap",
    59: "Ride Cymbal 2", 60: "Hi Bongo", 61: "Low Bongo",
    62: "Mute Hi Conga", 63: "Open Hi Conga", 64: "Low Conga",
    65: "High Timbale", 66: "Low Timbale", 67: "High Agogo",
    68: "Low Agogo", 69: "Cabasa", 70: "Maracas", 71: "Short Whistle",
    72: "Long Whistle", 73: "Short Guiro", 74: "Long Guiro", 75: "Claves",
    76: "Hi Wood Block", 77: "Low Wood Block", 78: "Mute Cuica",
    79: "Open Cuica", 80: "Mute Triangle", 81: "Open Triangle",
}

# GM percussion note -> canonical class.
GM_NOTE_TO_CANONICAL: dict[int, Canonical] = {
    35: Canonical.KICK, 36: Canonical.KICK,
    37: Canonical.SIDESTICK,
    38: Canonical.SNARE, 40: Canonical.SNARE,
    39: Canonical.CLAP,
    41: Canonical.TOM_LO, 43: Canonical.TOM_LO,
    42: Canonical.HH_CLOSED,
    44: Canonical.HH_PEDAL,
    46: Canonical.HH_OPEN,
    45: Canonical.TOM_MID, 47: Canonical.TOM_MID,
    48: Canonical.TOM_HI, 50: Canonical.TOM_HI,
    49: Canonical.CRASH, 52: Canonical.CRASH, 55: Canonical.CRASH, 57: Canonical.CRASH,
    51: Canonical.RIDE, 59: Canonical.RIDE,
    53: Canonical.RIDE_BELL,
}
# Everything else in 35-81 (tambourine, cowbell, bongos, congas, ...) -> PERC.
for _note in GM_PERCUSSION:
    GM_NOTE_TO_CANONICAL.setdefault(_note, Canonical.PERC)


def gm_note_to_canonical(note: int) -> Optional[Canonical]:
    """Map a GM percussion note number to a canonical class (None if out of range)."""
    return GM_NOTE_TO_CANONICAL.get(int(note))


# --- Reductions -------------------------------------------------------------
# Each scheme maps every canonical class to a reduced-class name, or None if the
# scheme does not score that class. Side stick is folded into snare; hand clap is
# folded into snare where a scheme has no clap class, and dropped in 3-class.

C = Canonical

_SCHEME_3: dict[Canonical, Optional[str]] = {  # MIREX / IDMT: KD, SD, HH only
    C.KICK: "KD", C.SNARE: "SD", C.SIDESTICK: "SD", C.CLAP: None,
    C.HH_CLOSED: "HH", C.HH_PEDAL: "HH", C.HH_OPEN: "HH",
    C.TOM_HI: None, C.TOM_MID: None, C.TOM_LO: None,
    C.CRASH: None, C.RIDE: None, C.RIDE_BELL: None, C.PERC: None,
}

_SCHEME_5: dict[Canonical, Optional[str]] = {  # ADTOF: KD, SD, TT, HH, CY
    C.KICK: "KD", C.SNARE: "SD", C.SIDESTICK: "SD", C.CLAP: "SD",
    C.HH_CLOSED: "HH", C.HH_PEDAL: "HH", C.HH_OPEN: "HH",
    C.TOM_HI: "TT", C.TOM_MID: "TT", C.TOM_LO: "TT",
    C.CRASH: "CY", C.RIDE: "CY", C.RIDE_BELL: "CY", C.PERC: None,
}

_SCHEME_8: dict[Canonical, Optional[str]] = {  # ours: split hats + cymbals, keep side stick
    C.KICK: "KD", C.SNARE: "SD", C.SIDESTICK: "SS", C.CLAP: "SD",
    C.HH_CLOSED: "HHC", C.HH_PEDAL: "HHC", C.HH_OPEN: "HHO",
    C.TOM_HI: "TT", C.TOM_MID: "TT", C.TOM_LO: "TT",
    C.CRASH: "CR", C.RIDE: "RD", C.RIDE_BELL: "RD", C.PERC: None,
}

_SCHEME_CANONICAL: dict[Canonical, Optional[str]] = {c: c.value for c in Canonical}

SCHEMES: dict[str, dict[Canonical, Optional[str]]] = {
    "3": _SCHEME_3,
    "5": _SCHEME_5,
    "8": _SCHEME_8,
    "canonical": _SCHEME_CANONICAL,
}


def reduce(canonical: Canonical, scheme: str) -> Optional[str]:
    """Collapse a canonical class to a reduced-class name, or None if not scored."""
    try:
        mapping = SCHEMES[scheme]
    except KeyError:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}") from None
    return mapping[canonical]


def scheme_classes(scheme: str) -> list[str]:
    """Ordered list of reduced-class names for a scheme (the columns of a report)."""
    mapping = SCHEMES[scheme] if scheme in SCHEMES else None
    if mapping is None:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}")
    # Order by the canonical enum (kit order), not by dict-literal insertion order,
    # so report columns are stable regardless of how each scheme map is written.
    seen: list[str] = []
    for canonical in Canonical:
        name = mapping[canonical]
        if name is not None and name not in seen:
            seen.append(name)
    return seen
