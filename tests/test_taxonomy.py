from drumml.taxonomy import (
    Canonical,
    GM_NOTE_TO_CANONICAL,
    gm_note_to_canonical,
    reduce,
    scheme_classes,
)


def test_gm_notes_map_to_expected_canonical():
    assert gm_note_to_canonical(36) is Canonical.KICK
    assert gm_note_to_canonical(38) is Canonical.SNARE
    assert gm_note_to_canonical(37) is Canonical.SIDESTICK
    assert gm_note_to_canonical(42) is Canonical.HH_CLOSED
    assert gm_note_to_canonical(46) is Canonical.HH_OPEN
    assert gm_note_to_canonical(49) is Canonical.CRASH
    assert gm_note_to_canonical(51) is Canonical.RIDE
    assert gm_note_to_canonical(53) is Canonical.RIDE_BELL
    assert gm_note_to_canonical(56) is Canonical.PERC  # cowbell -> perc bucket


def test_every_gm_note_has_a_canonical():
    assert all(n in GM_NOTE_TO_CANONICAL for n in range(35, 82))


def test_out_of_range_note_is_none():
    assert gm_note_to_canonical(20) is None


def test_reduction_3class():
    assert reduce(Canonical.KICK, "3") == "KD"
    assert reduce(Canonical.HH_OPEN, "3") == "HH"
    assert reduce(Canonical.SIDESTICK, "3") == "SD"
    assert reduce(Canonical.CRASH, "3") is None  # cymbals not scored at 3-class
    assert scheme_classes("3") == ["KD", "SD", "HH"]


def test_reduction_5class():
    assert reduce(Canonical.TOM_LO, "5") == "TT"
    assert reduce(Canonical.RIDE, "5") == "CY"
    assert reduce(Canonical.CRASH, "5") == "CY"
    assert reduce(Canonical.PERC, "5") is None
    assert scheme_classes("5") == ["KD", "SD", "TT", "HH", "CY"]


def test_reduction_8class_splits_hats_and_cymbals():
    assert reduce(Canonical.HH_CLOSED, "8") == "HHC"
    assert reduce(Canonical.HH_OPEN, "8") == "HHO"
    assert reduce(Canonical.CRASH, "8") == "CR"
    assert reduce(Canonical.RIDE, "8") == "RD"
    assert reduce(Canonical.SIDESTICK, "8") == "SS"


def test_canonical_scheme_is_identity():
    for c in Canonical:
        assert reduce(c, "canonical") == c.value
