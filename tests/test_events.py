import numpy as np
import pytest

from drumml.events import DrumAnnotation, DrumEvent
from drumml.taxonomy import Canonical


def test_events_sort_by_time():
    ann = DrumAnnotation(
        "t",
        [DrumEvent(2.0, Canonical.KICK), DrumEvent(1.0, Canonical.SNARE)],
    )
    assert [e.time for e in ann.events] == [1.0, 2.0]


def test_velocity_bounds_validated():
    with pytest.raises(ValueError):
        DrumEvent(0.0, Canonical.KICK, velocity=200)
    with pytest.raises(ValueError):
        DrumEvent(-0.1, Canonical.KICK)


def test_from_gm_notes_skips_unmapped():
    ann = DrumAnnotation.from_gm_notes(
        "t",
        [(0.0, 36, 100), (0.5, 38, 90), (1.0, 20, 50)],  # 20 is out of range
    )
    assert [e.canonical for e in ann.events] == [Canonical.KICK, Canonical.SNARE]


def test_onsets_by_class_groups_and_drops_unscored():
    ann = DrumAnnotation.from_gm_notes(
        "t",
        [(0.0, 36, None), (0.5, 42, None), (0.7, 46, None), (1.0, 49, None)],
    )
    grouped = ann.onsets_by_class("5")
    assert set(grouped) == {"KD", "HH", "CY"}
    np.testing.assert_allclose(grouped["HH"], [0.5, 0.7])
    # at 3-class the cymbal is dropped entirely
    assert "CY" not in ann.onsets_by_class("3")


def test_tsv_round_trip(tmp_path):
    ann = DrumAnnotation(
        "song",
        [DrumEvent(0.0, Canonical.KICK, 100), DrumEvent(0.25, Canonical.HH_CLOSED)],
    )
    path = tmp_path / "song.tsv"
    ann.to_tsv(path)
    back = DrumAnnotation.from_tsv(path)
    assert back.track_id == "song"
    assert back.events[0] == DrumEvent(0.0, Canonical.KICK, 100)
    assert back.events[1].velocity is None
