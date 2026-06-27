import pytest

from drumml.data.adtof import annotation_from_adtof_labels
from drumml.data.mdb import MDB_CLASS_LABELS
from drumml.events import DrumAnnotation
from drumml.taxonomy import Canonical


def test_mdb_label_map_round_trips_through_label_rows():
    rows = [(0.0, "KD"), (0.5, "SD"), (0.7, "CHH"), (1.0, "ZZZ")]  # ZZZ unmapped
    ann = DrumAnnotation.from_label_rows("t", rows, MDB_CLASS_LABELS)
    assert [e.canonical for e in ann.events] == [
        Canonical.KICK,
        Canonical.SNARE,
        Canonical.HH_CLOSED,
    ]


def test_mdb_strict_raises_on_unmapped():
    with pytest.raises(KeyError):
        DrumAnnotation.from_label_rows("t", [(0.0, "???")], MDB_CLASS_LABELS, strict=True)


def test_adtof_label_parser(tmp_path):
    f = tmp_path / "song.txt"
    f.write_text("0.000000\t36\n0.500000\t38\n1.000000\t42\n1.500000\t49\n")
    ann = annotation_from_adtof_labels(f)
    assert ann.track_id == "song"
    assert [e.canonical for e in ann.events] == [
        Canonical.KICK,
        Canonical.SNARE,
        Canonical.HH_CLOSED,
        Canonical.CRASH,
    ]


def test_egmd_midi_round_trip(tmp_path):
    pretty_midi = pytest.importorskip("pretty_midi")
    from drumml.data.egmd import annotation_from_midi

    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True)
    for t, pitch in [(0.0, 36), (0.5, 38), (1.0, 42)]:
        drum.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=t, end=t + 0.05))
    pm.instruments.append(drum)
    midi_path = tmp_path / "beat.mid"
    pm.write(str(midi_path))

    ann = annotation_from_midi(midi_path)
    assert [e.canonical for e in ann.events] == [
        Canonical.KICK,
        Canonical.SNARE,
        Canonical.HH_CLOSED,
    ]
    assert ann.events[0].velocity == 100
