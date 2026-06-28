import pytest

from drumml.data.adtof import annotation_from_adtof_labels
from drumml.data.mdb import MDB_CLASS_LABELS, MDBDrumsAdapter
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


def test_mdb_adapter_real_layout(tmp_path):
    """The official checkout nests under 'MDB Drums/' and uses suffix-based joins.

    Annotation stems carry '_class' while audio carries '_MIX', so the adapter
    must derive the bare 'MusicDelta_<Genre>' track id and match audio by suffix.
    """
    base = tmp_path / "MDB Drums"
    ann_dir = base / "annotations" / "class"
    mix_dir = base / "audio" / "full_mix"
    ann_dir.mkdir(parents=True)
    mix_dir.mkdir(parents=True)
    # Real files have stray spaces around the tab: "<onset> \t <label> ".
    (ann_dir / "MusicDelta_Rock_class.txt").write_text(
        "0.000000 \t KD \n0.020000 \t HH \n0.528254 \t SD \n0.700000 \t OT \n"
    )
    (mix_dir / "MusicDelta_Rock_MIX.wav").write_bytes(b"")  # presence is enough

    tracks = list(MDBDrumsAdapter(tmp_path).tracks())
    assert len(tracks) == 1
    t = tracks[0]
    assert t.track_id == "MusicDelta_Rock"
    assert t.audio_path == mix_dir / "MusicDelta_Rock_MIX.wav"
    # OT -> PERC drops out of the 5-class view; KD/HH/SD survive.
    assert [e.canonical for e in t.annotation.events] == [
        Canonical.KICK,
        Canonical.HH_CLOSED,
        Canonical.SNARE,
        Canonical.PERC,
    ]
    assert set(t.annotation.onsets_by_class("5")) == {"KD", "HH", "SD"}


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


def test_adtof_midi_parser_maps_labels5(tmp_path):
    """ADTOF-pytorch emits MIDI at LABELS_5 pitches 35/38/47/42/49 -> KD/SD/TT/HH/CY."""
    pretty_midi = pytest.importorskip("pretty_midi")
    from drumml.data.adtof import annotation_from_adtof_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for t, pitch in [(0.0, 35), (0.25, 38), (0.5, 47), (0.75, 42), (1.0, 49)]:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=t, end=t + 0.05))
    pm.instruments.append(inst)
    midi_path = tmp_path / "song.mid"
    pm.write(str(midi_path))

    ann = annotation_from_adtof_midi(midi_path)
    assert ann.track_id == "song"
    assert [e.canonical for e in ann.events] == [
        Canonical.KICK,
        Canonical.SNARE,
        Canonical.TOM_MID,
        Canonical.HH_CLOSED,
        Canonical.CRASH,
    ]
    # the 5 classes reduce 1:1 onto the 5-class scheme used for scoring
    assert set(ann.onsets_by_class("5")) == {"KD", "SD", "TT", "HH", "CY"}


def _write_drum_midi(pretty_midi, path, pitch=36):
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=0.05))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_egmd_adapter_hf_mirror_layout(tmp_path):
    """schism-audio/e-gmd mirror: metadata.csv with midi_path/audio_path columns."""
    pretty_midi = pytest.importorskip("pretty_midi")
    from drumml.data.egmd import EGMDAdapter

    midi_rel = "midi/test/acoustic-kit/drummer1/eval_session/x.midi"
    audio_rel = "audio/test/acoustic-kit/drummer1/eval_session/x.wav"
    _write_drum_midi(pretty_midi, tmp_path / midi_rel, pitch=36)
    midi_rel_tr = "midi/train/acoustic-kit/drummer1/session1/y.midi"
    _write_drum_midi(pretty_midi, tmp_path / midi_rel_tr, pitch=38)
    (tmp_path / "metadata.csv").write_text(
        "file_name,split,midi_path,audio_path\n"
        f"{audio_rel},test,{midi_rel},{audio_rel}\n"
        f"audio/train/acoustic-kit/drummer1/session1/y.wav,train,{midi_rel_tr},audio/train/acoustic-kit/drummer1/session1/y.wav\n"
    )

    tracks = list(EGMDAdapter(tmp_path).tracks())
    assert len(tracks) == 2
    test_track = next(t for t in tracks if t.track_id == "x")
    assert test_track.annotation.events[0].canonical is Canonical.KICK
    assert test_track.audio_path == tmp_path / audio_rel

    # split filtering uses the metadata 'split' column
    only_test = list(EGMDAdapter(tmp_path, split="test").tracks())
    assert [t.track_id for t in only_test] == ["x"]


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
