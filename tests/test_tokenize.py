import numpy as np
import pytest

from drumml.events import DrumAnnotation, DrumEvent
from drumml.taxonomy import Canonical
from drumml.tokenize import DrumTokenizer


def _assert_onsets_equal(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_allclose(a[name], b[name])


def test_special_ids_fixed_and_distinct():
    tok = DrumTokenizer()
    assert (tok.pad_id, tok.bos_id, tok.eos_id) == (0, 1, 2)
    assert len({tok.pad_id, tok.bos_id, tok.eos_id}) == 3


def test_vocab_size_default_layout():
    tok = DrumTokenizer(scheme="5", segment_seconds=2.048, frame_hz=100.0, with_velocity=False)
    # 3 special + (round(2.048*100)+1 = 206) time + 5 class ("5") + 0 velocity
    assert tok.n_time_tokens == 206
    assert tok.classes == ["KD", "SD", "TT", "HH", "CY"]
    assert tok.vocab_size == 3 + 206 + 5
    assert tok.vocab_size == 214


def test_vocab_size_with_velocity():
    tok = DrumTokenizer(with_velocity=True, velocity_bins=4)
    assert tok.vocab_size == 3 + 206 + 5 + 4


def test_token_ranges_tile_the_vocab():
    # Every id 0..vocab_size-1 is exactly one of: special / time / class / velocity.
    tok = DrumTokenizer(with_velocity=True, velocity_bins=4)
    specials = {tok.pad_id, tok.bos_id, tok.eos_id}
    for i in range(tok.vocab_size):
        kinds = [
            i in specials,
            tok._is_time(i),
            tok._is_class(i),
            tok._is_velocity(i),
        ]
        assert sum(kinds) == 1, f"id {i} classified into {sum(kinds)} ranges"


def test_encode_wraps_in_bos_eos():
    tok = DrumTokenizer()
    ann = DrumAnnotation("t", [DrumEvent(0.0, Canonical.KICK), DrumEvent(0.5, Canonical.SNARE)])
    seq = tok.encode(ann)
    assert seq[0] == tok.bos_id
    assert seq[-1] == tok.eos_id
    # bos/eos appear exactly once and only at the ends
    assert seq.count(tok.bos_id) == 1
    assert seq.count(tok.eos_id) == 1


def test_round_trip_lossless_at_scheme_level():
    # All onsets on the 100 Hz grid and inside the default window.
    # Snare + clap at the same time both reduce to "SD" -> two SD onsets.
    ann = DrumAnnotation(
        "song",
        [
            DrumEvent(0.0, Canonical.KICK),
            DrumEvent(0.5, Canonical.SNARE),
            DrumEvent(0.5, Canonical.CLAP),
            DrumEvent(0.5, Canonical.HH_CLOSED),
            DrumEvent(1.0, Canonical.TOM_HI),
            DrumEvent(1.5, Canonical.CRASH),
        ],
    )
    tok = DrumTokenizer(scheme="5")
    back = tok.decode(tok.encode(ann))
    _assert_onsets_equal(back.onsets_by_class("5"), ann.onsets_by_class("5"))
    # Spot-check the representative-canonical inverse: SD -> SNARE (first in kit order).
    sd_canons = {e.canonical for e in back.events if e.time == 0.5 and e.canonical == Canonical.SNARE}
    assert sd_canons == {Canonical.SNARE}


def test_round_trip_with_segment_start_offset():
    ann = DrumAnnotation(
        "song",
        [
            DrumEvent(0.5, Canonical.KICK),   # before window -> excluded
            DrumEvent(1.0, Canonical.KICK),
            DrumEvent(2.0, Canonical.SNARE),
            DrumEvent(5.0, Canonical.HH_CLOSED),  # after window -> excluded
        ],
    )
    tok = DrumTokenizer(scheme="5")
    start = 1.0  # window [1.0, 3.048)
    back = tok.decode(tok.encode(ann, segment_start=start), segment_start=start)
    expected = DrumAnnotation(
        "win",
        [DrumEvent(1.0, Canonical.KICK), DrumEvent(2.0, Canonical.SNARE)],
    )
    _assert_onsets_equal(back.onsets_by_class("5"), expected.onsets_by_class("5"))


def test_events_outside_window_excluded():
    tok = DrumTokenizer(scheme="5")  # window [0, 2.048)
    ann = DrumAnnotation(
        "t",
        [
            DrumEvent(0.0, Canonical.KICK),
            DrumEvent(2.048, Canonical.SNARE),  # exactly at end -> excluded (half-open)
            DrumEvent(3.0, Canonical.SNARE),    # past end -> excluded
        ],
    )
    seq = tok.encode(ann)
    # Only the kick survives: BOS, TIME, CLASS, EOS
    n_class_tokens = sum(1 for t in seq if tok._is_class(t))
    assert n_class_tokens == 1
    back = tok.decode(seq)
    assert len(back.events) == 1
    assert back.events[0].canonical == Canonical.KICK


def test_unscored_classes_excluded():
    # PERC is not scored by scheme "5" -> dropped on encode.
    tok = DrumTokenizer(scheme="5")
    ann = DrumAnnotation(
        "t",
        [DrumEvent(0.0, Canonical.KICK), DrumEvent(0.3, Canonical.PERC)],
    )
    seq = tok.encode(ann)
    assert sum(1 for t in seq if tok._is_class(t)) == 1
    back = tok.decode(seq)
    _assert_onsets_equal(back.onsets_by_class("5"), ann.onsets_by_class("5"))


def test_with_velocity_emits_one_velocity_token_per_event():
    tok = DrumTokenizer(scheme="5", with_velocity=True, velocity_bins=4)
    ann = DrumAnnotation(
        "t",
        [
            DrumEvent(0.0, Canonical.KICK, velocity=110),
            DrumEvent(0.5, Canonical.SNARE, velocity=20),
            DrumEvent(1.0, Canonical.HH_CLOSED, velocity=64),
        ],
    )
    seq = tok.encode(ann)
    n_events = 3
    assert sum(1 for t in seq if tok._is_velocity(t)) == n_events
    assert sum(1 for t in seq if tok._is_class(t)) == n_events
    # Layout: BOS + 3*(TIME,CLASS,VELOCITY) + EOS
    assert len(seq) == 2 + 3 * n_events
    # Velocities decode into the valid MIDI range and the round-trip still holds.
    back = tok.decode(seq)
    assert all(1 <= e.velocity <= 127 for e in back.events)
    _assert_onsets_equal(back.onsets_by_class("5"), ann.onsets_by_class("5"))


def test_decode_ignores_padding_after_eos():
    tok = DrumTokenizer(scheme="5")
    ann = DrumAnnotation("t", [DrumEvent(0.0, Canonical.KICK)])
    seq = tok.encode(ann) + [tok.pad_id] * 5
    back = tok.decode(seq)
    assert len(back.events) == 1


def test_other_schemes_round_trip():
    ann = DrumAnnotation(
        "t",
        [
            DrumEvent(0.0, Canonical.KICK),
            DrumEvent(0.25, Canonical.HH_OPEN),
            DrumEvent(0.5, Canonical.HH_CLOSED),
            DrumEvent(0.75, Canonical.RIDE),
            DrumEvent(1.0, Canonical.SIDESTICK),
        ],
    )
    for scheme in ("3", "5", "8", "canonical"):
        tok = DrumTokenizer(scheme=scheme)
        back = tok.decode(tok.encode(ann))
        _assert_onsets_equal(back.onsets_by_class(scheme), ann.onsets_by_class(scheme))
