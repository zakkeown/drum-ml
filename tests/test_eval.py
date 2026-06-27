import numpy as np

from drumml.eval import aggregate, cross_dataset_macro_f, score_track
from drumml.events import DrumAnnotation, DrumEvent
from drumml.taxonomy import Canonical


def _ann(track_id, events):
    return DrumAnnotation(track_id, events)


def test_perfect_match_scores_one():
    ref = _ann("t", [DrumEvent(1.0, Canonical.KICK), DrumEvent(2.0, Canonical.SNARE)])
    est = _ann("t", [DrumEvent(1.0, Canonical.KICK), DrumEvent(2.0, Canonical.SNARE)])
    s = score_track(ref, est, "5")
    assert s.macro_f == 1.0
    assert s.micro_f == 1.0


def test_known_counts_within_and_outside_window():
    # kick: 1.0 matches 1.01 (<=50ms), 2.0 vs 2.5 misses -> tp=1, fp=1, fn=1
    ref = _ann("t", [DrumEvent(1.0, Canonical.KICK), DrumEvent(2.0, Canonical.KICK)])
    est = _ann("t", [DrumEvent(1.01, Canonical.KICK), DrumEvent(2.5, Canonical.KICK)])
    s = score_track(ref, est, "5")
    kd = s.per_class["KD"]
    assert (kd.tp, kd.n_ref, kd.n_est) == (1, 2, 2)
    assert kd.p == 0.5 and kd.r == 0.5
    assert abs(kd.f - 0.5) < 1e-9


def test_missing_class_in_estimate_is_zero_recall():
    ref = _ann("t", [DrumEvent(1.0, Canonical.KICK), DrumEvent(1.0, Canonical.SNARE)])
    est = _ann("t", [DrumEvent(1.0, Canonical.KICK)])  # snare missed entirely
    s = score_track(ref, est, "5")
    assert s.per_class["KD"].f == 1.0
    assert s.per_class["SD"].f == 0.0
    # macro over the two reference classes -> mean(1.0, 0.0) = 0.5
    assert abs(s.macro_f - 0.5) < 1e-9


def test_aggregate_and_cross_dataset():
    ref = _ann("t", [DrumEvent(1.0, Canonical.KICK)])
    good = score_track(ref, _ann("t", [DrumEvent(1.0, Canonical.KICK)]), "5")
    bad = score_track(ref, _ann("t", [DrumEvent(5.0, Canonical.KICK)]), "5")

    ds_a = aggregate([good, good], name="A")  # pooled KD F = 1.0
    ds_b = aggregate([good, bad], name="B")   # pooled KD: tp=1,ref=2,est=2 -> F=0.5
    assert abs(ds_a.macro_f_pooled - 1.0) < 1e-9
    assert abs(ds_b.macro_f_pooled - 0.5) < 1e-9
    assert abs(cross_dataset_macro_f([ds_a, ds_b]) - 0.75) < 1e-9


def test_pooled_vs_per_track_macro_diverge():
    # A short clip (1 event) and a long track (8 events) with uneven counts.
    # Track 1: kick only, perfect.
    t1 = score_track(
        _ann("clip", [DrumEvent(1.0, Canonical.KICK)]),
        _ann("clip", [DrumEvent(1.0, Canonical.KICK)]),
        "5",
    )
    # Track 2: 4 kicks (all hit) + 4 snares (all missed).
    ref2 = _ann("song", [DrumEvent(float(i), Canonical.KICK) for i in range(1, 5)]
                + [DrumEvent(float(i) + 0.5, Canonical.SNARE) for i in range(1, 5)])
    est2 = _ann("song", [DrumEvent(float(i), Canonical.KICK) for i in range(1, 5)])
    t2 = score_track(ref2, est2, "5")

    ds = aggregate([t1, t2], name="mix")
    # per-track: mean(1.0, mean(KD=1.0, SD=0.0)=0.5) = 0.75
    assert abs(ds.macro_f_per_track - 0.75) < 1e-9
    # pooled: KD ref=5/est=5/tp=5 -> 1.0 ; SD ref=4/est=0/tp=0 -> 0.0 ; mean = 0.5
    assert abs(ds.macro_f_pooled - 0.5) < 1e-9
    assert ds.macro_f_pooled != ds.macro_f_per_track  # convention matters


def test_window_widening_changes_match():
    ref = _ann("t", [DrumEvent(1.0, Canonical.KICK)])
    est = _ann("t", [DrumEvent(1.08, Canonical.KICK)])  # 80 ms off
    assert score_track(ref, est, "5", window=0.05).per_class["KD"].tp == 0
    assert score_track(ref, est, "5", window=0.10).per_class["KD"].tp == 1
