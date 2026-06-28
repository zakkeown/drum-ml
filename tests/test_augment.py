"""Tests for train-time accompaniment-mixing augmentation (drumml.data.augment)."""

from __future__ import annotations

import numpy as np
import pytest

from drumml.data.augment import AccompanimentMixer, MixingFrontend, _rms


def _write_wav(path, x, sr=44100):
    import soundfile as sf

    sf.write(str(path), np.asarray(x, dtype=np.float32), sr)


def test_mixer_achieves_target_snr(tmp_path):
    """At prob=1 and a fixed SNR, mixed RMS ratio must match the requested SNR."""
    pytest.importorskip("soundfile")
    rng = np.random.default_rng(0)
    # long accompaniment so chunks are random offsets of real signal
    _write_wav(tmp_path / "acc.wav", rng.standard_normal(44100 * 5))

    drum = rng.standard_normal(44100).astype(np.float32)
    for snr_db in (-6.0, 0.0, 6.0):
        mixer = AccompanimentMixer(tmp_path, [snr_db], prob=1.0, seed=1)
        mixed = mixer.mix(drum, 44100)
        acc_component = mixed - drum  # exact, since mix() adds acc to the drum
        measured = 20 * np.log10(_rms(drum) / _rms(acc_component))
        assert abs(measured - snr_db) < 0.1  # accompaniment scaled to the target ratio


def test_negative_snr_makes_accompaniment_louder(tmp_path):
    """SNR < 0 must put MORE energy in accompaniment than drums (the key regime)."""
    pytest.importorskip("soundfile")
    rng = np.random.default_rng(0)
    _write_wav(tmp_path / "acc.wav", rng.standard_normal(44100 * 3))
    drum = rng.standard_normal(44100).astype(np.float32)

    mixer = AccompanimentMixer(tmp_path, [-6.0], prob=1.0, seed=2)
    acc_component = mixer.mix(drum, 44100) - drum
    assert _rms(acc_component) > _rms(drum)  # accompaniment-dominant mix


def test_prob_zero_is_identity(tmp_path):
    pytest.importorskip("soundfile")
    _write_wav(tmp_path / "acc.wav", np.random.default_rng(0).standard_normal(44100 * 2))
    drum = np.random.default_rng(1).standard_normal(44100).astype(np.float32)

    mixer = AccompanimentMixer(tmp_path, [0.0], prob=0.0, seed=0)
    assert np.array_equal(mixer.mix(drum, 44100), drum)


def test_prob_controls_augmentation_fraction(tmp_path):
    pytest.importorskip("soundfile")
    _write_wav(tmp_path / "acc.wav", np.random.default_rng(0).standard_normal(44100 * 2))
    drum = np.ones(2048, dtype=np.float32)

    mixer = AccompanimentMixer(tmp_path, [0.0], prob=0.5, seed=0)
    augmented = sum(not np.array_equal(mixer.mix(drum, 44100), drum) for _ in range(400))
    assert 150 <= augmented <= 250  # ~50% over 400 draws


def test_no_accompaniment_files_is_identity(tmp_path):
    drum = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    mixer = AccompanimentMixer(tmp_path, [0.0], prob=1.0, seed=0)  # empty dir
    assert np.array_equal(mixer.mix(drum, 44100), drum)


def test_silent_drum_segment_is_untouched(tmp_path):
    pytest.importorskip("soundfile")
    _write_wav(tmp_path / "acc.wav", np.random.default_rng(0).standard_normal(44100 * 2))
    silent = np.zeros(2048, dtype=np.float32)
    mixer = AccompanimentMixer(tmp_path, [0.0], prob=1.0, seed=0)
    assert np.array_equal(mixer.mix(silent, 44100), silent)


def test_accompaniment_shorter_than_segment_is_tiled(tmp_path):
    pytest.importorskip("soundfile")
    _write_wav(tmp_path / "acc.wav", np.random.default_rng(0).standard_normal(500))  # < segment
    drum = np.random.default_rng(1).standard_normal(2048).astype(np.float32)
    mixer = AccompanimentMixer(tmp_path, [0.0], prob=1.0, seed=0)
    mixed = mixer.mix(drum, 44100)
    assert mixed.shape == drum.shape  # tiled to length, no crash
    assert not np.array_equal(mixed, drum)


def test_mixing_frontend_delegates_and_mixes(tmp_path):
    """MixingFrontend mixes then calls the base front-end; passes through metadata."""
    pytest.importorskip("soundfile")
    from drumml.features import LogMelFrontend

    _write_wav(tmp_path / "acc.wav", np.random.default_rng(0).standard_normal(44100 * 2))
    base = LogMelFrontend()
    mixer = AccompanimentMixer(tmp_path, [0.0], prob=1.0, seed=0)
    fe = MixingFrontend(base, mixer)

    assert fe.feature_dim == base.feature_dim and fe.frame_hz == base.frame_hz
    drum = np.random.default_rng(1).standard_normal(44100).astype(np.float32)
    clean_feat = base(drum, 44100)
    mixed_feat = fe(drum, 44100)
    assert mixed_feat.shape == clean_feat.shape
    assert not np.allclose(mixed_feat, clean_feat)  # accompaniment changed the spectrogram
