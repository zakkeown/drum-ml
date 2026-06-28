"""Tests for the audio front-end feature extractors (drumml.features)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drumml.features import LogMelFrontend, MERTFrontend


def test_module_imports_without_transformers():
    """Importing the module / constructing MERTFrontend must not need transformers."""
    fe = MERTFrontend()
    assert fe.feature_dim == 1024
    assert fe.frame_hz == 75.0


def test_logmel_attributes():
    fe = LogMelFrontend()
    assert fe.feature_dim == 128
    # frame_hz = sample_rate / hop_length = 44100 / 441 = 100.0
    assert fe.frame_hz == pytest.approx(100.0)


def test_logmel_shape_and_finite():
    fe = LogMelFrontend(sample_rate=44100, n_mels=128, hop_length=441)
    audio = np.random.randn(44100).astype(np.float32)  # 1 s mono

    out = fe(audio)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    n_frames, feat = out.shape
    assert feat == 128
    # ~100 frames for 1 s at 100 Hz frame rate (centered STFT -> ~101).
    assert 95 <= n_frames <= 105
    assert torch.isfinite(out).all()


def test_logmel_accepts_torch_tensor():
    fe = LogMelFrontend()
    audio = torch.randn(44100)
    out = fe(audio)
    assert out.shape[1] == 128
    assert torch.isfinite(out).all()


def test_logmel_resample_path():
    """Passing sr != sample_rate must exercise the resample branch."""
    fe = LogMelFrontend(sample_rate=44100, n_mels=128, hop_length=441)
    # 1 s of audio at 22050 Hz; resampled up to 44100 -> still ~1 s -> ~100 frames.
    audio = np.random.randn(22050).astype(np.float32)

    out = fe(audio, sr=22050)

    n_frames, feat = out.shape
    assert feat == 128
    assert 95 <= n_frames <= 105
    assert torch.isfinite(out).all()


def test_mert_skips_without_weights():
    """MERT requires transformers + weights; skip cleanly if unavailable."""
    pytest.importorskip("transformers")
    pytest.skip("Avoid downloading MERT weights in CI; interface validated elsewhere.")
