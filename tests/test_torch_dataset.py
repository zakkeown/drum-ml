"""Tests for the segmenting torch Dataset (drumml.data.torch_dataset).

No real audio: the front-end and audio loader are injected stubs.
"""

from __future__ import annotations

import numpy as np
import torch

from drumml.data.base import Track
from drumml.data.torch_dataset import ADTSegmentDataset, collate_segments
from drumml.events import DrumAnnotation, DrumEvent
from drumml.taxonomy import Canonical
from drumml.tokenize import DrumTokenizer

SR = 44100
SEG = 2.048
# (T, F) the dummy front-end always returns, regardless of the input waveform.
FRONTEND_T = int(round(SEG * 100))  # 205
FRONTEND_F = 16

# Track A has audio of exactly 5.0 s -> 5.0 / 2.048 = 2.4414... -> ceil = 3.
N_A = 220500  # 5.0 s at 44100 Hz
SEGS_A = 3
# Track B is annotation-only (no audio): fallback duration = last_onset + SEG.
# last_onset = 0.5 -> 2.548 / 2.048 = 1.244... -> ceil = 2.
SEGS_B = 2


def _frontend(waveform, sr):
    return torch.zeros(FRONTEND_T, FRONTEND_F)


def _load_audio_stub(path):
    return np.zeros(N_A, dtype=np.float32), SR


def _make_tracks():
    ann_a = DrumAnnotation(
        "track_a",
        [
            DrumEvent(0.0, Canonical.KICK),
            DrumEvent(0.5, Canonical.SNARE),
            DrumEvent(2.5, Canonical.HH_CLOSED),  # lands in the 2nd segment
            DrumEvent(4.5, Canonical.CRASH),      # lands in the 3rd segment
        ],
    )
    ann_b = DrumAnnotation(
        "track_b",
        [
            DrumEvent(0.0, Canonical.KICK),
            DrumEvent(0.5, Canonical.SNARE),  # last onset 0.5 -> 2 fallback segments
        ],
    )
    track_a = Track("track_a", ann_a, audio_path="a.wav")
    track_b = Track("track_b", ann_b, audio_path=None)
    return track_a, track_b


def _make_dataset():
    track_a, track_b = _make_tracks()
    return ADTSegmentDataset(
        [track_a, track_b],
        tokenizer=DrumTokenizer(),
        frontend=_frontend,
        segment_seconds=SEG,
        sample_rate=SR,
        load_audio=_load_audio_stub,
    )


def test_len_matches_expected_segment_count():
    ds = _make_dataset()
    assert len(ds) == SEGS_A + SEGS_B  # 3 + 2 == 5


def test_segment_index_starts_are_hop_spaced():
    ds = _make_dataset()
    starts_a = [s for (ti, s) in ds.index if ti == 0]
    starts_b = [s for (ti, s) in ds.index if ti == 1]
    assert starts_a == [0.0, SEG, 2 * SEG]
    assert starts_b == [0.0, SEG]


def test_item_keys_and_shapes():
    ds = _make_dataset()
    item = ds[0]
    assert set(item) == {"features", "tokens", "track_id", "segment_start"}

    assert isinstance(item["features"], torch.Tensor)
    assert item["features"].shape == (FRONTEND_T, FRONTEND_F)

    assert isinstance(item["tokens"], torch.Tensor)
    assert item["tokens"].dtype == torch.int64
    assert item["tokens"].ndim == 1
    # tokens are produced by the tokenizer for this segment -> BOS ... EOS.
    tok = ds.tokenizer
    assert item["tokens"][0].item() == tok.bos_id
    assert item["tokens"][-1].item() == tok.eos_id

    assert item["track_id"] == "track_a"
    assert item["segment_start"] == 0.0


def test_tokens_track_the_segment_window():
    """Each segment encodes only the events in its own time window."""
    ds = _make_dataset()
    tok = ds.tokenizer
    # Segment 0 of track_a ([0, 2.048)) has the kick+snare -> 2 class tokens.
    item0 = ds[0]
    n_class_0 = sum(1 for t in item0["tokens"].tolist() if tok._is_class(t))
    assert n_class_0 == 2
    # Segment 1 of track_a ([2.048, 4.096)) has the hat at 2.5 -> 1 class token.
    item1 = ds[1]
    assert item1["segment_start"] == SEG
    n_class_1 = sum(1 for t in item1["tokens"].tolist() if tok._is_class(t))
    assert n_class_1 == 1


def test_annotation_only_fallback_yields_segments():
    """A track with audio_path=None still produces >=1 segment."""
    ann = DrumAnnotation("solo", [DrumEvent(0.0, Canonical.KICK)])
    track = Track("solo", ann, audio_path=None)
    ds = ADTSegmentDataset(
        [track],
        tokenizer=DrumTokenizer(),
        frontend=_frontend,
        segment_seconds=SEG,
        sample_rate=SR,
        load_audio=_load_audio_stub,  # never called (no audio_path)
    )
    assert len(ds) >= 1
    # last_onset 0.0 -> duration SEG -> exactly one segment.
    assert len(ds) == 1
    item = ds[0]
    assert item["track_id"] == "solo"
    assert item["features"].shape == (FRONTEND_T, FRONTEND_F)


def test_collate_pads_features_and_tokens_and_masks():
    pad_id = 0
    item1 = {
        "features": torch.ones(FRONTEND_T, FRONTEND_F),  # T = 205
        "tokens": torch.tensor([1, 7, 8, 2], dtype=torch.long),  # L = 4
        "track_id": "a",
        "segment_start": 0.0,
    }
    item2 = {
        "features": torch.ones(100, FRONTEND_F),  # T = 100 (shorter)
        "tokens": torch.tensor([1, 2], dtype=torch.long),  # L = 2 (shorter)
        "track_id": "b",
        "segment_start": 0.0,
    }
    out = collate_segments([item1, item2], pad_id)

    assert set(out) == {
        "features",
        "feature_padding_mask",
        "tokens",
        "tgt_padding_mask",
    }

    # Padded to the batch max on both axes.
    assert out["features"].shape == (2, FRONTEND_T, FRONTEND_F)
    assert out["tokens"].shape == (2, 4)

    # Feature mask: bool, True == PAD. Row 0 fully valid; row 1 padded past 100.
    fmask = out["feature_padding_mask"]
    assert fmask.dtype == torch.bool
    assert not fmask[0].any()
    assert not fmask[1, :100].any()
    assert fmask[1, 100:].all()
    # Padded feature rows are zeros; valid rows preserved.
    assert torch.equal(out["features"][1, 100:], torch.zeros(FRONTEND_T - 100, FRONTEND_F))
    assert torch.equal(out["features"][1, :100], torch.ones(100, FRONTEND_F))

    # Token mask: bool, True == PAD. Row 1 padded past length 2 with pad_id.
    tmask = out["tgt_padding_mask"]
    assert tmask.dtype == torch.bool
    assert not tmask[0].any()
    assert not tmask[1, :2].any()
    assert tmask[1, 2:].all()
    assert out["tokens"].dtype == torch.int64
    assert out["tokens"][1].tolist() == [1, 2, pad_id, pad_id]


def test_collate_on_real_dataset_items():
    ds = _make_dataset()
    batch = [ds[0], ds[1], ds[3]]
    out = collate_segments(batch, ds.tokenizer.pad_id)

    b = len(batch)
    l_max = max(item["tokens"].shape[0] for item in batch)
    assert out["features"].shape == (b, FRONTEND_T, FRONTEND_F)
    assert out["feature_padding_mask"].shape == (b, FRONTEND_T)
    assert out["tokens"].shape == (b, l_max)
    assert out["tgt_padding_mask"].shape == (b, l_max)
    # All feature frames are real here (constant T) -> no feature padding.
    assert not out["feature_padding_mask"].any()
    # Wherever tokens equal pad_id beyond a row's real length, the mask is True.
    for i, item in enumerate(batch):
        real = item["tokens"].shape[0]
        assert not out["tgt_padding_mask"][i, :real].any()
        assert out["tgt_padding_mask"][i, real:].all()
