"""End-to-end: train -> checkpoint -> load -> transcribe a dataset -> score.

Exercises the pieces a real run depends on, with no real audio: checkpoint
round-trip fidelity, dataset-level transcription with an injected loader, and
that the predictions are scorable by the eval harness.
"""

from pathlib import Path

import numpy as np
import torch

from drumml.checkpoint import load_checkpoint, save_checkpoint
from drumml.data.base import Track
from drumml.data.torch_dataset import ADTSegmentDataset
from drumml.eval import aggregate, score_track
from drumml.events import DrumAnnotation, DrumEvent
from drumml.model import Seq2SeqADT, Seq2SeqConfig
from drumml.taxonomy import Canonical
from drumml.tokenize import DrumTokenizer
from drumml.train import train
from drumml.transcribe import transcribe_dataset

FEAT_DIM = 8


def _frontend(waveform, sr):
    return torch.zeros(int(round(2.048 * 100)), FEAT_DIM)


def _load_audio_stub(path):
    return np.zeros(int(5.0 * 22050), dtype=np.float32), 22050


def _toy_tracks():
    return [
        Track("a", DrumAnnotation("a", [DrumEvent(0.1, Canonical.KICK)]), audio_path=Path("a.wav")),
        Track("b", DrumAnnotation("b", [DrumEvent(0.2, Canonical.SNARE)]), audio_path=Path("b.wav")),
    ]


def _config(tok):
    return Seq2SeqConfig(
        feature_dim=FEAT_DIM,
        vocab_size=tok.vocab_size,
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dim_feedforward=32,
    )


def test_pick_device():
    from drumml.train import pick_device

    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda") == "cuda"  # explicit choices pass through
    assert pick_device("auto") in {"mps", "cuda", "cpu"}


def test_checkpoint_round_trip_is_faithful(tmp_path):
    torch.manual_seed(0)
    tok = DrumTokenizer()
    cfg = _config(tok)
    model = Seq2SeqADT(cfg).eval()

    feats = torch.randn(1, 30, FEAT_DIM)
    out_before = model.greedy_decode(feats, tok.bos_id, tok.eos_id, max_len=12)

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, cfg, tok)
    model2, tok2, cfg2 = load_checkpoint(ckpt)

    assert cfg2 == cfg
    assert tok2.vocab_size == tok.vocab_size and tok2.scheme == tok.scheme
    out_after = model2.greedy_decode(feats, tok2.bos_id, tok2.eos_id, max_len=12)
    assert torch.equal(out_before, out_after)  # weights restored exactly


def test_train_save_load_transcribe_score(tmp_path):
    torch.manual_seed(0)
    tok = DrumTokenizer()
    tracks = _toy_tracks()
    dataset = ADTSegmentDataset(tracks, tok, _frontend, load_audio=_load_audio_stub)

    model = Seq2SeqADT(_config(tok))
    history = train(model, dataset, tok.pad_id, epochs=3, batch_size=2, lr=1e-2)
    assert len(history) >= 3

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, _config(tok), tok)
    model2, tok2, _ = load_checkpoint(ckpt)

    preds = transcribe_dataset(
        model2, tracks, tok2, _frontend, load_audio=_load_audio_stub, max_len=16
    )
    assert set(preds) == {"a", "b"}

    scores = [score_track(t.annotation, preds[t.track_id], tok2.scheme) for t in tracks]
    ds = aggregate(scores, name="toy")
    assert ds.n_tracks == 2
    assert 0.0 <= ds.macro_f_pooled <= 1.0  # a valid, scorable result


def test_transcribe_dataset_skips_unreadable_track():
    torch.manual_seed(0)
    tok = DrumTokenizer()
    model = Seq2SeqADT(_config(tok))

    def bad_loader(path):
        raise FileNotFoundError(path)

    tracks = _toy_tracks()
    import pytest

    with pytest.warns(UserWarning):
        preds = transcribe_dataset(
            model, tracks, tok, _frontend, load_audio=bad_loader, max_len=8
        )
    assert preds == {}  # both skipped, no crash
