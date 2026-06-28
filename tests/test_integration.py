"""End-to-end: tokenizer -> dataset -> collate -> model -> loss -> backward.

This is the integration check the per-module tests can't give — it proves the four
step-3 modules agree on vocab size and tensor shapes and that gradients flow
through the whole stack on CPU, with no real audio.
"""

import torch

from drumml.data.base import Track
from drumml.data.torch_dataset import ADTSegmentDataset, collate_segments
from drumml.events import DrumAnnotation, DrumEvent
from drumml.model import Seq2SeqADT, Seq2SeqConfig
from drumml.taxonomy import Canonical
from drumml.tokenize import DrumTokenizer
from drumml.train import compute_loss, train

FEATURE_DIM = 16
SEG_FRAMES = int(round(2.048 * 100))  # 205


def _dummy_frontend(waveform, sr):
    # ignores audio, returns a fixed-shape feature map
    return torch.zeros(SEG_FRAMES, FEATURE_DIM)


def _toy_tracks():
    return [
        Track("a", DrumAnnotation("a", [DrumEvent(0.1, Canonical.KICK), DrumEvent(0.5, Canonical.SNARE)])),
        Track("b", DrumAnnotation("b", [DrumEvent(0.2, Canonical.HH_CLOSED), DrumEvent(0.9, Canonical.CRASH)])),
    ]


def _toy_model(vocab_size):
    return Seq2SeqADT(
        Seq2SeqConfig(
            feature_dim=FEATURE_DIM,
            vocab_size=vocab_size,
            d_model=32,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            dim_feedforward=64,
        )
    )


def test_vocab_size_flows_tokenizer_to_model():
    tok = DrumTokenizer()
    model = _toy_model(tok.vocab_size)
    # decoder embedding rows must match the tokenizer vocab exactly
    n_embeddings = model.state_dict()[
        next(k for k in model.state_dict() if "embed" in k.lower() and k.endswith("weight"))
    ].shape[0]
    assert n_embeddings == tok.vocab_size


def test_forward_backward_step():
    tok = DrumTokenizer()
    ds = ADTSegmentDataset(_toy_tracks(), tok, _dummy_frontend, load_audio=None)
    assert len(ds) >= 2
    batch = collate_segments([ds[i] for i in range(len(ds))], tok.pad_id)

    # shapes line up: features feed the encoder, tokens feed the decoder
    assert batch["features"].shape[-1] == FEATURE_DIM
    assert batch["tokens"].dtype == torch.int64

    model = _toy_model(tok.vocab_size)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    loss = compute_loss(model, batch, tok.pad_id)
    assert torch.isfinite(loss)
    loss.backward()

    # gradients flowed to at least some parameters
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())

    # an optimizer step actually changes weights
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.step()
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())


def test_train_loop_runs_and_overfits_tiny():
    tok = DrumTokenizer()
    ds = ADTSegmentDataset(_toy_tracks(), tok, _dummy_frontend, load_audio=None)
    model = _toy_model(tok.vocab_size)
    history = train(model, ds, tok.pad_id, epochs=5, batch_size=2, lr=1e-2)
    assert len(history) >= 5
    assert all(torch.isfinite(torch.tensor(h)) for h in history)
    # with a fixed (zero) feature map the targets are still learnable; loss should drop
    assert history[-1] < history[0]
