"""Tests for the MT3/T5-style seq2seq transcription model (drumml.model)."""

from __future__ import annotations

import torch

from drumml.model import Seq2SeqADT, Seq2SeqConfig


def _tiny_config(**overrides) -> Seq2SeqConfig:
    """A fast CPU-sized config matching the step-3 contract."""
    base = dict(
        feature_dim=128,
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    base.update(overrides)
    return Seq2SeqConfig(**base)


def test_config_defaults():
    """Required fields are positional; the rest match the contract defaults."""
    cfg = Seq2SeqConfig(feature_dim=128, vocab_size=50)
    assert cfg.feature_dim == 128
    assert cfg.vocab_size == 50
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.n_encoder_layers == 6
    assert cfg.n_decoder_layers == 6
    assert cfg.dim_feedforward == 1024
    assert cfg.dropout == 0.1
    assert cfg.max_len == 2048


def test_forward_shape():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = Seq2SeqADT(cfg)

    features = torch.randn(2, 20, 128)
    tgt = torch.randint(0, cfg.vocab_size, (2, 7))

    logits = model(features, tgt)

    assert logits.shape == (2, 7, cfg.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_greedy_decode_shape_and_bos():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = Seq2SeqADT(cfg)

    features = torch.randn(2, 20, 128)
    bos_id, eos_id, max_len = 1, 2, 10

    out = model.greedy_decode(features, bos_id=bos_id, eos_id=eos_id, max_len=max_len)

    assert out.dtype == torch.int64
    assert out.dim() == 2
    assert out.size(0) == 2
    assert 1 <= out.size(1) <= max_len
    # Every sequence must start with BOS.
    assert torch.equal(out[:, 0], torch.full((2,), bos_id, dtype=torch.int64))


def test_greedy_decode_stops_at_eos():
    """If EOS is forced to be the argmax, decoding should terminate early."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = Seq2SeqADT(cfg)
    model.eval()

    # Bias the output head so EOS always wins -> sequences stop after 1 step.
    eos_id = 2
    with torch.no_grad():
        model.output_head.bias.zero_()
        model.output_head.bias[eos_id] = 1e4

    features = torch.randn(2, 20, 128)
    out = model.greedy_decode(features, bos_id=1, eos_id=eos_id, max_len=10)

    # [BOS, EOS] => length 2, shorter than max_len.
    assert out.size(1) == 2
    assert torch.equal(out[:, 1], torch.full((2,), eos_id, dtype=torch.int64))


def test_padding_mask_path_runs():
    """Both forward and greedy_decode must accept bool padding masks (True=PAD)."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = Seq2SeqADT(cfg)

    features = torch.randn(2, 20, 128)
    tgt = torch.randint(0, cfg.vocab_size, (2, 7))

    feature_padding_mask = torch.zeros(2, 20, dtype=torch.bool)
    feature_padding_mask[0, 15:] = True  # last 5 encoder frames are padding
    feature_padding_mask[1, 18:] = True

    tgt_padding_mask = torch.zeros(2, 7, dtype=torch.bool)
    tgt_padding_mask[0, 5:] = True  # last 2 target tokens are padding

    logits = model(
        features,
        tgt,
        feature_padding_mask=feature_padding_mask,
        tgt_padding_mask=tgt_padding_mask,
    )
    assert logits.shape == (2, 7, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    out = model.greedy_decode(
        features,
        bos_id=1,
        eos_id=2,
        max_len=8,
        feature_padding_mask=feature_padding_mask,
    )
    assert out.size(0) == 2
    assert out.size(1) <= 8


def test_forward_is_causal():
    """A target token's logits must not depend on future target tokens."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = Seq2SeqADT(cfg)
    model.eval()

    features = torch.randn(1, 12, 128)
    tgt = torch.randint(3, cfg.vocab_size, (1, 6))

    with torch.no_grad():
        base = model(features, tgt)
        # Change the LAST token; earlier-position logits must be unchanged.
        tgt2 = tgt.clone()
        tgt2[0, -1] = (tgt2[0, -1] + 1) % cfg.vocab_size
        changed = model(features, tgt2)

    assert torch.allclose(base[:, :-1, :], changed[:, :-1, :], atol=1e-5)
