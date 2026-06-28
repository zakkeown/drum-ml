"""Save / load a trained transcriber (model weights + config + tokenizer params).

A checkpoint is self-describing: it carries the ``Seq2SeqConfig`` and the
``DrumTokenizer`` settings, so ``load_checkpoint`` reconstructs a ready-to-run
(model, tokenizer) pair with no out-of-band config. Requires the ``model`` extra.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from drumml.model import Seq2SeqADT, Seq2SeqConfig
from drumml.tokenize import DrumTokenizer

FORMAT_VERSION = 1


def tokenizer_params(tokenizer: DrumTokenizer) -> dict:
    """The constructor kwargs needed to rebuild an equivalent tokenizer."""
    return {
        "scheme": tokenizer.scheme,
        "segment_seconds": tokenizer.segment_seconds,
        "frame_hz": tokenizer.frame_hz,
        "with_velocity": tokenizer.with_velocity,
        "velocity_bins": tokenizer.velocity_bins,
    }


def save_checkpoint(
    path: str | Path,
    model: Seq2SeqADT,
    config: Seq2SeqConfig,
    tokenizer: DrumTokenizer,
) -> None:
    """Write a checkpoint bundling model state, model config, and tokenizer params."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model_state": model.state_dict(),
            "config": asdict(config),
            "tokenizer": tokenizer_params(tokenizer),
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[Seq2SeqADT, DrumTokenizer, Seq2SeqConfig]:
    """Reconstruct ``(model, tokenizer, config)`` from a checkpoint, in eval mode."""
    # weights_only=True is safe here: the checkpoint holds only tensors and plain
    # dicts (config/tokenizer metadata), no custom classes — and it blocks the
    # arbitrary-code-execution risk of the default pickle path.
    payload = torch.load(path, map_location=device, weights_only=True)
    config = Seq2SeqConfig(**payload["config"])
    model = Seq2SeqADT(config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    tokenizer = DrumTokenizer(**payload["tokenizer"])
    return model, tokenizer, config
