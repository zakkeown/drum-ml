"""Training glue: tie tokenizer + dataset + model into a teacher-forced loop.

This is the integration layer over the four step-3 modules. The decoder is
trained teacher-forced: input is ``tokens[:, :-1]``, target is ``tokens[:, 1:]``,
cross-entropy ignores the pad id. Requires the ``model`` extra (torch).
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from drumml.data.torch_dataset import collate_segments


def pick_device(name: str = "auto") -> str:
    """Resolve "auto" to the best available backend (MPS on Apple Silicon)."""
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_loss(model: torch.nn.Module, batch: dict, pad_id: int) -> torch.Tensor:
    """Teacher-forced cross-entropy for one collated batch."""
    feats = batch["features"]
    fmask = batch["feature_padding_mask"]
    tokens = batch["tokens"]
    tmask = batch["tgt_padding_mask"]

    dec_in = tokens[:, :-1]
    target = tokens[:, 1:]
    dec_mask = tmask[:, :-1]

    logits = model(feats, dec_in, feature_padding_mask=fmask, tgt_padding_mask=dec_mask)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_id,
    )


def train(
    model: torch.nn.Module,
    dataset: Dataset,
    pad_id: int,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 3e-4,
    device: str = "cpu",
    num_workers: int = 0,
    on_step: Optional[Callable[[int, float], None]] = None,
) -> list[float]:
    """Minimal training loop. Returns the per-step loss history.

    ``on_step(global_step, loss)`` is an optional callback for logging.
    """
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(collate_segments, pad_id=pad_id),
    )

    history: list[float] = []
    step = 0
    for _ in range(epochs):
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            loss = compute_loss(model, batch, pad_id)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            if on_step is not None:
                on_step(step, history[-1])
            step += 1
    return history
