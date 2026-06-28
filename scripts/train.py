#!/usr/bin/env python
"""Train an MT3-style seq2seq drum transcriber from a dataset adapter.

Wires together: dataset adapter -> ADTSegmentDataset (LogMel features +
tokenized targets) -> Seq2SeqADT -> teacher-forced training loop. Needs real
audio, so this is the integration entry point (not exercised by unit tests).

Example:
    uv run python scripts/train.py --dataset egmd --root datasets/e-gmd \\
        --scheme 5 --epochs 5 --batch-size 16
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_adapter(name: str, root: Path):
    if name == "egmd":
        from drumml.data.egmd import EGMDAdapter

        return EGMDAdapter(root)
    if name == "mdb":
        from drumml.data.mdb import MDBDrumsAdapter

        return MDBDrumsAdapter(root)
    raise SystemExit(f"unknown dataset adapter {name!r} (have: egmd, mdb)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="adapter: egmd | mdb")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--scheme", default="5", choices=["3", "5", "8", "canonical"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="cap #tracks (smoke runs)")
    args = ap.parse_args(argv)

    from drumml.data.torch_dataset import ADTSegmentDataset
    from drumml.features import LogMelFrontend
    from drumml.model import Seq2SeqADT, Seq2SeqConfig
    from drumml.tokenize import DrumTokenizer
    from drumml.train import train

    adapter = build_adapter(args.dataset, args.root)
    tracks = list(adapter.tracks())
    if args.limit:
        tracks = tracks[: args.limit]
    print(f"{len(tracks)} tracks from {adapter.name}")

    tokenizer = DrumTokenizer(scheme=args.scheme)
    frontend = LogMelFrontend()
    dataset = ADTSegmentDataset(tracks, tokenizer, frontend)
    print(f"{len(dataset)} segments")

    model = Seq2SeqADT(
        Seq2SeqConfig(
            feature_dim=frontend.feature_dim,
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
        )
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params, vocab={tokenizer.vocab_size}")

    def log(step, loss):
        if step % 20 == 0:
            print(f"step {step:>6}  loss {loss:.4f}")

    history = train(
        model,
        dataset,
        tokenizer.pad_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        on_step=log,
    )
    print(f"done. final loss {history[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
