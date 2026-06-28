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


def build_adapter(name: str, root: Path, split: str | None = None):
    if name == "egmd":
        from drumml.data.egmd import EGMDAdapter

        return EGMDAdapter(root, split=split)
    if name == "mdb":
        from drumml.data.mdb import MDBDrumsAdapter

        if split:
            print(f"(note: MDB has no splits; ignoring split {split!r})")
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
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (feed the GPU)")
    ap.add_argument("--split", default="train", help="train split (egmd: train/validation/test)")
    ap.add_argument("--limit", type=int, default=None, help="cap #tracks (smoke runs)")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="shuffle tracks with this seed BEFORE --limit, for a diverse "
                         "subset (E-GMD is ordered by groove, so the head is low-diversity)")
    ap.add_argument("--accompaniment-dir", type=Path, default=None,
                    help="dir of drum-free accompaniment wavs + snr_db.json; if set, "
                         "training segments are mixed with accompaniment (training-only)")
    ap.add_argument("--aug-prob", type=float, default=0.7,
                    help="fraction of segments to augment when --accompaniment-dir is set")
    ap.add_argument("--aug-seed", type=int, default=0, help="augmentation RNG seed")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/seq2seq.pt"),
                    help="checkpoint output path")
    ap.add_argument("--eval-after", action="store_true", help="score a held-out split after training")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--eval-limit", type=int, default=100)
    args = ap.parse_args(argv)

    from drumml.checkpoint import save_checkpoint
    from drumml.data.torch_dataset import ADTSegmentDataset
    from drumml.features import LogMelFrontend
    from drumml.model import Seq2SeqADT, Seq2SeqConfig
    from drumml.tokenize import DrumTokenizer
    from drumml.train import pick_device, train

    device = pick_device(args.device)
    print(f"device: {device}")

    adapter = build_adapter(args.dataset, args.root, args.split)
    tracks = list(adapter.tracks())
    if args.shuffle_seed is not None:
        import random

        random.Random(args.shuffle_seed).shuffle(tracks)
    if args.limit:
        tracks = tracks[: args.limit]
    print(f"{len(tracks)} tracks from {adapter.name}/{args.split}"
          + (f" (shuffled seed={args.shuffle_seed})" if args.shuffle_seed is not None else ""))

    tokenizer = DrumTokenizer(scheme=args.scheme)
    frontend = LogMelFrontend()  # bare front-end; eval uses this (real mixes need no aug)

    train_frontend = frontend
    if args.accompaniment_dir is not None:
        import json

        from drumml.data.augment import AccompanimentMixer, MixingFrontend

        snr = json.loads((args.accompaniment_dir / "snr_db.json").read_text())
        mixer = AccompanimentMixer(
            args.accompaniment_dir, snr, prob=args.aug_prob, seed=args.aug_seed
        )
        train_frontend = MixingFrontend(frontend, mixer)
        print(f"augment: {len(mixer.paths)} accompaniment tracks, prob={args.aug_prob}, "
              f"{len(snr)} empirical SNRs (median {sorted(snr)[len(snr)//2]:.1f} dB)")

    dataset = ADTSegmentDataset(tracks, tokenizer, train_frontend)
    print(f"{len(dataset)} segments")

    config = Seq2SeqConfig(
        feature_dim=frontend.feature_dim,
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
    )
    model = Seq2SeqADT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params, vocab={tokenizer.vocab_size}")

    import time

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    def log(step, loss):
        if step % 50 == 0:
            rate = step / (time.monotonic() - t0) if step else 0.0
            print(f"step {step:>7}  loss {loss:.4f}  ({rate:.1f} steps/s)", flush=True)

    def on_epoch(epoch, history):
        # Epoch-stamp every checkpoint: the OOD (cross-dataset) optimum usually
        # arrives *before* the in-domain optimum, so we need each epoch to sweep
        # later, not just the last. ~45 MB each at 11M params — cheap.
        done = epoch + 1
        ckpt = args.out.with_name(f"{args.out.stem}_epoch{done}{args.out.suffix}")
        save_checkpoint(ckpt, model, config, tokenizer)
        elapsed = time.monotonic() - t0
        eta = elapsed / done * (args.epochs - done)
        recent = history[-200:]
        mean_loss = sum(recent) / len(recent)
        print(
            f"== epoch {done}/{args.epochs}  mean-loss {mean_loss:.4f}  "
            f"elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m  -> saved {ckpt}",
            flush=True,
        )

    history = train(
        model,
        dataset,
        tokenizer.pad_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        num_workers=args.num_workers,
        on_step=log,
        on_epoch=on_epoch,
    )
    print(f"done. final loss {history[-1]:.4f}  total {(time.monotonic()-t0)/60:.1f}m")

    save_checkpoint(args.out, model, config, tokenizer)
    print(f"saved checkpoint -> {args.out}")

    if args.eval_after:
        from drumml.eval import aggregate, format_report, score_track
        from drumml.transcribe import transcribe_dataset

        eval_adapter = build_adapter(args.dataset, args.root, args.eval_split)
        eval_tracks = list(eval_adapter.tracks())[: args.eval_limit]
        print(f"\nevaluating on {len(eval_tracks)} {args.eval_split} tracks ...")
        preds = transcribe_dataset(model, eval_tracks, tokenizer, frontend, device=device)
        scores = [
            score_track(t.annotation, preds[t.track_id], args.scheme)
            for t in eval_tracks
            if t.track_id in preds
        ]
        if scores:
            print(format_report(aggregate(scores, name=f"{adapter.name}/{args.eval_split}")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
