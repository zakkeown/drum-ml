"""Inference bridge: model + audio -> DrumAnnotation.

This is the connector between the two halves of the repo — it segments audio the
same way training does, runs the model's greedy decode per segment, decodes each
token sequence back to events at the correct absolute time, and concatenates.
The result is a ``DrumAnnotation`` that ``drumml.eval`` can score directly, which
is what makes a trained model *measurable*. Requires the ``model`` extra (torch).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from drumml.events import DrumAnnotation
from drumml.tokenize import DrumTokenizer


def _to_mono(waveform) -> np.ndarray:
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 2:
        # average channels; soundfile yields (frames, channels) so the channel
        # axis is the smaller one.
        ch_axis = 0 if arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ch_axis)
    return np.ascontiguousarray(arr, dtype=np.float32)


@torch.no_grad()
def transcribe(
    model: torch.nn.Module,
    waveform,
    sr: int,
    tokenizer: DrumTokenizer,
    frontend,
    *,
    segment_seconds: Optional[float] = None,
    hop_seconds: Optional[float] = None,
    max_len: int = 1024,
    device: str = "cpu",
    track_id: str = "transcribed",
) -> DrumAnnotation:
    """Transcribe a full waveform into a canonical ``DrumAnnotation``.

    Segments are non-overlapping by default (``hop_seconds == segment_seconds``),
    matching the tokenizer's per-segment absolute-time grid so concatenated
    events keep correct global timing.
    """
    segment_seconds = segment_seconds or tokenizer.segment_seconds
    hop_seconds = hop_seconds or segment_seconds

    wav = _to_mono(waveform)
    duration = len(wav) / sr
    n_segments = max(1, math.ceil(duration / hop_seconds - 1e-9))

    model.to(device).eval()
    events = []
    for k in range(n_segments):
        start = k * hop_seconds
        s = int(round(start * sr))
        e = int(round((start + segment_seconds) * sr))
        seg = wav[s:e]
        if seg.size == 0:
            continue
        feats = frontend(seg, sr)  # (T, F) float32
        feats = feats.to(device).unsqueeze(0)  # (1, T, F)
        tokens = model.greedy_decode(
            feats, tokenizer.bos_id, tokenizer.eos_id, max_len
        )
        seg_ann = tokenizer.decode(tokens[0].tolist(), segment_start=start)
        events.extend(seg_ann.events)

    return DrumAnnotation(track_id=track_id, events=events)
