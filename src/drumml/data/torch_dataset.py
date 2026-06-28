"""Segmenting :class:`torch.utils.data.Dataset` over :class:`Track`s.

Turns each drum track into fixed-length ``(features, target tokens)`` training
items for the MT3-style seq2seq stack. The audio front-end and the audio loader
are *injected* callables so the dataset is fully testable with no real audio:

* ``frontend(waveform, sr) -> Tensor(T, F)`` converts a (segment) waveform into
  a frame-by-feature matrix.
* ``load_audio(path) -> (np.ndarray mono, sr)`` reads an audio file. The default
  is a lazy ``soundfile`` reader, imported only when first used.

A flat segment index ``[(track_idx, segment_start), ...]`` is built up front,
spanning each track's duration. Duration is derived from the loaded audio length;
if a track has no (readable) ``audio_path`` it falls back to
``last_annotation_onset + segment_seconds`` so annotation-only tracks still
yield at least one segment.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from drumml.data.base import Track
from drumml.events import DrumAnnotation


def _default_load_audio(path) -> tuple[np.ndarray, int]:
    """Lazy ``soundfile``-based mono loader (only imported when first called)."""
    import soundfile as sf  # lazy: keeps the module importable without soundfile

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav)
    if wav.ndim > 1:  # (frames, channels) -> mono
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False), int(sr)


def _last_onset(ann: DrumAnnotation) -> float:
    """Time (seconds) of the last event, or 0.0 for an empty annotation."""
    return max((ev.time for ev in ann.events), default=0.0)


class ADTSegmentDataset(torch.utils.data.Dataset):
    """Segments a sequence of :class:`Track`s into seq2seq training items."""

    def __init__(
        self,
        tracks: Sequence[Track],
        tokenizer,
        frontend: Callable[[object, int], torch.Tensor],
        *,
        segment_seconds: float = 2.048,
        hop_seconds: Optional[float] = None,
        sample_rate: int = 44100,
        load_audio: Optional[Callable[[object], tuple[np.ndarray, int]]] = None,
    ) -> None:
        self.tracks: list[Track] = list(tracks)
        self.tokenizer = tokenizer
        self.frontend = frontend
        self.segment_seconds = float(segment_seconds)
        self.hop_seconds = (
            float(hop_seconds) if hop_seconds is not None else self.segment_seconds
        )
        self.sample_rate = int(sample_rate)
        self.load_audio = load_audio if load_audio is not None else _default_load_audio

        # Features (sliced by self.segment_seconds) and tokens (encoded with the
        # tokenizer's own segment length) must cover the same window, else they
        # silently describe different spans of audio.
        tok_seg = getattr(self.tokenizer, "segment_seconds", None)
        if tok_seg is not None and abs(float(tok_seg) - self.segment_seconds) > 1e-9:
            warnings.warn(
                "ADTSegmentDataset.segment_seconds "
                f"({self.segment_seconds}) != tokenizer.segment_seconds ({tok_seg}); "
                "features and target tokens will cover different windows.",
                stacklevel=2,
            )

        # Per-track: whether usable audio was found (drives the silence fallback
        # in __getitem__ so init-duration and getitem-waveform stay consistent).
        self._has_audio: list[bool] = []
        # Flat segment index spanning each track's duration.
        self.index: list[tuple[int, float]] = []
        for track_idx, track in enumerate(self.tracks):
            duration, has_audio = self._track_duration(track)
            self._has_audio.append(has_audio)
            for k in range(self._n_segments(duration)):
                self.index.append((track_idx, k * self.hop_seconds))

        # Single-entry waveform cache: a DataLoader visits a track's segments
        # consecutively, so this avoids re-reading the file for every segment.
        self._cache_idx: Optional[int] = None
        self._cache_wav: Optional[np.ndarray] = None
        self._cache_sr: Optional[int] = None

    # --- index construction -------------------------------------------------
    def _track_duration(self, track: Track) -> tuple[float, bool]:
        """Return ``(duration_seconds, has_usable_audio)`` for a track."""
        if track.audio_path is not None:
            try:
                wav, sr = self.load_audio(track.audio_path)
                n = np.asarray(wav).shape[0]
                if sr and n:
                    return n / float(sr), True
            except Exception:
                pass  # missing/unreadable -> fall back to annotation span
        return _last_onset(track.annotation) + self.segment_seconds, False

    def _n_segments(self, duration: float) -> int:
        """Number of hop-spaced segments covering ``duration`` (at least one)."""
        if self.hop_seconds <= 0:
            return 1
        # ceil with a tiny epsilon so a duration that is an exact multiple of the
        # hop does not spawn an empty trailing segment.
        return max(1, int(math.ceil(duration / self.hop_seconds - 1e-9)))

    # --- Dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        track_idx, segment_start = self.index[i]
        track = self.tracks[track_idx]

        waveform, sr = self._segment_waveform(track_idx, track, segment_start)
        features = self.frontend(waveform, sr)
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)

        token_ids = self.tokenizer.encode(track.annotation, segment_start)
        tokens = torch.tensor(token_ids, dtype=torch.long)

        return {
            "features": features,
            "tokens": tokens,
            "track_id": track.track_id,
            "segment_start": float(segment_start),
        }

    # --- waveform access ----------------------------------------------------
    def _segment_waveform(
        self, track_idx: int, track: Track, segment_start: float
    ) -> tuple[np.ndarray, int]:
        """Slice (or synthesize) the waveform for one segment."""
        if self._has_audio[track_idx]:
            waveform, sr = self._load_cached(track_idx, track)
            seg_samples = int(round(self.segment_seconds * sr))
            start_sample = int(round(segment_start * sr))
            seg = waveform[start_sample : start_sample + seg_samples]
            return seg, sr
        # Annotation-only track: feed the front-end one segment of silence.
        sr = self.sample_rate
        seg_samples = int(round(self.segment_seconds * sr))
        return np.zeros(seg_samples, dtype=np.float32), sr

    def _load_cached(self, track_idx: int, track: Track) -> tuple[np.ndarray, int]:
        if self._cache_idx != track_idx:
            wav, sr = self.load_audio(track.audio_path)
            wav = np.asarray(wav)
            if wav.ndim > 1:  # mono-mix any stray multichannel loader
                wav = wav.mean(axis=1)
            self._cache_idx, self._cache_wav, self._cache_sr = track_idx, wav, int(sr)
        assert self._cache_wav is not None and self._cache_sr is not None
        return self._cache_wav, self._cache_sr


def collate_segments(batch: Sequence[dict], pad_id: int) -> dict:
    """Pad a list of dataset items into batched tensors.

    Returns padding masks following the project convention: ``bool``, ``True`` at
    padded positions.
    """
    feats = [b["features"] for b in batch]
    toks = [b["tokens"] for b in batch]
    batch_size = len(batch)

    # --- features: pad along the time axis to the batch max ------------------
    feature_dim = feats[0].shape[1]
    t_max = max(f.shape[0] for f in feats)
    features = feats[0].new_zeros((batch_size, t_max, feature_dim))
    feature_padding_mask = torch.ones((batch_size, t_max), dtype=torch.bool)
    for i, f in enumerate(feats):
        t = f.shape[0]
        features[i, :t] = f
        feature_padding_mask[i, :t] = False

    # --- tokens: pad with pad_id to the batch max ---------------------------
    l_max = max(t.shape[0] for t in toks)
    tokens = torch.full((batch_size, l_max), pad_id, dtype=torch.long)
    tgt_padding_mask = torch.ones((batch_size, l_max), dtype=torch.bool)
    for i, t in enumerate(toks):
        length = t.shape[0]
        tokens[i, :length] = t
        tgt_padding_mask[i, :length] = False

    return {
        "features": features,
        "feature_padding_mask": feature_padding_mask,
        "tokens": tokens,
        "tgt_padding_mask": tgt_padding_mask,
    }
