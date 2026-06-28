"""Train-time audio augmentation: mix isolated drums with drum-free accompaniment.

Our E-GMD-trained model over-generates on full mixes (it fires drums on
bass/guitar/vocal energy it has never heard). This augmentation closes that gap
by overlaying real **drum-free** accompaniment (summed bass+other+vocals stems,
e.g. from MUSDB18) onto each E-GMD drum segment at training time. The drum onset
labels are unchanged, so the model learns "this non-drum spectral energy is NOT
an onset" while keeping perfect targets.

Two pieces:

* :class:`AccompanimentMixer` — overlays a random accompaniment chunk at a random
  SNR drawn from an **empirical** distribution (the real drums-vs-accompaniment
  RMS ratios measured in the source dataset), so the range includes
  accompaniment-as-loud-as-or-louder-than-drums — the regime that actually
  teaches the model to ignore non-drum energy.
* :class:`MixingFrontend` — wraps a base front-end so the dataset is unchanged:
  it mixes the waveform, then delegates to the base front-end. Mixing is
  training-only; evaluation uses the bare front-end (real mixes need no mixing),
  keeping eval byte-identical to the no-augmentation runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


class AccompanimentMixer:
    """Overlay random drum-free accompaniment on a drum waveform at a random SNR.

    Args:
        accompaniment_dir: directory of mono accompaniment ``.wav`` files
            (precomputed bass+other+vocals sums; see ``scripts/prep_accompaniment.py``).
        snr_db_choices: empirical SNR values (dB, ``20*log10(rms_drums/rms_accomp)``)
            to sample the mix ratio from. Grounding this in the source dataset's real
            ratios is what puts accompaniment-dominant mixes in the training set.
        prob: probability a given segment is augmented (the rest stay clean so the
            model still handles isolated drums).
        seed: RNG seed (reproducible per process).
    """

    def __init__(
        self,
        accompaniment_dir: str | Path,
        snr_db_choices: Sequence[float],
        *,
        prob: float = 0.7,
        seed: int = 0,
        max_load_seconds: float = 30.0,
    ):
        self.accompaniment_dir = Path(accompaniment_dir)
        self.paths = sorted(self.accompaniment_dir.glob("*.wav"))
        self.snr_db_choices = np.asarray(list(snr_db_choices), dtype=np.float32)
        if self.snr_db_choices.size == 0:
            self.snr_db_choices = np.array([0.0], dtype=np.float32)
        self.prob = float(prob)
        # Cap per-file load length: full MUSDB tracks (~42 MB mono each) cached
        # across many DataLoader workers would OOM; 30 s/file still gives ample
        # distinct 2 s windows (~14/file x 150 files) for accompaniment variety.
        self.max_load_seconds = float(max_load_seconds)
        self._rng = np.random.default_rng(seed)
        self._cache: dict[Path, np.ndarray] = {}

    def _load(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is None:
            import soundfile as sf  # lazy

            n_frames = -1
            if self.max_load_seconds > 0:
                sr = sf.info(str(path)).samplerate
                n_frames = int(self.max_load_seconds * sr)
            wav, _ = sf.read(str(path), dtype="float32", always_2d=False, frames=n_frames)
            wav = np.asarray(wav)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            cached = np.ascontiguousarray(wav, dtype=np.float32)
            self._cache[path] = cached
        return cached

    def _random_chunk(self, n: int) -> np.ndarray:
        """A length-``n`` accompaniment chunk from a random file at a random offset."""
        acc = self._load(self.paths[int(self._rng.integers(len(self.paths)))])
        if acc.size == 0:
            return np.zeros(n, dtype=np.float32)
        if acc.size <= n:
            return np.resize(acc, n).astype(np.float32, copy=False)  # tile to length
        off = int(self._rng.integers(0, acc.size - n))
        return acc[off : off + n]

    def mix(self, drum_wav, sr: int) -> np.ndarray:
        """Return ``drum_wav`` with accompaniment overlaid (or unchanged, w.p. 1-prob)."""
        drum = np.asarray(drum_wav, dtype=np.float32)
        if not self.paths or self._rng.random() > self.prob:
            return drum
        d_rms = _rms(drum)
        if d_rms < 1e-6:  # silent drum tail: nothing meaningful to mix against
            return drum
        acc = self._random_chunk(drum.shape[-1])
        a_rms = _rms(acc)
        if a_rms < 1e-6:
            return drum
        # Scale accompaniment so 20*log10(d_rms / scaled_a_rms) == snr_db.
        snr_db = float(self.snr_db_choices[int(self._rng.integers(self.snr_db_choices.size))])
        target_a_rms = d_rms / (10.0 ** (snr_db / 20.0))
        return (drum + acc * (target_a_rms / a_rms)).astype(np.float32, copy=False)


class MixingFrontend:
    """Wrap a base front-end so each waveform is mixed before feature extraction.

    Exposes the base front-end's ``feature_dim``/``frame_hz`` so it is a drop-in
    replacement in :class:`~drumml.data.torch_dataset.ADTSegmentDataset`.
    """

    def __init__(self, base_frontend, mixer: AccompanimentMixer):
        self.base = base_frontend
        self.mixer = mixer
        self.feature_dim = base_frontend.feature_dim
        self.frame_hz = base_frontend.frame_hz

    def __call__(self, waveform, sr: Optional[int] = None):
        return self.base(self.mixer.mix(waveform, sr), sr)
