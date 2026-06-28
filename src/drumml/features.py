"""Audio front-end feature extractors.

Two interchangeable front-ends turn a mono waveform into a
``(n_frames, feature_dim)`` float32 tensor:

* :class:`LogMelFrontend` — a classic log-mel spectrogram (torchaudio only;
  always available).
* :class:`MERTFrontend` — hidden states from the MERT self-supervised audio
  model. ``transformers`` is imported *lazily* inside the methods so that
  importing this module never requires ``transformers`` to be installed.

Both expose ``feature_dim`` and ``frame_hz`` attributes and are callable as
``frontend(waveform, sr=None) -> torch.Tensor`` of shape ``(T, feature_dim)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio


def _to_float_tensor(waveform) -> torch.Tensor:
    """Coerce a 1D mono np.ndarray or torch.Tensor to a 1D float32 torch tensor."""
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(waveform))
    elif torch.is_tensor(waveform):
        wav = waveform
    else:
        wav = torch.as_tensor(waveform)
    return wav.to(torch.float32)


class LogMelFrontend:
    """Log-mel spectrogram front-end built on torchaudio.

    Attributes:
        feature_dim: number of mel bands (``n_mels``).
        frame_hz: frame rate, ``sample_rate / hop_length``.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 441,
        f_min: float = 20.0,
        f_max: float | None = None,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.eps = 1e-6  # small floor for log compression

        self.feature_dim = n_mels
        self.frame_hz = sample_rate / hop_length

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

    @torch.no_grad()
    def __call__(self, waveform, sr: int | None = None) -> torch.Tensor:
        """Return ``log(mel + eps)`` of shape ``(T, n_mels)`` as float32.

        Args:
            waveform: 1D mono ``np.ndarray`` or ``torch.Tensor``.
            sr: sample rate of ``waveform``; if given and != ``sample_rate`` the
                signal is resampled to ``sample_rate`` first.
        """
        wav = _to_float_tensor(waveform)
        if sr is not None and sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        mel = self.mel(wav)  # (n_mels, n_frames)
        log_mel = torch.log(mel + self.eps)
        return log_mel.transpose(0, 1).contiguous().to(torch.float32)  # (n_frames, n_mels)


class MERTFrontend:
    """MERT (self-supervised) audio front-end.

    ``transformers`` is imported lazily inside the methods, so importing this
    module does not require ``transformers`` to be installed.

    Attributes:
        feature_dim: hidden size (1024 for MERT-v1-330M).
        frame_hz: frame rate of MERT hidden states (75 Hz).
    """

    TARGET_SR = 24000

    def __init__(
        self,
        model_id: str = "m-a-p/MERT-v1-330M",
        layer: int = 10,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.layer = layer
        self.device = device

        self.feature_dim = 1024
        self.frame_hz = 75.0

        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        """Lazily load the MERT model and its feature extractor."""
        if self._model is not None:
            return
        # Imported here on purpose: transformers is an optional, lazily loaded
        # dependency and must not be required to import this module.
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        self._model.eval()
        self._model.to(self.device)

    @torch.no_grad()
    def __call__(self, waveform, sr: int | None = None) -> torch.Tensor:
        """Return MERT hidden states at ``layer``, shape ``(T, 1024)`` float32.

        Args:
            waveform: 1D mono ``np.ndarray`` or ``torch.Tensor``.
            sr: sample rate of ``waveform``; resampled to 24000 Hz if needed.
        """
        self._ensure_loaded()

        wav = _to_float_tensor(waveform)
        if sr is not None and sr != self.TARGET_SR:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.TARGET_SR)

        inputs = self._processor(
            wav.cpu().numpy(),
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs, output_hidden_states=True)
        # hidden_states: tuple length (num_layers + 1); index 0 is the embedding
        # output, index `layer` is the `layer`-th transformer layer.
        hidden = outputs.hidden_states[self.layer]  # (1, T, 1024)
        return hidden.squeeze(0).to(torch.float32)
