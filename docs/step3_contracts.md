# Step 3 integration contracts (MT3-style seq2seq stack)

These signatures are FIXED so independently-built modules integrate cleanly. Match
names and shapes exactly. Build on the existing `drumml` core
(`taxonomy.py`, `events.py`, `data/`). Tests must run dataset-free (synthetic
fixtures / injected stubs) on CPU.

Shared conventions:
- Onset times in seconds. Default frame grid = 100 Hz (10 ms hop).
- Reduced class names come from `drumml.taxonomy.scheme_classes(scheme)`.
- Torch tensors are float32; token tensors are int64.

---

## 1. `src/drumml/tokenize.py`  →  `tests/test_tokenize.py`

MT3-style absolute-time event tokenizer. Pure Python (no torch).

```python
class DrumTokenizer:
    def __init__(self, scheme: str = "5", segment_seconds: float = 2.048,
                 frame_hz: float = 100.0, with_velocity: bool = False,
                 velocity_bins: int = 4): ...

    # special ids (fixed): pad_id=0, bos_id=1, eos_id=2
    @property
    def vocab_size(self) -> int: ...
    pad_id: int; bos_id: int; eos_id: int

    def encode(self, ann: "DrumAnnotation", segment_start: float = 0.0) -> list[int]:
        """Tokens for events in [segment_start, segment_start+segment_seconds).
        Layout: [BOS, (TIME, CLASS[, VELOCITY])*, EOS], events sorted by (time, class order).
        TIME = absolute time within the segment quantized to the frame grid
        (n_time_tokens = round(segment_seconds*frame_hz)+1). One TIME token per event
        (repeats allowed) keeps decoding simple."""

    def decode(self, tokens: list[int], segment_start: float = 0.0) -> "DrumAnnotation":
        """Inverse. Map each reduced class name back to a representative Canonical
        (first c in Canonical with reduce(c, scheme)==name). Round-trip is lossless
        at the scheme level: encode->decode preserves `onsets_by_class(scheme)` for
        onsets that lie on the grid (test THAT, not event-object equality)."""
```
Tests: vocab_size accounting; ids distinct; encode wraps in BOS/EOS; **round-trip** an annotation whose onsets sit on the grid and assert equal `onsets_by_class(scheme)`; events outside the segment window are excluded; with_velocity path emits a velocity token per event.

---

## 2. `src/drumml/features.py`  →  `tests/test_features.py`

Audio front-ends. `__call__` returns `(n_frames, feature_dim)` float32.

```python
class LogMelFrontend:
    feature_dim: int        # = n_mels
    frame_hz: float         # = sample_rate / hop_length
    def __init__(self, sample_rate=44100, n_mels=128, n_fft=2048,
                 hop_length=441, f_min=20.0, f_max=None): ...
    def __call__(self, waveform, sr: int | None = None) -> "torch.Tensor":
        """waveform: 1D np.ndarray or torch.Tensor (mono). If sr given and != sample_rate,
        resample (torchaudio.functional.resample). Return log(mel + eps), shape (T, n_mels)."""

class MERTFrontend:
    feature_dim: int        # 1024 for MERT-v1-330M
    frame_hz: float         # 75.0
    def __init__(self, model_id="m-a-p/MERT-v1-330M", layer=10, device="cpu"): ...
    def __call__(self, waveform, sr=None) -> "torch.Tensor":  # (T, 1024)
        """Lazy-import transformers; resample to 24000; return hidden state at `layer`."""
```
Tests: LogMel on 1 s of random audio → shape (~100, 128) and finite; resample path runs; MERT test uses `pytest.importorskip("transformers")` and is allowed to be skipped (don't download weights in CI). transformers is NOT installed — keep its import lazy/inside methods.

---

## 3. `src/drumml/model/seq2seq.py`  (+ `src/drumml/model/__init__.py`)  →  `tests/test_model.py`

T5/MT3-style encoder-decoder. Uses torch (CPU-testable).

```python
@dataclass
class Seq2SeqConfig:
    feature_dim: int
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 2048

class Seq2SeqADT(torch.nn.Module):
    def __init__(self, config: Seq2SeqConfig): ...
    def forward(self, features, tgt_tokens,
                feature_padding_mask=None, tgt_padding_mask=None) -> "torch.Tensor":
        """features (B,T,feature_dim) -> Linear in-proj -> +sinusoidal pos -> encoder.
        tgt_tokens (B,L) -> embedding -> +pos -> decoder with causal mask + cross-attn.
        Returns logits (B, L, vocab_size). padding masks are bool, True = PAD."""
    @torch.no_grad()
    def greedy_decode(self, features, bos_id, eos_id, max_len,
                      feature_padding_mask=None) -> "torch.Tensor":  # (B, <=max_len) int64
```
Tests: tiny config (d_model=32, n_heads=4, layers=2, dim_ff=64, feature_dim=128, vocab_size=50); forward on random (B=2,T=20,128) + tgt (2,7) → logits (2,7,50); greedy_decode returns (2, L<=max_len) starting with bos_id. Keep `model/__init__.py` exporting `Seq2SeqADT, Seq2SeqConfig`.

---

## 4. `src/drumml/data/torch_dataset.py`  →  `tests/test_torch_dataset.py`

Segmenting torch `Dataset` over `Track`s. Frontend + audio loader are INJECTABLE so
tests need no real audio.

```python
class ADTSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, tracks, tokenizer, frontend, *,
                 segment_seconds=2.048, hop_seconds=None, sample_rate=44100,
                 load_audio=None):
        """tracks: Sequence[drumml.data.base.Track]. frontend: callable(waveform, sr)->Tensor(T,F).
        load_audio: callable(Path)->(np.ndarray mono, sr); default uses soundfile (lazy import).
        Builds a flat segment index [(track_idx, segment_start), ...] spanning each track's
        duration (derived from audio length, or from last annotation onset if audio missing)."""
    def __len__(self) -> int: ...
    def __getitem__(self, i) -> dict:
        # {"features": Tensor(T,F), "tokens": LongTensor(L), "track_id": str, "segment_start": float}

def collate_segments(batch, pad_id: int) -> dict:
    # {"features": (B,Tmax,F), "feature_padding_mask": (B,Tmax) bool True=PAD,
    #  "tokens": (B,Lmax) padded with pad_id, "tgt_padding_mask": (B,Lmax) bool True=PAD}
```
Tests: build 2 fake `Track`s with hand-made `DrumAnnotation`s (no audio file); inject a dummy frontend returning `torch.zeros(int(round(segment_seconds*100)), 16)` and a `load_audio` stub returning silence of a chosen duration; assert `len()`, item shapes/keys, and that `collate_segments` pads to the batch max and masks correctly.
