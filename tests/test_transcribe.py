"""Tests for the inference bridge + the things teacher-forcing on zeros can't catch.

The key test here is `test_encoder_is_actually_wired`: with all-zero features under
teacher forcing, a model whose encoder/cross-attention is disconnected would still
pass every other test in the suite. Here we give two examples IDENTICAL decoder
context but DIFFERENT features + DIFFERENT targets, overfit, then greedy-decode
from features alone (no teacher forcing). Only a wired encoder can produce two
different outputs — so this fails loudly if audio doesn't reach the predictions.
"""

import numpy as np
import torch

from drumml.data.torch_dataset import collate_segments
from drumml.events import DrumAnnotation, DrumEvent
from drumml.model import Seq2SeqADT, Seq2SeqConfig
from drumml.taxonomy import Canonical
from drumml.tokenize import DrumTokenizer
from drumml.train import compute_loss
from drumml.transcribe import transcribe


def _same(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


def _close(a: dict, b: dict, atol: float) -> bool:
    """Same classes and counts, with onset times matching within atol seconds."""
    if set(a) != set(b):
        return False
    return all(len(a[k]) == len(b[k]) and np.allclose(a[k], b[k], atol=atol) for k in a)


# --- stitching: multi-segment encode/decode keeps correct absolute time --------
def test_segment_stitching_preserves_absolute_times():
    tok = DrumTokenizer()  # segment_seconds=2.048, scheme="5"
    # events spanning two segments, all on the 10 ms grid
    ann = DrumAnnotation(
        "x",
        [
            DrumEvent(0.10, Canonical.KICK),
            DrumEvent(0.50, Canonical.SNARE),
            DrumEvent(2.20, Canonical.HH_CLOSED),   # in segment 1
            DrumEvent(2.60, Canonical.CRASH),       # in segment 1
        ],
    )
    starts = [0.0, tok.segment_seconds]
    stitched_events = []
    for start in starts:
        tokens = tok.encode(ann, start)
        stitched_events.extend(tok.decode(tokens, segment_start=start).events)
    stitched = DrumAnnotation("x", stitched_events)

    # Segment starts are multiples of segment_seconds (2.048 s), which are NOT on
    # the 10 ms grid, so reconstructed absolute times land within half a frame
    # (~5 ms) of the originals — exact classes/counts, times within tolerance
    # (and far inside the ±50 ms eval window).
    assert _close(stitched.onsets_by_class("5"), ann.onsets_by_class("5"), atol=0.011)


def test_decode_tolerates_malformed_ordering():
    tok = DrumTokenizer()
    # CLASS token before any TIME token, plus a stray pad — must not crash, and
    # must drop the orphan class.
    class_tok = tok._class_offset  # first class id
    time_tok = tok._time_offset + 5
    bad = [tok.bos_id, class_tok, time_tok, class_tok, tok.eos_id]
    ann = tok.decode(bad)
    assert len(ann.events) == 1  # only the class that followed a time survives


# --- the real proof: features must drive predictions --------------------------
def test_encoder_is_actually_wired():
    torch.manual_seed(0)
    tok = DrumTokenizer()
    feat_dim = 8
    n_frames = 50

    # Two inputs with the SAME decoder target structure but distinct features and
    # distinct classes at the SAME time -> only the encoder can disambiguate.
    featA = torch.zeros(1, n_frames, feat_dim)
    featA[..., 0] = 1.0
    featB = torch.zeros(1, n_frames, feat_dim)
    featB[..., 1] = 1.0
    annA = DrumAnnotation("A", [DrumEvent(0.10, Canonical.KICK)])    # -> KD
    annB = DrumAnnotation("B", [DrumEvent(0.10, Canonical.SNARE)])   # -> SD

    itemA = {"features": featA[0], "tokens": torch.tensor(tok.encode(annA), dtype=torch.long)}
    itemB = {"features": featB[0], "tokens": torch.tensor(tok.encode(annB), dtype=torch.long)}
    batch = collate_segments([itemA, itemB], tok.pad_id)

    model = Seq2SeqADT(
        Seq2SeqConfig(
            feature_dim=feat_dim,
            vocab_size=tok.vocab_size,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            dim_feedforward=128,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(400):
        opt.zero_grad()
        loss = compute_loss(model, batch, tok.pad_id)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05  # actually overfit

    # Greedy-decode each from FEATURES ALONE (no teacher forcing).
    outA = model.greedy_decode(featA, tok.bos_id, tok.eos_id, max_len=12)
    outB = model.greedy_decode(featB, tok.bos_id, tok.eos_id, max_len=12)
    decA = tok.decode(outA[0].tolist())
    decB = tok.decode(outB[0].tolist())

    assert _same(decA.onsets_by_class("5"), annA.onsets_by_class("5"))
    assert _same(decB.onsets_by_class("5"), annB.onsets_by_class("5"))
    # and the two outputs are genuinely different (not a coincidental tie)
    assert not _same(decA.onsets_by_class("5"), decB.onsets_by_class("5"))


# --- batched decode must equal per-segment decode -----------------------------
def test_batched_decode_matches_per_segment():
    import numpy as np

    from drumml.features import LogMelFrontend

    torch.manual_seed(0)
    tok = DrumTokenizer()
    fe = LogMelFrontend()
    model = Seq2SeqADT(
        Seq2SeqConfig(
            feature_dim=fe.feature_dim,
            vocab_size=tok.vocab_size,
            d_model=32,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            dim_feedforward=64,
        )
    ).eval()

    sr = 22050
    # 5.3 s is NOT a multiple of 2.048 s -> the last segment is shorter -> the
    # batched path pads it, exercising the encoder/cross-attn padding masks.
    wav = np.random.RandomState(0).randn(int(5.3 * sr)).astype("float32")

    per_seg = transcribe(model, wav, sr, tok, fe, max_len=12, batch_size=1)
    batched = transcribe(model, wav, sr, tok, fe, max_len=12, batch_size=8)

    # bit-identical events (times + classes), not just within tolerance
    assert [(round(e.time, 6), e.canonical) for e in per_seg.events] == [
        (round(e.time, 6), e.canonical) for e in batched.events
    ]


# --- the bridge itself: model + audio -> scorable annotation ------------------
def test_transcribe_returns_scorable_annotation_over_multiple_segments():
    torch.manual_seed(0)
    tok = DrumTokenizer()
    feat_dim = 8

    def frontend(waveform, sr):
        # deterministic dummy features sized to the segment grid
        return torch.zeros(int(round(tok.segment_seconds * 100)), feat_dim)

    model = Seq2SeqADT(
        Seq2SeqConfig(
            feature_dim=feat_dim,
            vocab_size=tok.vocab_size,
            d_model=16,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            dim_feedforward=32,
        )
    )
    # ~5 s of silence -> spans 3 segments of 2.048 s
    sr = 22050
    wav = np.zeros(int(5.0 * sr), dtype=np.float32)
    ann = transcribe(model, wav, sr, tok, frontend, max_len=16)

    # An untrained model may emit nonsense, but the bridge must return a valid,
    # scorable DrumAnnotation, and every onset must be a finite time >= 0.
    from drumml.eval import score_track

    assert isinstance(ann, DrumAnnotation)
    assert all(e.time >= 0 and np.isfinite(e.time) for e in ann.events)
    s = score_track(ann, ann, "5")  # self-score is well-defined
    assert 0.0 <= s.macro_f <= 1.0
