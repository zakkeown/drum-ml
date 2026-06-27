"""Onset-based ADT evaluation (mir_eval), per-class and aggregated.

Standard ADT scoring: per-class onset Precision/Recall/F at a +/-50 ms tolerance,
then averaged. We report both **macro-F** (mean over classes that have reference
support — the headline ADT number) and **micro-F** (pool all hits — dominated by
frequent classes like kick/snare).

The aggregation helpers exist to make the design doc's headline metric easy:
*cross-dataset / OOD* macro-F. Score per track, aggregate per dataset, then
average dataset macro-F across held-out datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from mir_eval.util import match_events

from drumml.events import DrumAnnotation
from drumml.taxonomy import scheme_classes

DEFAULT_WINDOW = 0.05  # +/-50 ms, the mir_eval / ADT convention

# TODO: velocity-aware F (E-GMD has velocity labels) — a matched onset should
# also be scored on |vel_ref - vel_est|. Deferred until a velocity-emitting model
# exists (ADTOF emits none); see design doc steps 4-5.


def _prf(n_ref: int, n_est: int, tp: int) -> tuple[float, float, float]:
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return f, p, r


@dataclass
class ClassScore:
    cls: str
    f: float
    p: float
    r: float
    n_ref: int
    n_est: int
    tp: int


@dataclass
class TrackScore:
    track_id: str
    scheme: str
    per_class: dict[str, ClassScore]
    macro_f: float
    macro_p: float
    macro_r: float
    micro_f: float
    micro_p: float
    micro_r: float


def score_track(
    ref: DrumAnnotation,
    est: DrumAnnotation,
    scheme: str,
    window: float = DEFAULT_WINDOW,
) -> TrackScore:
    """Score one estimate against one reference for a taxonomy scheme."""
    ref_by = ref.onsets_by_class(scheme)
    est_by = est.onsets_by_class(scheme)
    classes = scheme_classes(scheme)

    per_class: dict[str, ClassScore] = {}
    macro_fs: list[float] = []
    macro_ps: list[float] = []
    macro_rs: list[float] = []
    tot_ref = tot_est = tot_tp = 0

    for cls in classes:
        r_on = ref_by.get(cls, np.empty(0))
        e_on = est_by.get(cls, np.empty(0))
        n_ref, n_est = len(r_on), len(e_on)
        if n_ref == 0 and n_est == 0:
            continue  # class absent on both sides — undefined, skip
        tp = len(match_events(r_on, e_on, window)) if n_ref and n_est else 0
        f, p, r = _prf(n_ref, n_est, tp)
        per_class[cls] = ClassScore(cls, f, p, r, n_ref, n_est, tp)
        tot_ref += n_ref
        tot_est += n_est
        tot_tp += tp
        # Macro averages over classes the *reference* contains (standard ADT).
        if n_ref > 0:
            macro_fs.append(f)
            macro_ps.append(p)
            macro_rs.append(r)

    macro_f = float(np.mean(macro_fs)) if macro_fs else 0.0
    macro_p = float(np.mean(macro_ps)) if macro_ps else 0.0
    macro_r = float(np.mean(macro_rs)) if macro_rs else 0.0
    micro_f, micro_p, micro_r = _prf(tot_ref, tot_est, tot_tp)

    return TrackScore(
        track_id=ref.track_id,
        scheme=scheme,
        per_class=per_class,
        macro_f=macro_f,
        macro_p=macro_p,
        macro_r=macro_r,
        micro_f=micro_f,
        micro_p=micro_p,
        micro_r=micro_r,
    )


@dataclass
class DatasetScore:
    """Dataset-level summary under both standard aggregation conventions.

    These two macro numbers are NOT interchangeable, and the gap is largest on a
    mix of short clips (E-GMD) and full songs (MDB):

    * ``macro_f_pooled`` — pool TP/FP/FN per class across the *whole dataset*,
      compute per-class F, then average classes. This is the convention used by
      Vogl/ADTOF/Wu-survey ADT papers, so it is the **headline** number to quote
      against published baselines.
    * ``macro_f_per_track`` — average each track's macro-F. Weights a 5 s clip
      equally with a full song. Useful, but not what the literature reports.

    ``per_class_f`` is the pooled per-class F (matches ``macro_f_pooled``).
    Hallucinated classes (no reference events, some predictions) lower
    ``micro_f`` but are excluded from the macro averages, which only span classes
    with reference support.
    """

    name: str
    scheme: str
    n_tracks: int
    macro_f_pooled: float          # ADT-standard headline
    macro_f_per_track: float       # mean of per-track macro-F
    micro_f: float                 # pooled over all tracks/classes
    per_class_f: dict[str, float]  # pooled per-class F (matches macro_f_pooled)
    track_scores: list[TrackScore]


def aggregate(
    track_scores: Sequence[TrackScore],
    name: str = "dataset",
) -> DatasetScore:
    """Aggregate per-track scores into a dataset-level summary (both conventions)."""
    if not track_scores:
        raise ValueError("no track scores to aggregate")
    scheme = track_scores[0].scheme

    macro_f_per_track = float(np.mean([t.macro_f for t in track_scores]))

    # Pool raw counts per class across every track (the literature convention).
    counts: dict[str, list[int]] = {}  # cls -> [n_ref, n_est, tp]
    tot_ref = tot_est = tot_tp = 0
    for t in track_scores:
        for cls, cs in t.per_class.items():
            acc = counts.setdefault(cls, [0, 0, 0])
            acc[0] += cs.n_ref
            acc[1] += cs.n_est
            acc[2] += cs.tp
            tot_ref += cs.n_ref
            tot_est += cs.n_est
            tot_tp += cs.tp

    per_class_f: dict[str, float] = {}
    pooled_class_fs: list[float] = []
    for cls, (n_ref, n_est, tp) in counts.items():
        f, _, _ = _prf(n_ref, n_est, tp)
        per_class_f[cls] = f
        if n_ref > 0:  # macro spans only reference-present classes
            pooled_class_fs.append(f)

    macro_f_pooled = float(np.mean(pooled_class_fs)) if pooled_class_fs else 0.0
    micro_f, _, _ = _prf(tot_ref, tot_est, tot_tp)

    return DatasetScore(
        name=name,
        scheme=scheme,
        n_tracks=len(track_scores),
        macro_f_pooled=macro_f_pooled,
        macro_f_per_track=macro_f_per_track,
        micro_f=micro_f,
        per_class_f=per_class_f,
        track_scores=list(track_scores),
    )


def cross_dataset_macro_f(
    dataset_scores: Iterable[DatasetScore],
    convention: str = "pooled",
) -> float:
    """Headline OOD number: mean macro-F across held-out datasets.

    ``convention`` is "pooled" (default, ADT-standard) or "per_track".
    """
    attr = {"pooled": "macro_f_pooled", "per_track": "macro_f_per_track"}.get(convention)
    if attr is None:
        raise ValueError("convention must be 'pooled' or 'per_track'")
    fs = [getattr(d, attr) for d in dataset_scores]
    return float(np.mean(fs)) if fs else 0.0


def format_report(score: DatasetScore) -> str:
    """Human-readable per-class + summary table, self-labeling the convention."""
    lines = [
        f"== {score.name}  (scheme={score.scheme}, tracks={score.n_tracks}) ==",
        "per-class F = pooled TP/FP/FN across dataset",
        f"{'class':<8} {'F':>6}",
    ]
    for cls in scheme_classes(score.scheme):
        if cls in score.per_class_f:
            lines.append(f"{cls:<8} {score.per_class_f[cls]:>6.3f}")
    lines.append("-" * 24)
    lines.append(f"{'macro-F':<14} {score.macro_f_pooled:>6.3f}  (pooled, headline)")
    lines.append(f"{'macro-F':<14} {score.macro_f_per_track:>6.3f}  (per-track mean)")
    lines.append(f"{'micro-F':<14} {score.micro_f:>6.3f}  (pooled)")
    return "\n".join(lines)
