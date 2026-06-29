"""A2MD adapter (Audio-to-MIDI Drum dataset, Wei et al.).

A2MD pairs **real full-mix audio** (internet-sourced songs, mono mp3) with drum
labels obtained by **DTW-aligning a separate Lakh-MIDI arrangement** to the
recording. So unlike E-GMD (exact rendered MIDI) these are *weak* labels: the
dominant error is **presence noise** — the aligned arrangement may contain hits
not played in the recording (and vice-versa), not just timing jitter. A2MD ships
the data bucketed by alignment distance ``dist0p00`` (tightest) … ``dist0p60``
(loosest); ``max_dist`` keeps only buckets at or below a threshold so callers can
trade label quality for quantity.

The MIDI is standard GM percussion, so parsing reuses the exact E-GMD path
(:func:`drumml.data.egmd.annotation_from_midi` → GM→canonical). Auxiliary
percussion in the Lakh arrangements (tambourine/maracas/triangle/…) maps to
``PERC`` and is dropped at the 3-/5-class schemes, so it never pollutes SD/HH.

Layout (after extracting ``a2md_public.zip``)::

    <root>/align_mid/distXpYY/align_mid_NNNNN_<MSDID>.mid
    <root>/ytd_audio/distXpYY/ytd_audio_NNNNN_<MSDID>.mp3

License note: A2MD's audio is internet-sourced copyrighted music distributed
without a data license — personal/non-commercial training only, do not redistribute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from drumml.data.base import DatasetAdapter, Track
from drumml.data.egmd import annotation_from_midi


class A2MDAdapter(DatasetAdapter):
    name = "a2md"

    def __init__(
        self,
        root: str | Path,
        *,
        max_dist: float = 0.10,
        min_onsets: int = 8,
    ):
        """``root`` is the extracted ``a2md_public`` dir (holds ``align_mid``/``ytd_audio``).

        ``max_dist`` keeps only alignment-distance buckets ``<= max_dist`` (e.g.
        ``0.10`` keeps ``dist0p00`` + ``dist0p10`` — the tightest, lowest-label-noise
        tracks). ``min_onsets`` drops tracks whose parsed drum part is (near-)empty,
        the A2MD analogue of the silent-track filter in ``prep_accompaniment``.
        """
        self.root = Path(root)
        self.max_dist = float(max_dist)
        self.min_onsets = int(min_onsets)
        if not (self.root / "align_mid").is_dir():
            raise FileNotFoundError(
                f"no align_mid/ under {self.root} (point --root at the extracted a2md_public dir)"
            )

    def _buckets(self) -> list[str]:
        out = []
        for d in sorted((self.root / "align_mid").glob("dist*")):
            # "dist0p10" -> 0.10
            val = float(d.name[len("dist"):].replace("p", "."))
            if val <= self.max_dist + 1e-9:
                out.append(d.name)
        return out

    def tracks(self) -> Iterator[Track]:
        for bucket in self._buckets():
            mid_dir = self.root / "align_mid" / bucket
            aud_dir = self.root / "ytd_audio" / bucket
            for mid in sorted(mid_dir.glob("*.mid")):
                # align_mid_NNNNN_<id>.mid  <->  ytd_audio_NNNNN_<id>.mp3
                suffix = mid.stem[len("align_mid_"):]
                mp3 = aud_dir / f"ytd_audio_{suffix}.mp3"
                if not mp3.exists():
                    continue
                track_id = f"{bucket}_{suffix}"
                ann = annotation_from_midi(mid, track_id)
                if len(ann.events) < self.min_onsets:
                    continue  # empty/failed drum part
                yield Track(track_id=track_id, annotation=ann, audio_path=mp3)
