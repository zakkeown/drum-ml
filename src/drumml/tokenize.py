"""MT3-style absolute-time drum event tokenizer.

Pure Python (no torch / numpy). A :class:`DrumTokenizer` turns a
:class:`drumml.events.DrumAnnotation` into a flat integer token sequence for a
fixed-length time segment, and back again. Times are quantized to a frame grid
and class names come from the reduced taxonomy (:func:`drumml.taxonomy.reduce` /
:func:`drumml.taxonomy.scheme_classes`), so the round-trip is lossless *at the
scheme level*: ``decode(encode(ann)).onsets_by_class(scheme)`` reproduces
``ann.onsets_by_class(scheme)`` for onsets that lie on the grid.

Vocabulary layout (contiguous, no gaps)::

    [0]                          pad
    [1]                          bos
    [2]                          eos
    [3            .. T)          TIME  tokens  (T = round(seg*hz) + 1)
    [T            .. T+C)        CLASS tokens  (C = len(scheme_classes))
    [T+C          .. T+C+V)      VELOCITY tokens (V = velocity_bins if enabled)

A segment is encoded as ``[BOS, (TIME, CLASS[, VELOCITY])*, EOS]`` with events
sorted by ``(time, class order)``. One TIME token is emitted per event (repeats
allowed) which keeps decoding a simple linear scan.
"""

from __future__ import annotations

from drumml.events import DrumAnnotation, DrumEvent
from drumml.taxonomy import Canonical, reduce, scheme_classes


class DrumTokenizer:
    """Absolute-time event tokenizer for a single fixed-length drum segment."""

    # Special token ids are fixed by the integration contract.
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def __init__(
        self,
        scheme: str = "5",
        segment_seconds: float = 2.048,
        frame_hz: float = 100.0,
        with_velocity: bool = False,
        velocity_bins: int = 4,
    ) -> None:
        if velocity_bins < 1:
            raise ValueError(f"velocity_bins must be >= 1, got {velocity_bins}")

        self.scheme = scheme
        self.segment_seconds = float(segment_seconds)
        self.frame_hz = float(frame_hz)
        self.with_velocity = bool(with_velocity)
        self.velocity_bins = int(velocity_bins)

        # Ordered reduced-class names for this scheme (the CLASS sub-vocabulary).
        self.classes: list[str] = scheme_classes(scheme)
        self._class_to_idx: dict[str, int] = {n: i for i, n in enumerate(self.classes)}

        # Map each reduced-class name back to a representative canonical class:
        # the first canonical (in kit order) that reduces to that name. This is
        # the inverse used by decode; reduce() is many-to-one so the mapping is
        # only a representative, but onsets_by_class() keys on the reduced name
        # so the round-trip is exact at the scheme level.
        self._name_to_canonical: dict[str, Canonical] = {}
        for c in Canonical:
            name = reduce(c, scheme)
            if name is not None and name not in self._name_to_canonical:
                self._name_to_canonical[name] = c

        # Token id ranges.
        self.n_time_tokens: int = round(self.segment_seconds * self.frame_hz) + 1
        self._time_offset: int = 3  # after pad/bos/eos
        self._class_offset: int = self._time_offset + self.n_time_tokens
        self._velocity_offset: int = self._class_offset + len(self.classes)
        self._n_velocity: int = self.velocity_bins if self.with_velocity else 0

    # --- vocabulary ---------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._velocity_offset + self._n_velocity

    # --- token range predicates --------------------------------------------
    def _is_time(self, tok: int) -> bool:
        return self._time_offset <= tok < self._class_offset

    def _is_class(self, tok: int) -> bool:
        return self._class_offset <= tok < self._velocity_offset

    def _is_velocity(self, tok: int) -> bool:
        return self.with_velocity and self._velocity_offset <= tok < self.vocab_size

    # --- velocity <-> bin ---------------------------------------------------
    def _vel_to_bin(self, velocity: int | None) -> int:
        """Quantize a MIDI velocity (1..127) into a bin index.

        Unknown velocity (``None``) maps to the loudest bin.
        """
        if velocity is None:
            return self.velocity_bins - 1
        v = max(0, min(127, int(velocity)))
        b = (v * self.velocity_bins) // 128
        return max(0, min(self.velocity_bins - 1, b))

    def _bin_to_vel(self, b: int) -> int:
        """Representative MIDI velocity at the center of a bin (1..127)."""
        lo = (b * 128) // self.velocity_bins
        hi = ((b + 1) * 128) // self.velocity_bins - 1
        return max(1, min(127, (lo + hi) // 2))

    # --- encode / decode ----------------------------------------------------
    def encode(self, ann: DrumAnnotation, segment_start: float = 0.0) -> list[int]:
        """Tokens for events in ``[segment_start, segment_start + segment_seconds)``.

        Events outside the window, and events whose canonical class is not scored
        by ``scheme``, are excluded.
        """
        end = segment_start + self.segment_seconds
        # (time, frame, class_idx, velocity) sorted by (time, class order).
        items: list[tuple[float, int, int, int | None]] = []
        for ev in ann.events:
            if not (segment_start <= ev.time < end):
                continue
            name = reduce(ev.canonical, self.scheme)
            if name is None:
                continue
            frame = round((ev.time - segment_start) * self.frame_hz)
            frame = max(0, min(self.n_time_tokens - 1, frame))
            items.append((ev.time, frame, self._class_to_idx[name], ev.velocity))

        items.sort(key=lambda x: (x[0], x[2]))

        tokens = [self.bos_id]
        for _time, frame, cls, vel in items:
            tokens.append(self._time_offset + frame)
            tokens.append(self._class_offset + cls)
            if self.with_velocity:
                tokens.append(self._velocity_offset + self._vel_to_bin(vel))
        tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: list[int], segment_start: float = 0.0) -> DrumAnnotation:
        """Inverse of :meth:`encode`.

        Each TIME token sets the current onset time; each following CLASS token
        emits an event (mapped to the representative canonical for that reduced
        name); a VELOCITY token attaches to the most recent event. Scanning stops
        at the first EOS; pad/bos are ignored.
        """
        events: list[DrumEvent] = []
        cur_time: float | None = None
        for tok in tokens:
            if tok == self.eos_id:
                break
            if tok == self.pad_id or tok == self.bos_id:
                continue
            if self._is_time(tok):
                frame = tok - self._time_offset
                cur_time = segment_start + frame / self.frame_hz
            elif self._is_class(tok):
                if cur_time is None:
                    continue  # malformed: class with no preceding time
                name = self.classes[tok - self._class_offset]
                events.append(DrumEvent(cur_time, self._name_to_canonical[name]))
            elif self._is_velocity(tok):
                if events:
                    last = events[-1]
                    events[-1] = DrumEvent(
                        last.time, last.canonical, self._bin_to_vel(tok - self._velocity_offset)
                    )
            # anything else (out of range) is ignored
        return DrumAnnotation(track_id="decoded", events=events)
