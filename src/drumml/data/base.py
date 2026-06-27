"""Common dataset interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from drumml.events import DrumAnnotation


@dataclass
class Track:
    """One audio item and its reference annotation."""

    track_id: str
    annotation: DrumAnnotation
    audio_path: Optional[Path] = None


class DatasetAdapter(ABC):
    """Yields :class:`Track` objects with annotations in the canonical taxonomy."""

    #: short name used in reports (e.g. "egmd", "mdb").
    name: str = "dataset"

    @abstractmethod
    def tracks(self) -> Iterator[Track]:
        """Iterate over the dataset's tracks."""
        raise NotImplementedError

    def annotations(self) -> Iterator[DrumAnnotation]:
        for t in self.tracks():
            yield t.annotation
