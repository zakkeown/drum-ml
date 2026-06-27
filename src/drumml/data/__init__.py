"""Dataset adapters: normalize each corpus onto ``DrumAnnotation``.

Each adapter yields ``Track`` objects (id, optional audio path, annotation in the
canonical taxonomy). The evaluator and trainers depend only on this interface,
so the inconsistent formats of E-GMD / MDB / ADTOF / ENST stay quarantined here.
"""

from drumml.data.base import Track, DatasetAdapter

__all__ = ["Track", "DatasetAdapter"]
