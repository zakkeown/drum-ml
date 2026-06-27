"""drumml — automatic drum transcription toolkit.

Step 1 (this scaffold): a verifiable core for *measuring* ADT systems before
building one — a canonical drum taxonomy, dataset adapters that normalize every
source onto it, and a mir_eval-based onset-F harness with cross-dataset (OOD)
aggregation. See ADT_PIPELINE_2026.md for the design rationale.
"""

from drumml.taxonomy import Canonical, SCHEMES, reduce, scheme_classes
from drumml.events import DrumEvent, DrumAnnotation
from drumml import eval as evaluation

__all__ = [
    "Canonical",
    "SCHEMES",
    "reduce",
    "scheme_classes",
    "DrumEvent",
    "DrumAnnotation",
    "evaluation",
]
