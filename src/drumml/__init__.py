"""drumml — automatic drum transcription toolkit.

Step 1 (this scaffold): a verifiable core for *measuring* ADT systems before
building one — a canonical drum taxonomy, dataset adapters that normalize every
source onto it, and a mir_eval-based onset-F harness with cross-dataset (OOD)
aggregation. See ADT_PIPELINE_2026.md for the design rationale.
"""

from drumml.taxonomy import Canonical, SCHEMES, reduce, scheme_classes
from drumml.events import DrumEvent, DrumAnnotation
from drumml.tokenize import DrumTokenizer
from drumml import eval as evaluation

# NOTE: the torch-dependent stack (drumml.features, drumml.model,
# drumml.data.torch_dataset, drumml.train) is intentionally NOT imported here, so
# `import drumml` stays torch-free for eval-only users. Import those submodules
# explicitly after `pip install .[model]`.

__all__ = [
    "Canonical",
    "SCHEMES",
    "reduce",
    "scheme_classes",
    "DrumEvent",
    "DrumAnnotation",
    "DrumTokenizer",
    "evaluation",
]
