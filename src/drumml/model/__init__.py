"""Neural transcription models for drumml.

Currently exposes the MT3/T5-style encoder-decoder transcriber.
"""

from drumml.model.seq2seq import Seq2SeqADT, Seq2SeqConfig

__all__ = ["Seq2SeqADT", "Seq2SeqConfig"]
