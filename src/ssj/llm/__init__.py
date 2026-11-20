"""Local LLM stage: backends, lattice grammars and evidence fusion."""

from .backend import LLMBackend, LLMConfig, LLMError, NullBackend, load_backend
from .fusion import Candidate, DecodeResult, FusionConfig, SentenceDecoder
from .grammar import gbnf_from_slots

__all__ = [
    "Candidate",
    "DecodeResult",
    "FusionConfig",
    "LLMBackend",
    "LLMConfig",
    "LLMError",
    "NullBackend",
    "SentenceDecoder",
    "gbnf_from_slots",
    "load_backend",
]
