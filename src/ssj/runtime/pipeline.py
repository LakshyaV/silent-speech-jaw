"""The assembled recogniser: features -> phoneme posteriors -> sentence."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..decode.beam import BeamConfig
from ..decode.ngram import NGramLM, train_ngram
from ..dsp.activity import SegmenterConfig
from ..llm.backend import LLMBackend, LLMConfig, load_backend
from ..llm.fusion import DecodeResult, FusionConfig, SentenceDecoder
from ..phonology import CONTROL_WORDS, Inventory, Lexicon, normalize_text
from .bundle import Bundle

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    lexicon: str = ""  # CMUdict-format file; empty = the bundled core lexicon
    lm: str = ""  # n-gram model (.npz or .arpa[.gz]); empty = a small built-in model
    beam: BeamConfig = field(default_factory=BeamConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    threads: int = 1  # CPU threads for the recogniser
    history: int = 4  # accepted sentences kept as LLM context


def load_lexicon(path: str, inventory: Inventory) -> Lexicon:
    if not path:
        return Lexicon.core(inventory)
    lexicon = Lexicon.load(path, inventory=inventory)
    lexicon.add_control_words(inventory)
    return lexicon


def builtin_lm(lexicon: Lexicon) -> NGramLM:
    """A small bigram model over the built-in prompt sentences.

    It only keeps the system usable before a real language model has been built
    with ``ssj lm build``; every lexicon word stays reachable through the
    uniform floor of the unigram distribution.
    """
    from ..collect.prompts import builtin_sentences

    sentences = [normalize_text(s) for s in builtin_sentences()]
    return train_ngram(sentences, order=2, vocab=lexicon.words, extra_words=CONTROL_WORDS)


def load_lm(path: str, lexicon: Lexicon) -> NGramLM:
    if not path:
        log.warning("no language model configured: using the small built-in one (build one with `ssj lm build`)")
        return builtin_lm(lexicon)
    file = Path(path)
    if file.suffix == ".npz":
        return NGramLM.load(file)
    return NGramLM.from_arpa(file, vocab=lexicon.words)


class Pipeline:
    """Loads a bundle and its language resources once; decodes many utterances."""

    def __init__(self, bundle: Bundle, config: PipelineConfig | None = None, *, backend: LLMBackend | None = None):
        self.bundle = bundle
        self.config = config or PipelineConfig()
        inventory = bundle.inventory
        self.lexicon = load_lexicon(self.config.lexicon, inventory)
        self.lm = load_lm(self.config.lm, self.lexicon)
        self.backend = backend if backend is not None else load_backend(self.config.llm)
        self.decoder = SentenceDecoder(
            inventory, self.lexicon, self.lm, backend=self.backend, beam=self.config.beam, fusion=self.config.fusion
        )

    @classmethod
    def from_directory(cls, bundle_dir: str | Path, config: PipelineConfig | None = None, **kwargs) -> Pipeline:
        config = config or PipelineConfig()
        return cls(Bundle.load(bundle_dir, threads=config.threads), config, **kwargs)

    def describe(self) -> str:
        return (
            f"model: {self.bundle.describe()}\n"
            f"lexicon: {len(self.lexicon):,} words | n-gram: order {self.lm.order}, {sum(self.lm.n_grams()):,} n-grams | "
            f"LLM: {self.backend.name}"
        )

    def decode_features(self, feats: np.ndarray, history: Sequence[str] = (), hint: str = "") -> DecodeResult:
        """Decode one utterance from its front-end features ``[T, C]``."""
        started = time.perf_counter()
        log_probs = self.bundle.log_probs(feats)
        elapsed = time.perf_counter() - started
        result = self.decoder.decode(log_probs, history, hint)
        result.timings["recogniser"] = elapsed
        return result

    def close(self) -> None:
        self.backend.close()
