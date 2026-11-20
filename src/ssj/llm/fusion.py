"""From phoneme posteriors to a sentence: candidate generation and evidence fusion.

The phoneme recogniser's output is ambiguous by nature, so the sentence is found
by pooling candidates from several proposers and scoring all of them the same
way:

proposers
    * the lexicon-constrained beam search (n-gram fused N-best);
    * the best paths of the N-best's confusion network (sentences the beam lost);
    * the LLM choosing a path through that confusion network under a grammar;
    * the LLM writing a sentence freely from the detected phonemes and guesses.

scoring (identical for every candidate, whoever proposed it)
    total = w_am * log P_ctc(pronunciation | sensor)      exact CTC forward score
          + w_ngram * log P_ngram(words)
          + w_llm * log P_llm(sentence | context)
          + w_words * number of words

The division of labour is deliberate. The LLM is free to propose anything, but
a proposal only wins if its pronunciation is actually supported by the sensor
evidence: hallucinated sentences score terribly under the CTC term and are
discarded. The LLM contributes what it is good at - context, syntax, world
knowledge - without ever being trusted on what was articulated.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..decode.beam import BeamConfig, Hypothesis, beam_search
from ..decode.ctc_score import greedy_decode, sequence_score, tie_classes
from ..decode.lattice import Slot, best_paths, confusion_network
from ..decode.ngram import NGramLM
from ..decode.trie import LexiconTrie
from ..phonology import Inventory, Lexicon, normalize_text, render_sentence
from .backend import LLMBackend, NullBackend
from .grammar import gbnf_from_slots
from .prompts import lattice_prompt, proposal_prompt, scoring_context


@dataclass(frozen=True)
class FusionConfig:
    tau: float = 0.5  # trust in within-class phoneme detail (see decode.ctc_score.tie_classes)
    w_am: float = 1.0
    w_ngram: float = 0.4
    w_llm: float = 0.6
    w_words: float = 1.0
    rescore_top: int = 12  # beam hypotheses passed to the LLM
    lattice_paths: int = 6  # extra candidates from the confusion network
    lattice_proposal: bool = True  # grammar-constrained LLM proposal
    free_proposal: bool = True  # unconstrained LLM proposal
    max_alternatives: int = 6
    confidence_scale: float = 0.5  # flattens the candidate posterior
    min_confidence: float = 0.0  # results below this are flagged as uncertain
    min_frame_score: float = -4.0  # average per-frame acoustic score below this is noise


@dataclass
class Candidate:
    words: tuple[str, ...]
    source: str  # beam | lattice | llm-lattice | llm-free
    am: float = 0.0
    ngram: float = 0.0
    llm: float = 0.0
    score: float = 0.0

    @property
    def text(self) -> str:
        return render_sentence(self.words)

    def features(self) -> tuple[float, float, float, float]:
        return (self.am, self.ngram, self.llm, float(len(self.words)))


@dataclass
class DecodeResult:
    words: tuple[str, ...] = ()
    text: str = ""
    confidence: float = 0.0
    accepted: bool = False
    candidates: list[Candidate] = field(default_factory=list)
    phonemes: list[str] = field(default_factory=list)  # greedy phoneme string, for display
    slots: list[Slot] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    first_pass: tuple[str, ...] = ()  # the beam search's own best guess

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


def fused_score(candidate: Candidate, config: FusionConfig) -> float:
    return (
        config.w_am * candidate.am
        + config.w_ngram * candidate.ngram
        + config.w_llm * candidate.llm
        + config.w_words * len(candidate.words)
    )


class SentenceDecoder:
    """Phoneme posteriors -> sentence, combining beam search, n-gram and LLM."""

    def __init__(
        self,
        inventory: Inventory,
        lexicon: Lexicon,
        lm: NGramLM,
        *,
        backend: LLMBackend | None = None,
        beam: BeamConfig | None = None,
        fusion: FusionConfig | None = None,
    ):
        self.inventory = inventory
        self.lexicon = lexicon
        self.lm = lm
        self.backend = backend or NullBackend()
        self.beam = beam or BeamConfig()
        self.fusion = fusion or FusionConfig()
        self.trie = LexiconTrie.build(lexicon, inventory, lm)
        self._pron_cache: dict[str, list[list[int]]] = {}

    @property
    def has_llm(self) -> bool:
        return not isinstance(self.backend, NullBackend)

    # ------------------------------------------------------------ candidates
    def _pronunciations(self, word: str) -> list[list[int]]:
        cached = self._pron_cache.get(word)
        if cached is None:
            cached = [self.inventory.encode(p) for p in self.lexicon.prons(word) if all(s in self.inventory for s in p)]
            self._pron_cache[word] = cached
        return cached

    def _known(self, words: Sequence[str]) -> bool:
        return bool(words) and all(w in self.lexicon and w in self.lm and self._pronunciations(w) for w in words)

    def _acoustic(self, scores: np.ndarray, words: Sequence[str]) -> float:
        value, _ = sequence_score(scores, [self._pronunciations(w) for w in words])
        return value

    def propose(
        self,
        scores: np.ndarray,
        nbest: Sequence[Hypothesis],
        slots: Sequence[Slot],
        log_probs: np.ndarray,
        history: Sequence[str],
        hint: str,
    ) -> dict[tuple[str, ...], str]:
        """Collect candidate word sequences and remember who proposed each."""
        cfg = self.fusion
        pool: dict[tuple[str, ...], str] = {}
        for hypothesis in nbest[: cfg.rescore_top]:
            if hypothesis.words:
                pool.setdefault(hypothesis.words, "beam")
        if cfg.lattice_paths > 0 and slots:
            added = 0
            for words, _ in best_paths(slots, self.lm, k=cfg.lattice_paths + len(pool), lm_weight=self.beam.lm_weight):
                if words and words not in pool and self._known(words):
                    pool[words] = "lattice"
                    added += 1
                    if added >= cfg.lattice_paths:
                        break
        if not self.has_llm:
            return pool
        budget = max(len(slots) * 4, 16)
        if cfg.lattice_proposal and slots:
            system, user = lattice_prompt(slots, history, hint, cfg.max_alternatives)
            reply = self.backend.generate(
                system, user, max_tokens=budget, grammar=gbnf_from_slots(slots, cfg.max_alternatives)
            )
            words = tuple(normalize_text(reply))
            if self._known(words):
                pool.setdefault(words, "llm-lattice")
        if cfg.free_proposal:
            guesses = [render_sentence(h.words) for h in nbest[:5] if h.words]
            system, user = proposal_prompt(self.inventory.decode(greedy_decode(log_probs)), guesses, history, hint)
            reply = self.backend.generate(system, user, max_tokens=budget)
            words = tuple(normalize_text(reply.splitlines()[0] if reply.strip() else ""))
            if self._known(words):
                pool.setdefault(words, "llm-free")
        return pool

    # ---------------------------------------------------------------- decode
    def decode(self, log_probs: np.ndarray, history: Sequence[str] = (), hint: str = "") -> DecodeResult:
        """Decode one utterance's phoneme log-posteriors ``[T, V]``."""
        cfg = self.fusion
        timings: dict[str, float] = {}
        result = DecodeResult(phonemes=self.inventory.decode(greedy_decode(log_probs)), timings=timings)
        if len(log_probs) == 0:
            return result

        started = time.perf_counter()
        scores = tie_classes(log_probs, self.inventory, cfg.tau)
        nbest = beam_search(scores, self.trie, self.lm, self.beam)
        timings["beam"] = time.perf_counter() - started
        if not nbest or not nbest[0].words:
            return result  # the best explanation is "no speech"
        result.first_pass = nbest[0].words
        result.slots = confusion_network(nbest, max_alternatives=cfg.max_alternatives)

        started = time.perf_counter()
        pool = self.propose(scores, nbest, result.slots, log_probs, history, hint)
        timings["propose"] = time.perf_counter() - started

        started = time.perf_counter()
        candidates = [Candidate(words, source) for words, source in pool.items()]
        for candidate in candidates:
            candidate.am = self._acoustic(scores, candidate.words)
            candidate.ngram = self.lm.score(candidate.words)
        if self.has_llm and cfg.w_llm != 0.0:
            context = scoring_context(history, hint)
            llm_scores = self.backend.score(context, [c.text + "\n" for c in candidates])
            for candidate, value in zip(candidates, llm_scores):
                candidate.llm = value
        timings["score"] = time.perf_counter() - started
        return self.select(result, candidates, len(log_probs))

    def select(self, result: DecodeResult, candidates: list[Candidate], n_frames: int) -> DecodeResult:
        """Rank candidates by fused score and attach a confidence."""
        cfg = self.fusion
        weights = cfg if self.has_llm else _without_llm(cfg, self.beam)
        for candidate in candidates:
            candidate.score = fused_score(candidate, weights)
        candidates.sort(key=lambda c: c.score, reverse=True)
        result.candidates = candidates
        if not candidates:
            return result
        best = candidates[0]
        peak = best.score
        mass = sum(math.exp(cfg.confidence_scale * (c.score - peak)) for c in candidates)
        result.words = best.words
        result.text = best.text
        result.confidence = 1.0 / mass
        result.accepted = result.confidence >= cfg.min_confidence and best.am / max(n_frames, 1) >= cfg.min_frame_score
        return result


def _without_llm(config: FusionConfig, beam: BeamConfig) -> FusionConfig:
    """Fusion weights when no LLM is configured.

    The language side then consists of the n-gram alone, so the weights the
    first pass was tuned with are the right ones; the final ranking can differ
    from the beam's only through the exact CTC scores and the extra lattice paths.
    """
    from dataclasses import replace

    return replace(config, w_ngram=beam.lm_weight, w_llm=0.0, w_words=beam.word_bonus)
