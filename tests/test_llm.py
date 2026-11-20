from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pytest

from ssj.decode.lattice import Slot
from ssj.decode.ngram import train_ngram
from ssj.llm import FusionConfig, LLMBackend, LLMConfig, NullBackend, SentenceDecoder, gbnf_from_slots, load_backend
from ssj.llm.fusion import Candidate, fused_score
from ssj.llm.grammar import n_paths
from ssj.llm.prompts import lattice_prompt, proposal_prompt, scoring_context
from ssj.phonology import Lexicon, normalize_text

CORPUS = [
    s.split()
    for s in [
        "i met the cat", "i bet the dog sat", "pet the cat", "the cat sat on the mat", "the dog ate the bone",
        "i met a pet", "the pet sat on the mat", "i bet the dog ate the bone", "the cat ate the fish",
    ]
]  # fmt: skip


def peaky(size: int, groups: list[list[int]], confidence: float = 0.9) -> np.ndarray:
    """CTC-like posteriors; each step spreads its mass evenly over a group of tokens."""
    rows = []
    for group in groups:
        blank = np.full(size, (1 - confidence) / (size - 1))
        blank[0] = confidence
        rows.append(blank)
        for _ in range(4):
            row = np.full(size, (1 - confidence) / (size - len(group)))
            row[group] = confidence / len(group)
            rows.append(row)
    final = np.full(size, (1 - confidence) / (size - 1))
    final[0] = confidence
    return np.log(np.asarray([*rows, final]))


class ScriptedLLM(LLMBackend):
    """A stand-in LLM with fixed opinions, to test how its output is used."""

    name = "scripted"

    def __init__(self, favourite: str = "", replies: Sequence[str] = ()):
        self.favourite = favourite
        self.replies = list(replies)
        self.scored: list[tuple[str, list[str]]] = []
        self.prompts: list[tuple[str, str, str | None]] = []

    def score(self, context: str, candidates: Sequence[str]) -> list[float]:
        self.scored.append((context, list(candidates)))
        return [0.0 if self.favourite and self.favourite in c.lower() else -30.0 for c in candidates]

    def generate(self, system: str, user: str, *, max_tokens: int = 48, grammar: str | None = None) -> str:
        self.prompts.append((system, user, grammar))
        return self.replies.pop(0) if self.replies else ""


@pytest.fixture(scope="module")
def resources(inventory):
    core = Lexicon.core(inventory)
    lexicon = Lexicon()
    for word in sorted({w for s in CORPUS for w in s} | {"bed", "red", "fish"}):
        for pron in core.prons(word):
            lexicon.add(word, pron)
    return lexicon, train_ngram(CORPUS, order=3, vocab=lexicon.words)


def ambiguous_scores(inventory, lexicon, sentence: str) -> np.ndarray:
    """Posteriors for a sentence in which P, B and M cannot be told apart."""
    bilabial = [inventory.token_id(s) for s in "PBM"]
    tokens = inventory.encode(lexicon.phonemize(sentence.split()))
    return peaky(inventory.size, [bilabial if t in bilabial else [t] for t in tokens])


# ------------------------------------------------------------------- grammar
def test_grammar_from_slots():
    slots = [
        Slot({"pat": 0.5, "bat": 0.3, "mat": 0.2}),
        Slot({"the": 0.7, "a": 0.3}),
        Slot({"": 0.6, "now": 0.4}),
        Slot({'say "hi"': 1.0}),
    ]
    grammar = gbnf_from_slots(slots)
    lines = grammar.strip().splitlines()
    assert lines[0] == "root ::= s0 s1 s2 s3"
    assert lines[1] == 's0 ::= " pat" | " bat" | " mat"'  # most likely first
    assert lines[3] == 's2 ::= (" now")?'  # optional position
    assert lines[4] == r's3 ::= " say \"hi\""'  # quotes are escaped
    assert n_paths(slots) == 3 * 2 * 2 * 1
    assert gbnf_from_slots(slots, max_alternatives=1).splitlines()[1] == 's0 ::= " pat"'
    assert gbnf_from_slots([]) == 'root ::= ""'


def test_prompts():
    context = scoring_context(["I am thirsty.", "Hello."], hint="at a cafe")
    assert context.endswith("I am thirsty.\nHello.\n") and "Situation: at a cafe\n" in context
    assert scoring_context() == scoring_context([], "")
    assert len(scoring_context([str(i) for i in range(10)]).splitlines()) == 5  # history is bounded
    slots = [Slot({"met": 0.5, "bet": 0.5}), Slot({"": 0.5, "the": 0.5})]
    system, user = lattice_prompt(slots, ["Hello."], "at home")
    assert "jaw" in system and "1. met / bet" in user and "2. [the]" in user and "Previously said: Hello." in user
    _, user = proposal_prompt(["HH", "AH", "L", "OW"], ["Hello.", "Yellow."])
    assert "HH AH L OW" in user and "- Hello." in user


# -------------------------------------------------------------------- fusion
def test_without_llm_the_ngram_decides(resources, inventory):
    lexicon, lm = resources
    decoder = SentenceDecoder(inventory, lexicon, lm)
    assert not decoder.has_llm and isinstance(load_backend(LLMConfig()), NullBackend)
    result = decoder.decode(ambiguous_scores(inventory, lexicon, "i met the cat"))
    assert result.words == ("i", "met", "the", "cat") and result.text == "I met the cat."
    assert result.accepted and 0 < result.confidence <= 1
    assert result.first_pass == result.words
    assert result.phonemes[0] == "AY" and len(result.slots) >= 4
    assert {"beam"} <= {c.source for c in result.candidates}
    assert result.candidates == sorted(result.candidates, key=lambda c: c.score, reverse=True)
    assert set(result.timings) == {"beam", "propose", "score"}


def test_llm_overrules_the_ngram_among_sensor_equivalent_words(resources, inventory):
    """The n-gram prefers 'met'; an LLM that knows better picks 'bet' - both fit the sensor equally."""
    lexicon, lm = resources
    scores = ambiguous_scores(inventory, lexicon, "i met the cat")
    llm = ScriptedLLM(favourite="bet")
    result = SentenceDecoder(inventory, lexicon, lm, backend=llm).decode(scores, history=["Hello."], hint="betting")
    assert result.words == ("i", "bet", "the", "cat")
    assert result.first_pass == ("i", "met", "the", "cat")
    context, candidates = llm.scored[0]
    assert "Hello." in context and "Situation: betting" in context
    assert all(c.endswith(".\n") and c[0].isupper() for c in candidates)  # scored as rendered sentences
    met = next(c for c in result.candidates if c.words == ("i", "met", "the", "cat"))
    assert abs(met.am - result.best.am) < 0.5  # the sensor evidence cannot separate them


def test_llm_proposals_must_be_supported_by_the_sensor(resources, inventory):
    lexicon, lm = resources
    scores = ambiguous_scores(inventory, lexicon, "i met the cat")
    # A fluent hallucination, a sentence with an unknown word, and a lattice answer.
    llm = ScriptedLLM(replies=[" i bet the cat", "The dog ate the bone."])
    decoder = SentenceDecoder(inventory, lexicon, lm, backend=llm, fusion=FusionConfig(w_llm=0.0))
    result = decoder.decode(scores)
    sources = {c.words: c.source for c in result.candidates}
    assert ("the", "dog", "ate", "the", "bone") in sources and sources[
        ("the", "dog", "ate", "the", "bone")
    ] == "llm-free"
    hallucination = next(c for c in result.candidates if c.source == "llm-free")
    assert hallucination.am < result.best.am - 30  # it does not match what was articulated
    assert result.words == ("i", "met", "the", "cat")
    _, _, grammar = llm.prompts[0]
    assert grammar.startswith("root ::=") and '" met"' in grammar  # the first proposal was lattice-constrained
    assert llm.prompts[1][2] is None

    llm = ScriptedLLM(replies=["", "I met the xylophone."])
    result = SentenceDecoder(inventory, lexicon, lm, backend=llm).decode(scores)
    assert all("xylophone" not in c.words for c in result.candidates)  # unpronounceable proposals are dropped


def test_silence_and_noise_are_rejected(resources, inventory):
    lexicon, lm = resources
    decoder = SentenceDecoder(inventory, lexicon, lm)
    silence = np.log(np.r_[0.999, np.full(inventory.size - 1, 0.001 / (inventory.size - 1))])[None].repeat(40, axis=0)
    result = decoder.decode(silence)
    assert result.words == () and result.text == "" and not result.accepted
    assert decoder.decode(np.zeros((0, inventory.size))).words == ()
    uniform = np.full((60, inventory.size), -np.log(inventory.size))
    assert not decoder.decode(uniform).accepted or decoder.decode(uniform).confidence < 0.9


def test_fused_score_weights():
    candidate = Candidate(("a", "b"), "beam", am=-10.0, ngram=-4.0, llm=-6.0)
    assert fused_score(candidate, FusionConfig(w_am=1.0, w_ngram=0.5, w_llm=2.0, w_words=3.0)) == pytest.approx(
        -10 - 2 - 12 + 6
    )
    assert candidate.text == "A b." and candidate.features() == (-10.0, -4.0, -6.0, 2.0)


# ------------------------------------------------------------- real GGUF model
needs_model = pytest.mark.skipif(not os.environ.get("SSJ_TEST_GGUF"), reason="set SSJ_TEST_GGUF to a GGUF file to run")


@pytest.fixture(scope="module")
def gguf():
    from ssj.llm.backend import LlamaCppBackend

    backend = LlamaCppBackend(LLMConfig(model=os.environ["SSJ_TEST_GGUF"], n_ctx=1024))
    yield backend
    backend.close()


@pytest.mark.llm
@needs_model
def test_real_llm_ranks_sentences_sensibly(gguf):
    context = scoring_context(["I am thirsty."])
    candidates = ["Can I have a glass of water?\n", "Can I have a class of water?\n", "Glass water a of have I can?\n"]
    scores = gguf.score(context, candidates)
    assert scores[0] > scores[1] > scores[2]
    again = gguf.score(context, candidates[::-1])[::-1]
    assert np.allclose(scores, again, atol=0.5)  # independent of evaluation order and cache state
    joint = gguf.score(context, ["Hello there.\nHow are you?\n"])[0]
    chained = gguf.score(context, ["Hello there.\n"])[0] + gguf.score(context + "Hello there.\n", ["How are you?\n"])[0]
    assert joint == pytest.approx(chained, abs=0.5)  # scores obey the chain rule


@pytest.mark.llm
@needs_model
def test_real_llm_respects_the_lattice_grammar(gguf):
    slots = [
        Slot({"pat": 0.4, "bat": 0.35, "mat": 0.25}),
        Slot({"the": 0.7, "a": 0.3}),
        Slot({"dog": 0.5, "talk": 0.3, "dock": 0.2}),
    ]
    system, user = lattice_prompt(slots)
    words = normalize_text(gguf.generate(system, user, max_tokens=16, grammar=gbnf_from_slots(slots)))
    assert len(words) == 3 and all(word in slot.words for word, slot in zip(words, slots))
    assert gguf.score(scoring_context(), ["Pat the dog.\n"])[0] == pytest.approx(
        gguf.score(scoring_context(), ["Pat the dog.\n"])[0], abs=0.5
    )
