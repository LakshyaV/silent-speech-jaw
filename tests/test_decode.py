from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from ssj.decode.beam import BeamConfig, beam_search
from ssj.decode.ctc_score import ctc_forward, greedy_decode, log_softmax, marginalize, sequence_score, tie_classes
from ssj.decode.lattice import EPS, Slot, best_paths, confusion_network, describe, posteriors
from ssj.decode.ngram import BOS_ID, EOS_ID, UNK_ID, NGramLM, train_ngram
from ssj.decode.trie import ROOT, LexiconTrie
from ssj.eval.metrics import ErrorRate, align, character_error_rate, edit_distance, error_rate, oracle_error_rate
from ssj.phonology import Lexicon

CORPUS = [
    s.split()
    for s in [
        "the cat sat on the mat", "the dog sat on the log", "a cat and a dog", "the cat ate the fish",
        "a dog ate the bone", "the mat is on the floor", "i met the cat", "i bet the dog sat", "pet the cat",
        "i met a pet", "the pet sat on the mat", "i bet the dog ate the bone", "cat",
    ]
]  # fmt: skip


# ------------------------------------------------------------------ metrics
def test_edit_distance_and_alignment():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], [1, 2]) == 2 and edit_distance([1, 2], [1, 2]) == 0
    ops = align("the cat sat".split(), "the bat sat down".split())
    assert ops == [("=", "the", "the"), ("S", "cat", "bat"), ("=", "sat", "sat"), ("I", None, "down")]
    assert [op for op, _, _ in align(list("abc"), list("ac"))] == ["=", "D", "="]
    assert sum(op != "=" for op, _, _ in align(list("kitten"), list("sitting"))) == 3


def test_error_rates():
    rate = ErrorRate()
    rate.add("a b c d".split(), "a x c".split())
    rate.add("e f".split(), "e f".split())
    assert rate.rate == pytest.approx(2 / 6) and rate.sentence_rate == 0.5
    assert error_rate([["a", "b"]], [["a"]]) == 0.5
    assert character_error_rate([["ab", "c"]], [["ab", "d"]]) == pytest.approx(1 / 4)
    assert oracle_error_rate([["a", "b"]], [[["x", "b"], ["a", "b"], ["a"]]]) == 0.0


# -------------------------------------------------------------------- n-gram
@pytest.mark.parametrize("order,min_count", [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)])
def test_ngram_is_a_proper_distribution(order, min_count):
    lm = train_ngram(CORPUS, order=order, min_count=min_count, extra_words=["zebra"])
    the, cat = lm.word_id("the"), lm.word_id("cat")
    histories = [
        (BOS_ID,),
        (the,),
        (the, cat),
        (BOS_ID, the),
        (lm.word_id("fish"), lm.word_id("bone")),
        (UNK_ID, UNK_ID),
    ]
    for history in histories:
        total = sum(math.exp(lm.logp(w, history)) for w in range(len(lm.vocab)) if w != BOS_ID)
        assert total == pytest.approx(1.0, abs=1e-5), (order, history)
    assert "zebra" in lm and math.isfinite(lm.logp(lm.word_id("zebra"), (the,)))  # unseen words keep some mass
    assert lm.word_id("giraffe") == UNK_ID


def test_control_gestures_are_transparent_to_the_language_model():
    lm = train_ngram(CORPUS, order=3, extra_words=["<clench>"])
    the, clench = lm.word_id("the"), lm.word_id("<clench>")
    assert lm.logp(clench, (the,)) == 0.0 and lm.next_state((the,), clench) == (the,)
    assert lm.score(["the", "cat", "<clench>"]) == pytest.approx(lm.score(["the", "cat"]))


def test_ngram_prefers_seen_word_order():
    lm = train_ngram(CORPUS, order=3)
    assert lm.score("the cat sat on the mat".split()) > lm.score("mat the on sat cat the".split()) + 5
    assert lm.logp(lm.word_id("sat"), (lm.word_id("the"), lm.word_id("cat"))) > lm.logp(
        lm.word_id("sat"), (lm.word_id("the"), lm.word_id("fish"))
    )
    assert lm.perplexity(CORPUS) < train_ngram(CORPUS, order=1).perplexity(CORPUS)
    assert lm.next_state((BOS_ID,), 5) == (BOS_ID, 5) and lm.next_state((1, 2), 3) == (2, 3)


def test_ngram_storage_round_trips(tmp_path):
    lm = train_ngram(CORPUS, order=3)
    lm.save(tmp_path / "lm.npz")
    lm.to_arpa(tmp_path / "lm.arpa")
    sentences = ["the dog sat on the mat".split(), "fish the".split(), "unknownword cat".split()]
    for other in (NGramLM.load(tmp_path / "lm.npz"), NGramLM.from_arpa(tmp_path / "lm.arpa")):
        assert other.n_grams() == lm.n_grams()
        for sentence in sentences:
            assert other.score(sentence) == pytest.approx(lm.score(sentence), abs=1e-4)
    restricted = NGramLM.from_arpa(tmp_path / "lm.arpa", vocab=["the", "cat", "sat"])
    assert set(restricted.vocab) == {"<s>", "</s>", "<unk>", "the", "cat", "sat"}
    with pytest.raises(ValueError):
        train_ngram([], order=2)


def test_ngram_vocabulary_restriction():
    lm = train_ngram(CORPUS, order=2, vocab=["the", "cat", "sat", "zebra"])
    assert set(lm.vocab) == {"<s>", "</s>", "<unk>", "the", "cat", "sat", "zebra"}
    assert lm.logp(lm.word_id("zebra"), (BOS_ID,)) < lm.logp(lm.word_id("the"), (BOS_ID,))
    assert math.isfinite(lm.logp(lm.word_id("zebra"), (BOS_ID,)))
    top = train_ngram(CORPUS, order=1, max_vocab=3)
    assert len(top.vocab) == 6 and "the" in top


# ----------------------------------------------------------------- CTC maths
def brute_force_ctc(scores: np.ndarray, tokens: list[int]) -> float:
    """Sum over every frame labelling that collapses to ``tokens``."""
    T, V = scores.shape
    total = -math.inf
    for path in itertools.product(range(V), repeat=T):
        collapsed = [k for k, _ in itertools.groupby(path) if k != 0]
        if collapsed == tokens:
            total = np.logaddexp(total, sum(scores[t, k] for t, k in enumerate(path)))
    return float(total)


@pytest.mark.parametrize("tokens", [[], [1], [1, 2], [1, 1], [2, 1, 2], [3, 3, 1]])
def test_ctc_forward_matches_enumeration(tokens):
    rng = np.random.default_rng(len(tokens))
    scores = log_softmax(rng.normal(size=(5, 4)))
    assert ctc_forward(scores, tokens) == pytest.approx(brute_force_ctc(scores, tokens), abs=1e-9)


def test_ctc_forward_edge_cases():
    scores = log_softmax(np.random.default_rng(0).normal(size=(3, 4)))
    assert ctc_forward(scores, [1, 2, 3, 1]) < -1e20  # more labels than frames
    assert ctc_forward(scores, [1, 1]) > -1e20 and ctc_forward(scores, [1, 1, 1]) < -1e20  # repeats need blanks
    assert ctc_forward(np.zeros((0, 4)), []) == 0.0


def test_greedy_decode():
    path = [0, 3, 3, 0, 3, 2, 2, 0, 0, 1]
    scores = np.full((len(path), 4), -10.0)
    scores[np.arange(len(path)), path] = 0.0
    assert greedy_decode(scores) == [3, 3, 2, 1]
    assert greedy_decode(np.zeros((0, 4))) == []


def test_class_tying(inventory):
    rng = np.random.default_rng(0)
    log_probs = log_softmax(rng.normal(size=(20, inventory.size)))
    classes = marginalize(log_probs, inventory)
    assert classes.shape == (20, inventory.n_classes)
    assert np.allclose(np.exp(classes).sum(axis=1), 1.0)  # marginalising keeps a distribution
    assert np.allclose(classes[:, 0], log_probs[:, 0])  # blank is its own class

    assert tie_classes(log_probs, inventory, 1.0) is log_probs
    tied = tie_classes(log_probs, inventory, 0.0)
    p, b, m, t = (inventory.token_id(s) for s in "PBMT")
    assert np.allclose(tied[:, p], tied[:, b]) and np.allclose(tied[:, p], tied[:, m])  # one class, one score
    assert not np.allclose(tied[:, p], tied[:, t])
    assert np.all(tied[:, 1:] >= log_probs[:, 1:] - 1e-9)  # a class is at least as likely as its member
    assert np.array_equal(tied[:, 0], log_probs[:, 0])
    half = tie_classes(log_probs, inventory, 0.5)
    assert np.allclose(half, 0.5 * (tied + log_probs))


def test_sequence_score_chooses_the_supported_pronunciation(inventory):
    read_present = inventory.encode(["R", "IY", "D"])
    read_past = inventory.encode(["R", "EH", "D"])
    scores = peaky(inventory.size, read_present)
    value, choice = sequence_score(scores, [[read_past, read_present]])
    assert choice == [1] and value > ctc_forward(scores, read_past) + 5


# ---------------------------------------------------------------- beam search
def peaky(
    size: int,
    tokens: list[int],
    frames_per_token: int = 4,
    confidence: float = 0.9,
    alternatives: dict[int, list[int]] | None = None,
) -> np.ndarray:
    """Posteriors as a CTC model would emit them: a few frames per token, blank in between."""
    rows = []
    for token in tokens:
        group = (alternatives or {}).get(token, [token])
        for kind in ["blank"] + ["token"] * frames_per_token:
            row = np.full(size, (1 - confidence) / (size - len(group)))
            if kind == "blank":
                row[:] = (1 - confidence) / (size - 1)
                row[0] = confidence
            else:
                row[group] = confidence / len(group)
            rows.append(row)
    rows.append(np.r_[confidence, np.full(size - 1, (1 - confidence) / (size - 1))])
    return np.log(np.asarray(rows))


@pytest.fixture(scope="module")
def decoding(inventory):
    lexicon = Lexicon()
    words = sorted({w for s in CORPUS for w in s} | {"met", "bet", "pet", "bed", "red", "read"})
    core = Lexicon.core(inventory)
    for word in words:
        for pron in core.prons(word):
            lexicon.add(word, pron)
    lm = train_ngram(CORPUS, order=3, vocab=lexicon.words)
    return lexicon, lm, LexiconTrie.build(lexicon, inventory, lm)


def test_trie_structure(decoding, inventory):
    lexicon, lm, trie = decoding
    assert trie.n_entries == lexicon.n_pronunciations()
    node = ROOT
    for token in inventory.encode(["K", "AE", "T"]):
        node = trie.children[node][token]
    assert [lm.vocab[w] for w in trie.words[node]] == ["cat"]
    assert trie.lookahead[ROOT] == 0.0
    assert all(trie.lookahead[child] <= 0 for child in trie.children[ROOT].values())
    read = trie.children[trie.children[trie.children[ROOT][inventory.token_id("R")]][inventory.token_id("EH")]][
        inventory.token_id("D")
    ]
    assert {lm.vocab[w] for w in trie.words[read]} == {"red", "read"}  # homophones share a node


def test_beam_search_recovers_a_clean_sentence(decoding, inventory):
    lexicon, lm, trie = decoding
    words = "the cat sat on the mat".split()
    scores = peaky(inventory.size, inventory.encode(lexicon.phonemize(words)))
    nbest = beam_search(scores, trie, lm, BeamConfig())
    assert nbest[0].words == tuple(words)
    best = nbest[0]
    assert best.score == pytest.approx(
        best.am + BeamConfig().lm_weight * best.lm + BeamConfig().word_bonus * len(words)
    )
    assert best.lm == pytest.approx(lm.score(words))
    assert best.am == pytest.approx(ctc_forward(scores, inventory.encode(lexicon.phonemize(words))), abs=0.5)
    assert [h.score for h in nbest] == sorted((h.score for h in nbest), reverse=True)
    assert len({h.words for h in nbest}) == len(nbest)  # each word sequence appears once


def test_language_model_resolves_homophenes(decoding, inventory):
    """met / bet / pet are one articulation; only context can tell them apart."""
    lexicon, lm, trie = decoding
    bilabial = [inventory.token_id(s) for s in "PBM"]
    ambiguous = dict.fromkeys(bilabial, bilabial)
    for sentence, expected in [("i met the cat", "met"), ("i bet the dog sat", "bet"), ("pet the cat", "pet")]:
        words = sentence.split()
        scores = peaky(inventory.size, inventory.encode(lexicon.phonemize(words)), alternatives=ambiguous)
        nbest = beam_search(scores, trie, lm, BeamConfig())
        assert expected in nbest[0].words, (sentence, nbest[0].words)
        index = words.index(expected)
        rivals = {h.words[index] for h in nbest if len(h.words) == len(words)}
        assert {"met", "bet", "pet"} <= rivals  # the alternatives stay in the N-best for the LLM


def test_beam_search_on_silence_and_empty_input(decoding, inventory):
    _, lm, trie = decoding
    silence = np.log(np.r_[0.999, np.full(inventory.size - 1, 0.001 / (inventory.size - 1))])[None].repeat(30, axis=0)
    assert beam_search(silence, trie, lm)[0].words == ()  # "nothing was said" is a valid answer
    assert beam_search(np.zeros((0, inventory.size)), trie, lm)[0].words == ()


def test_small_beam_still_finds_the_answer(decoding, inventory):
    lexicon, lm, trie = decoding
    words = "the dog sat on the log".split()
    scores = peaky(inventory.size, inventory.encode(lexicon.phonemize(words)), confidence=0.7)
    assert beam_search(scores, trie, lm, BeamConfig(beam_size=20, nbest=5))[0].words == tuple(words)


# -------------------------------------------------------------------- lattice
def test_confusion_network_and_paths(decoding, inventory):
    lexicon, lm, trie = decoding
    bilabial = [inventory.token_id(s) for s in "PBM"]
    words = "i met the cat".split()
    scores = peaky(
        inventory.size, inventory.encode(lexicon.phonemize(words)), alternatives=dict.fromkeys(bilabial, bilabial)
    )
    nbest = beam_search(scores, trie, lm, BeamConfig())
    assert sum(posteriors(nbest)) == pytest.approx(1.0)
    slots = confusion_network(nbest)
    required = [slot for slot in slots if not slot.optional]
    assert len(required) == 4
    assert {"met", "bet", "pet"} <= set(required[1].words)
    for slot in slots:
        assert sum(slot.words.values()) == pytest.approx(1.0, abs=0.05) or slot.optional
    assert "met|" in describe(slots) or "|met" in describe(slots)
    paths = best_paths(slots, lm, k=10)
    assert paths[0][0] == tuple(words)
    assert len({p for p, _ in paths}) == len(paths)
    assert all(a[1] >= b[1] for a, b in itertools.pairwise(paths))


def test_confusion_network_handles_insertions_and_deletions():
    from ssj.decode.beam import Hypothesis

    def hyp(text: str, score: float) -> Hypothesis:
        words = tuple(text.split())
        return Hypothesis(words, (), score, 0.0, score)

    slots = confusion_network(
        [hyp("turn the light on", 0.0), hyp("turn light on", -0.2), hyp("turn the light on now", -0.4)], scale=1.0
    )
    assert [slot.choices()[0] for slot in slots if not slot.optional] == ["turn", "light", "on"]
    optional = [slot for slot in slots if slot.optional]
    assert {"the"} <= set(optional[0].words) and EPS in optional[0].words  # "the" may be dropped
    assert optional[-1].choices() == ["now"]  # a trailing word may be added
    assert confusion_network([]) == []
    assert Slot({"a": 0.6, "": 0.4}).optional and not Slot({"a": 1.0}).optional
    assert EOS_ID == 1
