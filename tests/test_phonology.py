from __future__ import annotations

import numpy as np
import pytest

from ssj.phonology import (
    BLANK_ID,
    CONTROL_WORDS,
    Inventory,
    InventoryError,
    Lexicon,
    OOVError,
    normalize_text,
    number_to_words,
    render_sentence,
)


def test_default_inventory_shape(inventory):
    assert inventory.size == 41  # 39 ARPAbet phonemes + CLENCH + blank
    assert inventory.n_classes == 20
    assert inventory.gestures == ("CLENCH",)
    assert inventory.members("C_BILABIAL") == ("P", "B", "M")
    assert inventory.is_vowel("AA") and not inventory.is_vowel("S")


def test_encode_decode_round_trip(inventory):
    symbols = ["HH", "AH", "L", "OW"]
    ids = inventory.encode(symbols)
    assert BLANK_ID not in ids
    assert inventory.decode(ids) == symbols
    with pytest.raises(InventoryError):
        inventory.token_id("XX")


def test_class_structure_is_a_partition(inventory):
    membership = inventory.membership
    assert membership.shape == (inventory.size, inventory.n_classes)
    assert np.all(membership.sum(axis=1) == 1)  # every token in exactly one class
    assert np.all(membership.sum(axis=0) >= 1)  # no empty class
    assert membership[BLANK_ID, BLANK_ID]
    assert inventory.class_ids[BLANK_ID] == BLANK_ID


def test_voicing_pairs_share_a_class(inventory):
    for a, b in [("P", "B"), ("T", "D"), ("K", "G"), ("F", "V"), ("S", "Z"), ("SH", "ZH"), ("CH", "JH"), ("TH", "DH")]:
        assert inventory.to_class_ids(inventory.encode([a])) == inventory.to_class_ids(inventory.encode([b]))


def test_inventory_serialisation_and_refinement(inventory, tmp_path):
    assert Inventory.from_dict(inventory.to_dict()) == inventory
    mapping = {p: ("EVERYTHING" if k != "gesture" else "G") for p, k in zip(inventory.phonemes, inventory.kinds)}
    coarse = inventory.with_class_map(mapping)
    assert coarse.phonemes == inventory.phonemes  # the output layer does not change
    assert coarse.n_classes == 3
    assert coarse.fingerprint != inventory.fingerprint
    with pytest.raises(InventoryError):
        inventory.with_class_map({"P": "X"})
    slim = inventory.without(["CLENCH"])
    assert slim.size == 40 and "G_CLENCH" not in slim.classes


def test_normalize_text():
    assert normalize_text("It's 21 o'clock — I'll take the 3rd one!") == [
        "it's", "twenty", "one", "o'clock", "i'll", "take", "the", "third", "one",
    ]  # fmt: skip
    assert normalize_text("1,250 items & 7%") == [
        "one",
        "thousand",
        "two",
        "hundred",
        "fifty",
        "items",
        "and",
        "seven",
        "percent",
    ]
    assert normalize_text("Café <clench> déjà") == ["cafe", "<clench>", "deja"]
    assert normalize_text("  ...  ") == []
    assert normalize_text("room 2b") == ["room", "two", "b"]


@pytest.mark.parametrize(
    "number,words",
    [
        (0, "zero"),
        (13, "thirteen"),
        (40, "forty"),
        (99, "ninety nine"),
        (101, "one hundred one"),
        (2026, "two thousand twenty six"),
    ],
)
def test_number_to_words(number, words):
    assert " ".join(number_to_words(number)) == words


def test_render_sentence():
    assert render_sentence(["i", "don't", "know", "<clench>"]) == "I don't know."
    assert render_sentence(["hello"], punctuate=False) == "Hello"
    assert render_sentence(["<clench>"]) == ""


def test_core_lexicon(lexicon, inventory):
    assert len(lexicon) > 5000
    assert lexicon.prons("read") == [("R", "EH", "D"), ("R", "IY", "D")]
    assert lexicon.prons("<clench>") == [("CLENCH",)]
    assert set(CONTROL_WORDS) <= set(lexicon.words)
    assert lexicon.phonemize("Hello, world!") == ["HH", "AH", "L", "OW", "W", "ER", "L", "D"]
    for word in lexicon.words[:500]:  # every pronunciation uses inventory symbols only
        for pron in lexicon.prons(word):
            assert all(p in inventory for p in pron)


def test_out_of_vocabulary_is_an_error(lexicon):
    with pytest.raises(OOVError) as error:
        lexicon.phonemize("hello zzyzx qwfp")
    assert error.value.words == ("zzyzx", "qwfp")


def test_lexicon_file_round_trip(tmp_path, inventory):
    source = tmp_path / "mini.dict"
    source.write_text(
        ";;; comment\nhello HH AH0 L OW1\nhello(2) HH EH0 L OW1\nworld W ER1 L D # trailing comment\nbad!word B AE1 D\n"
    )
    lexicon = Lexicon.load(source, inventory=inventory)
    assert lexicon.words == ["hello", "world"]
    assert lexicon.prons("hello") == [("HH", "AH", "L", "OW"), ("HH", "EH", "L", "OW")]
    lexicon.save(tmp_path / "out.dict")
    again = Lexicon.load(tmp_path / "out.dict")
    assert dict(again.items()) == dict(lexicon.items())
    assert Lexicon.load(source, keep=["world"]).words == ["world"]


def test_homophenes_exist_and_are_grouped(lexicon, inventory):
    groups = lexicon.homophene_sets(inventory)
    by_word = {word: members for members in groups.values() for word in members}
    # P/B/M differ only inside the bilabial class, so these collapse together.
    assert {"pet", "bet", "met"} <= set(by_word["pet"])
    assert max(len(members) for members in groups.values()) >= 5
