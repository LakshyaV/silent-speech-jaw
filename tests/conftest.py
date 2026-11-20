"""Shared fixtures.

The expensive ones are session-scoped and build on each other, mirroring the
pipeline: simulated sessions -> feature dataset -> trained run -> exported bundle
-> runtime pipeline. Imports are local so that collecting the tests of one layer
does not import the others.
"""

from __future__ import annotations

import pytest

# === phonology ===


@pytest.fixture(scope="session")
def inventory():
    from ssj.phonology import Inventory

    return Inventory.default()


@pytest.fixture(scope="session")
def lexicon(inventory):
    from ssj.phonology import Lexicon

    return Lexicon.core(inventory)


# === simulator ===

SIM_SENTENCES = [
    "hello how are you today",
    "i need a glass of water",
    "please open the window",
    "what time is it",
    "thank you very much",
    "the weather is beautiful today",
    "can you help me please",
    "turn the lights off",
    "i am feeling tired",
    "where are we going",
    "good morning everyone",
    "see you later",
]
SIM_WORDS = ["water", "hello", "today", "window", "please", "morning", "happy", "open"]


@pytest.fixture(scope="session")
def sim_session(lexicon):
    """One simulated session (calibration routine + 12 sentences) with ground truth."""
    from ssj.sim import simulate_session

    return simulate_session(SIM_SENTENCES, lexicon, seed=3)


@pytest.fixture(scope="session")
def sim_root(tmp_path_factory, lexicon):
    """Three simulated sessions on disk, each with its own random sensor mounting."""
    import numpy as np

    from ssj.sim import simulate_session

    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    for index in range(3):
        texts = [*SIM_SENTENCES, *SIM_SENTENCES[:6], *SIM_WORDS]
        rng.shuffle(texts)
        session = simulate_session(texts, lexicon, seed=100 + index, clench_rate=0.1, utt_prefix="u")
        session.write(root / "sim" / f"s{index:02d}", subject="sim", session=f"s{index:02d}")
    return root


# === dataset ===


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory, sim_root, lexicon, inventory):
    from ssj.data import build_dataset

    out = tmp_path_factory.mktemp("dataset")
    build_dataset(sim_root, out, lexicon, inventory)
    return out
