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
