"""Language resources: where they live and how they are fetched.

Three kinds of resources sit beside the model, all kept outside the package in
an assets directory (``./assets`` or ``$SSJ_ASSETS``)::

    lexicon/cmudict.dict     full pronunciation dictionary (~135k words)
    corpus/sentences.txt     text for the n-gram model and for prompt selection
    lm/en.npz                n-gram model built by ``ssj lm build``
    llm/<name>.gguf          quantized LLM

``ssj assets fetch`` downloads the first, second and last; ``ssj lm build``
derives the third. Files found at these conventional paths are picked up
automatically by ``ssj eval``, ``ssj decode`` and ``ssj live``.
"""

from __future__ import annotations

import bz2
import os
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "silent-speech-jaw"
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"


@dataclass(frozen=True)
class LLMPreset:
    repo: str
    filename: str
    megabytes: int
    note: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"


#: Small instruction-tuned models in 4-bit GGUF form, smallest first.
LLM_PRESETS: dict[str, LLMPreset] = {
    "tiny": LLMPreset(
        "bartowski/SmolLM2-360M-Instruct-GGUF",
        "SmolLM2-360M-Instruct-Q4_K_M.gguf",
        271,
        "fits beside the recogniser on a 2 GB board",
    ),
    "small": LLMPreset(
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        491,
        "default: good ranking quality for its size",
    ),
    "base": LLMPreset(
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", 1117, "4 GB boards and laptops"
    ),
    "large": LLMPreset(
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        2019,
        "laptops; strongest context modelling",
    ),
}


def assets_dir(explicit: str | Path | None = None) -> Path:
    return Path(explicit or os.environ.get("SSJ_ASSETS") or "assets")


def lexicon_path(root: Path) -> Path:
    return root / "lexicon" / "cmudict.dict"


def corpus_path(root: Path) -> Path:
    return root / "corpus" / "sentences.txt"


def lm_path(root: Path) -> Path:
    return root / "lm" / "en.npz"


def find_llm(root: Path) -> Path | None:
    """The GGUF model to use by default: ``$SSJ_LLM`` or the only one in ``llm/``."""
    env = os.environ.get("SSJ_LLM")
    if env:
        return Path(env)
    models = sorted((root / "llm").glob("*.gguf")) if (root / "llm").is_dir() else []
    return models[0] if len(models) == 1 else None


def download(url: str, destination: Path, *, report: Callable[[str], None] | None = None) -> Path:
    """Download to ``destination`` atomically, resuming a partial file if present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = response.status == 206
        if offset and not resumed:
            offset = 0  # the server ignored the range request: start over
        total = int(response.headers.get("Content-Length") or 0) + offset
        done, last = offset, -1
        with partial.open("ab" if resumed else "wb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                percent = int(100 * done / total) if total else -1
                if report and percent != last and (percent % 5 == 0 or percent < 0):
                    report(f"  {destination.name}: {done / 1e6:,.0f} MB" + (f" ({percent}%)" if total else ""))
                    last = percent
    partial.replace(destination)
    return destination


def fetch_lexicon(root: Path, report: Callable[[str], None] | None = None) -> Path:
    return download(CMUDICT_URL, lexicon_path(root), report=report)


def fetch_corpus(root: Path, report: Callable[[str], None] | None = None) -> Path:
    """Download the Tatoeba English sentences and store them one per line."""
    target = corpus_path(root)
    if target.exists():
        return target
    archive = download(TATOEBA_URL, root / "corpus" / "eng_sentences.tsv.bz2", report=report)
    temporary = target.with_suffix(".part")
    count = 0
    with (
        bz2.open(archive, "rt", encoding="utf-8", errors="ignore") as source,
        temporary.open("w", encoding="utf-8") as sink,
    ):
        for line in source:
            sentence = line.rstrip("\n").rsplit("\t", 1)[-1].strip()
            if sentence:
                sink.write(sentence + "\n")
                count += 1
    temporary.replace(target)
    archive.unlink()
    if report:
        report(f"  {target.name}: {count:,} sentences")
    return target


def fetch_llm(root: Path, preset: str = "small", report: Callable[[str], None] | None = None) -> Path:
    if preset not in LLM_PRESETS:
        raise KeyError(f"unknown LLM preset {preset!r}; choose from {sorted(LLM_PRESETS)}")
    spec = LLM_PRESETS[preset]
    return download(spec.url, root / "llm" / spec.filename.lower(), report=report)


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
