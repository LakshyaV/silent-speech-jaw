"""Local LLM backends.

The LLM is used in two ways, both local and both cheap enough for a small
quantized model:

``score``     log-probability of candidate sentences in context. One forward
              pass per candidate over a dozen tokens; the shared prompt prefix is
              evaluated once and reused from the KV cache.
``generate``  short greedy generation, optionally constrained by a GBNF grammar
              so the model can only emit words the sensor supports.

:class:`LlamaCppBackend` runs any GGUF model through llama.cpp. The interface is
small on purpose: anything that can score and generate can be dropped in.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    model: str = ""  # path to a GGUF file; empty disables the LLM stage
    n_ctx: int = 1024
    n_threads: int = 0  # 0 = llama.cpp default
    n_gpu_layers: int = 0  # -1 offloads everything (Metal / CUDA builds)
    seed: int = 0


class LLMBackend(ABC):
    name = "llm"

    @abstractmethod
    def score(self, context: str, candidates: Sequence[str]) -> list[float]:
        """``log P(candidate | context)`` in nats for each candidate continuation."""

    @abstractmethod
    def generate(self, system: str, user: str, *, max_tokens: int = 48, grammar: str | None = None) -> str:
        """Greedy completion of a chat prompt, optionally grammar-constrained."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


def _log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class LlamaCppBackend(LLMBackend):
    """GGUF model via llama-cpp-python (CPU, Metal or CUDA depending on the build)."""

    name = "llama.cpp"

    def __init__(self, config: LLMConfig):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise LLMError("llama-cpp-python is required: pip install 'silent-speech-jaw[llm]'") from exc
        path = Path(config.model).expanduser()
        if not path.exists():
            raise LLMError(f"LLM model not found: {path} (fetch one with `ssj assets fetch llm`)")
        self.config = config
        self._llm = Llama(
            model_path=str(path),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads or None,
            n_gpu_layers=config.n_gpu_layers,
            seed=config.seed,
            logits_all=True,  # per-position logits are what scoring reads
            verbose=False,
        )
        self._evaluated: list[int] = []
        self.name = f"llama.cpp:{path.name}"

    # ------------------------------------------------------------- internals
    def _tokenize(self, text: str, *, bos: bool) -> list[int]:
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=bos, special=False))

    def _eval_to(self, tokens: list[int]) -> None:
        """Bring the model state to ``tokens``, re-using the longest evaluated prefix."""
        llm = self._llm
        common = 0
        for a, b in zip(self._evaluated, tokens):
            if a != b:
                break
            common += 1
        if common == len(tokens) and common > 0:
            common -= 1  # re-evaluate the last token so its logits row is fresh
        if common == 0:
            llm.reset()
        else:
            llm.n_tokens = common
        llm.eval(tokens[common:])
        self._evaluated = list(tokens)

    # ------------------------------------------------------------------ API
    def score(self, context: str, candidates: Sequence[str]) -> list[float]:
        llm = self._llm
        prefix = self._tokenize(context, bos=True)
        if not prefix:
            prefix = [llm.token_bos()]
        limit = llm.n_ctx() - 8
        results: list[float] = []
        for candidate in candidates:
            tokens = self._tokenize(candidate, bos=False)
            if not tokens:
                results.append(0.0)
                continue
            head = prefix
            if len(head) + len(tokens) > limit:  # keep BOS, drop the oldest context
                keep = max(limit - len(tokens) - 1, 0)
                head = prefix[:1] + (prefix[len(prefix) - keep :] if keep else [])
            sequence = head + tokens
            self._eval_to(sequence)
            rows = np.asarray(llm.scores[len(head) - 1 : len(sequence) - 1], dtype=np.float64)
            log_probs = _log_softmax_rows(rows)
            results.append(float(log_probs[np.arange(len(tokens)), tokens].sum()))
        return results

    def generate(self, system: str, user: str, *, max_tokens: int = 48, grammar: str | None = None) -> str:
        from llama_cpp import LlamaGrammar

        compiled = LlamaGrammar.from_string(grammar, verbose=False) if grammar else None
        self._evaluated = []  # chat completion replaces the model state
        self._llm.reset()
        response = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
            grammar=compiled,
        )
        return str(response["choices"][0]["message"]["content"] or "")

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if close is not None:
            close()


class NullBackend(LLMBackend):
    """Stand-in used when no LLM is configured: neutral scores, no proposals."""

    name = "none"

    def score(self, context: str, candidates: Sequence[str]) -> list[float]:
        return [0.0] * len(candidates)

    def generate(self, system: str, user: str, *, max_tokens: int = 48, grammar: str | None = None) -> str:
        return ""


def load_backend(config: LLMConfig | None) -> LLMBackend:
    """Build the configured backend; without a model path this is :class:`NullBackend`.

    The ``SSJ_LLM`` environment variable supplies a model path when the
    configuration leaves it empty.
    """
    config = config or LLMConfig()
    model = config.model or os.environ.get("SSJ_LLM", "")
    if not model:
        return NullBackend()
    return LlamaCppBackend(replace(config, model=model))
