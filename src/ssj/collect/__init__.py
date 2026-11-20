"""Data collection: prompt design, guided recording, quality checks."""

from .prompts import Prompt, build_prompt_set, builtin_sentences, coverage_report, load_prompts, save_prompts
from .qc import QCConfig, QCResult, check_stream, check_utterance
from .recorder import ConsoleUI, Recorder, RecorderConfig

__all__ = [
    "ConsoleUI",
    "Prompt",
    "QCConfig",
    "QCResult",
    "Recorder",
    "RecorderConfig",
    "build_prompt_set",
    "builtin_sentences",
    "check_stream",
    "check_utterance",
    "coverage_report",
    "load_prompts",
    "save_prompts",
]
