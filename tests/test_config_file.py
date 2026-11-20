"""``configs/default.yaml`` documents the defaults; it must never drift from the code."""

from __future__ import annotations

from pathlib import Path

from ssj.collect.qc import QCConfig
from ssj.collect.recorder import RecorderConfig
from ssj.config import from_dict, load_yaml, to_dict
from ssj.data.augment import AugmentConfig
from ssj.data.build import DatasetConfig
from ssj.models.encoder import ModelConfig
from ssj.runtime.pipeline import PipelineConfig
from ssj.train.adapt import AdaptConfig
from ssj.train.trainer import TrainConfig

SECTIONS = {
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "augment": AugmentConfig,
    "adapt": AdaptConfig,
    "recorder": RecorderConfig,
    "qc": QCConfig,
    "pipeline": PipelineConfig,
}


def test_default_config_matches_the_code():
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert list(config) == list(SECTIONS)
    for name, cls in SECTIONS.items():
        assert from_dict(cls, config[name]) == cls(), f"{name}: run tools/write_default_config.py"
        assert config[name] == to_dict(cls()), f"{name}: run tools/write_default_config.py"
