"""PyTorch dataset, collation and length-bucketed batching."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .augment import AugmentConfig, augment
from .build import FeatureDataset, Item


class UtteranceDataset(Dataset):
    """Utterances of one split, normalised and (optionally) augmented."""

    def __init__(
        self,
        data: FeatureDataset,
        items: Sequence[Item],
        session_index: dict[str, int],
        *,
        augment_config: AugmentConfig | None = None,
        seed: int = 0,
        label_permutation: Sequence[int] | None = None,
    ):
        self.data = data
        self.items = list(items)
        self.session_index = session_index
        self.augment_config = augment_config
        self.frontend_config = data.frontend_config
        self.rng = np.random.default_rng(seed)
        self._feats = [data.load(item) for item in self.items]
        self._labels = [data.inventory.encode(item.phonemes) for item in self.items]
        if label_permutation is not None:  # shuffled-label control
            self._labels = [self._labels[i] for i in label_permutation]
        self._by_session: dict[str, list[int]] = {}
        for index, item in enumerate(self.items):
            self._by_session.setdefault(item.session_key, []).append(index)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def lengths(self) -> list[int]:
        return [len(f) for f in self._feats]

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        feats, labels = self._feats[index], list(self._labels[index])
        cfg = self.augment_config
        if cfg is not None and cfg.enabled:
            feats = augment(feats, self.rng, cfg, self.frontend_config)
            if cfg.concat_prob > 0 and self.rng.random() < cfg.concat_prob:
                other = int(self.rng.choice(self._by_session[item.session_key]))
                gap = np.zeros((int(self.rng.integers(10, 40)), feats.shape[1]), dtype=np.float32)
                feats = np.concatenate((feats, gap, augment(self._feats[other], self.rng, cfg, self.frontend_config)))
                labels = labels + list(self._labels[other])
        return {
            "x": torch.from_numpy(self.data.normalize(feats)),
            "y": torch.tensor(labels, dtype=torch.long),
            "session": self.session_index.get(item.session_key, 0),
            "utt_id": item.utt_id,
        }


def collate(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(b["x"]) for b in batch], dtype=torch.long)
    target_lengths = torch.tensor([len(b["y"]) for b in batch], dtype=torch.long)
    x = torch.zeros(len(batch), int(lengths.max()), batch[0]["x"].shape[1])
    y = torch.zeros(len(batch), max(int(target_lengths.max()), 1), dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["x"])] = b["x"]
        y[i, : len(b["y"])] = b["y"]
    return {
        "x": x,
        "lengths": lengths,
        "y": y,
        "target_lengths": target_lengths,
        "session": torch.tensor([b["session"] for b in batch], dtype=torch.long),
        "utt_ids": [b["utt_id"] for b in batch],
    }


class BucketSampler(Sampler[list[int]]):
    """Batches of similar length, in random order.

    Similar lengths keep padding (and its effect on normalisation layers) small.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        curriculum_epochs: int = 0,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.curriculum_epochs = curriculum_epochs
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.lengths)
        self._epoch += 1
        if self._epoch <= self.curriculum_epochs:
            # Shortest utterances first: CTC finds its first alignments on isolated
            # words far sooner than on sentences, which shortens the initial plateau.
            order = np.argsort(self.lengths, kind="stable")
            return iter([order[i : i + self.batch_size].tolist() for i in range(0, n, self.batch_size)])
        if self.shuffle:
            # Sort by length within large random chunks: similar lengths, different batches each epoch.
            order = self.rng.permutation(n)
            chunk = self.batch_size * 20
            order = np.concatenate(
                [c[np.argsort(self.lengths[c], kind="stable")] for c in np.array_split(order, max(1, n // chunk))]
            )
        else:
            order = np.argsort(self.lengths, kind="stable")
        batches = [order[i : i + self.batch_size].tolist() for i in range(0, n, self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)
