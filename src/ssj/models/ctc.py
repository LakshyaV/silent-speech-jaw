"""Multi-resolution CTC objective and greedy decoding.

A jaw IMU cannot distinguish every phoneme, so a plain phoneme CTC loss spends
its capacity punishing confusions the sensor makes unavoidable, and the model
over-fits spurious cues to tell P from B. The objective here trains one output
distribution at two resolutions at once:

    L = (1 - w) * CTC(p_phoneme, y) + w * CTC(p_class, class(y))

where ``p_class`` is not a second head but the *marginal* of the phoneme
distribution over each articulatory class (log-sum-exp over its members). The
class term rewards putting probability mass on the right class regardless of how
it is split inside the class; the phoneme term keeps whatever finer distinction
the data supports. Because both are views of the same distribution they share
one time alignment, and the class posterior used at decode time is exact.

Class targets are the plain image of the phoneme targets (adjacent repeats are
kept), so any alignment satisfying the phoneme target also satisfies the class
target: the two terms never pull in opposite directions.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..decode.ctc_score import greedy_decode
from ..phonology import BLANK_ID, Inventory


class MultiResolutionCTC(nn.Module):
    def __init__(self, inventory: Inventory, coarse_weight: float = 0.3, inter_weight: float = 0.3):
        super().__init__()
        if not 0.0 <= coarse_weight <= 1.0:
            raise ValueError("coarse_weight must be in [0, 1]")
        self.coarse_weight = coarse_weight
        self.inter_weight = inter_weight
        membership = torch.from_numpy(inventory.membership)
        self.register_buffer("log_membership", torch.where(membership, 0.0, float("-inf")), persistent=False)
        self.register_buffer("class_ids", torch.from_numpy(inventory.class_ids), persistent=False)

    def marginalize(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Phoneme log-posteriors ``[..., V]`` -> class log-posteriors ``[..., K]``."""
        return torch.logsumexp(log_probs.unsqueeze(-1) + self.log_membership, dim=-2)

    @staticmethod
    def _ctc(
        log_probs: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor
    ) -> torch.Tensor:
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            targets,
            lengths,
            target_lengths,
            blank=BLANK_ID,
            reduction="sum",
            zero_infinity=True,
        )
        return loss / target_lengths.sum().clamp_min(1)

    def _two_level(
        self, logits, lengths, targets, target_lengths, class_targets
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        fine = self._ctc(log_probs, lengths, targets, target_lengths)
        coarse = self._ctc(self.marginalize(log_probs), lengths, class_targets, target_lengths)
        return (1 - self.coarse_weight) * fine + self.coarse_weight * coarse, fine, coarse

    def forward(
        self,
        logits: torch.Tensor,  # [B, T, V]
        lengths: torch.Tensor,  # [B] valid output frames
        targets: torch.Tensor,  # [B, S] padded phoneme ids
        target_lengths: torch.Tensor,  # [B]
        inter_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        class_targets = self.class_ids[targets]
        loss, fine, coarse = self._two_level(logits, lengths, targets, target_lengths, class_targets)
        parts = {"phoneme": float(fine.detach()), "class": float(coarse.detach())}
        if inter_logits is not None and self.inter_weight > 0:
            inter, _, _ = self._two_level(inter_logits, lengths, targets, target_lengths, class_targets)
            loss = (1 - self.inter_weight) * loss + self.inter_weight * inter
            parts["intermediate"] = float(inter.detach())
        return loss, parts


def greedy_decode_batch(log_probs: torch.Tensor, lengths: Sequence[int]) -> list[list[int]]:
    array = log_probs.detach().float().cpu().numpy()
    return [greedy_decode(array[i, : int(n)]) for i, n in enumerate(lengths)]
