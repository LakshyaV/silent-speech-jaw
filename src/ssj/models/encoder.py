"""Phoneme recogniser: multi-stream TCN + BiGRU with CTC output.

The architecture keeps what worked for fixed commands - separate encoders for
the jaw and reference sensors, residual dilated temporal convolutions with
GroupNorm, masking of padded frames - and replaces the pooled classifier with a
per-frame output so that the network emits a phoneme sequence of any length:

    features [B, T, 23] @ 100 Hz
      -> per-session input adapter (affine, identity for unseen sessions)
      -> one conv stem per stream group (jaw / ref / rel / env), concatenated
      -> strided fusion conv                      (50 Hz)
      -> residual dilated depthwise-separable TCN (local articulatory context)
      -> bidirectional GRU                        (utterance-level context)
      -> linear -> phoneme logits [B, T/2, V]

The model is under a million parameters by default, small enough to train on
minutes of data and to run on a single Cortex-A53 core in real time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..dsp.frontend import FEATURE_GROUPS, N_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    streams: tuple[str, ...] = ("jaw", "ref", "rel", "env")  # stream groups fed to the model
    stem_channels: int = 24
    d_model: int = 128
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2)
    tcn_kernel: int = 5
    gru_layers: int = 2
    gru_hidden: int = 128
    dropout: float = 0.2
    subsample: int = 2
    intermediate_ctc: bool = True
    session_adapters: bool = True
    adapter_dropout: float = 0.3  # share of training batches that use the shared (identity) adapter

    def __post_init__(self) -> None:
        unknown = set(self.streams) - set(FEATURE_GROUPS)
        if unknown or not self.streams:
            raise ValueError(f"streams must be a non-empty subset of {sorted(FEATURE_GROUPS)}, got {self.streams}")


def _groups(channels: int) -> int:
    for g in (8, 4, 2):
        if channels % g == 0:
            return g
    return 1


def output_lengths(lengths: torch.Tensor, subsample: int) -> torch.Tensor:
    """Number of output frames for inputs of the given lengths (ceil division)."""
    return torch.div(lengths + subsample - 1, subsample, rounding_mode="floor")


class MaskedGroupNorm(nn.Module):
    """GroupNorm whose statistics ignore padded frames.

    Plain GroupNorm averages over time including padding, so an utterance's
    output would depend on how much padding its batch happens to add. Computing
    the statistics over valid frames only makes a batched utterance identical to
    the same utterance run alone, which is how the model is used at run time.
    """

    def __init__(self, groups: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return F.group_norm(x, self.groups, self.weight, self.bias, self.eps)
        batch, channels, steps = x.shape
        grouped = x.view(batch, self.groups, channels // self.groups, steps)
        valid = mask.view(batch, 1, 1, steps)
        count = (valid.sum(dim=(2, 3), keepdim=True) * (channels // self.groups)).clamp_min(1.0)
        mean = (grouped * valid).sum(dim=(2, 3), keepdim=True) / count
        variance = ((grouped - mean).pow(2) * valid).sum(dim=(2, 3), keepdim=True) / count
        normed = ((grouped - mean) * torch.rsqrt(variance + self.eps)).view(batch, channels, steps)
        return normed * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class ConvNormAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.norm = MaskedGroupNorm(_groups(c_out), c_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        y = F.silu(self.norm(self.conv(x), mask))
        return y * mask if mask is not None else y


class DepthwiseConv1d(nn.Conv1d):
    """Dilated depthwise convolution with a vectorised CPU path.

    PyTorch runs grouped convolutions on CPU one group at a time, which makes
    depthwise layers far slower to train than their arithmetic cost suggests.
    The same result is a sum of ``kernel`` shifted, per-channel-scaled copies of
    the input, which is what :meth:`forward` computes. With ``native = True``
    the stock convolution is used instead, so exported graphs contain a single
    depthwise ``Conv`` node that inference runtimes optimise well.
    """

    native = False

    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__(
            channels,
            channels,
            kernel,
            padding=dilation * (kernel - 1) // 2,
            dilation=dilation,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.native or x.is_cuda:
            return super().forward(x)
        dilation, steps = self.dilation[0], x.shape[-1]
        padded = F.pad(x, (self.padding[0], self.padding[0]))
        out = padded[:, :, :steps] * self.weight[:, 0, 0].view(1, -1, 1)
        for tap in range(1, self.kernel_size[0]):
            out = out + padded[:, :, tap * dilation : tap * dilation + steps] * self.weight[:, 0, tap].view(1, -1, 1)
        return out


class SeparableConv(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.depthwise = DepthwiseConv1d(channels, kernel, dilation)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = MaskedGroupNorm(_groups(channels), channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.norm(self.pointwise(self.depthwise(x)), mask)


class BiGRU(nn.Module):
    """Stacked bidirectional GRU over padded batches, without packed sequences.

    The backward direction runs a forward GRU over each sequence reversed
    *within its own length* (one gather), so padded frames never reach valid
    frames in either direction. This matches a packed bidirectional GRU exactly
    while avoiding the per-timestep gradient scatter that makes packed
    sequences slow to train on CPU.
    """

    def __init__(self, d_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        dims = [d_in] + [2 * hidden] * (layers - 1)
        self.forward_cells = nn.ModuleList(nn.GRU(d, hidden, batch_first=True) for d in dims)
        self.backward_cells = nn.ModuleList(nn.GRU(d, hidden, batch_first=True) for d in dims)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _reverse(x: torch.Tensor, index: torch.Tensor | None) -> torch.Tensor:
        if index is None:
            return x.flip(1)
        return torch.gather(x, 1, index[:, :, None].expand(-1, -1, x.shape[2]))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        index = None
        if lengths is not None:
            position = torch.arange(x.shape[1], device=x.device)[None, :]
            flipped = lengths[:, None] - 1 - position
            index = torch.where(flipped >= 0, flipped, position.expand_as(flipped))
        last = len(self.forward_cells) - 1
        for layer, (fwd, bwd) in enumerate(zip(self.forward_cells, self.backward_cells)):
            ahead, _ = fwd(x)
            behind, _ = bwd(self._reverse(x, index))
            x = torch.cat((ahead, self._reverse(behind, index)), dim=-1)
            if layer < last:
                x = self.dropout(x)
        return x


class TCNBlock(nn.Module):
    """Residual block of two dilated separable convolutions."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = SeparableConv(channels, kernel, dilation)
        self.conv2 = SeparableConv(channels, kernel, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        y = self.dropout(F.silu(self.conv1(x, mask)))
        if mask is not None:
            y = y * mask
        y = self.dropout(F.silu(self.conv2(y, mask) + x))
        return y * mask if mask is not None else y


class StreamStem(nn.Module):
    """Independent stems per sensor stream, fused by a strided convolution."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.slices = [FEATURE_GROUPS[name] for name in config.streams]
        self.stems = nn.ModuleList(ConvNormAct(b - a, config.stem_channels, 5) for a, b in self.slices)
        self.fuse = ConvNormAct(config.stem_channels * len(self.slices), config.d_model, 3, stride=config.subsample)

    def forward(self, x: torch.Tensor, mask_in: torch.Tensor | None, mask_out: torch.Tensor | None) -> torch.Tensor:
        parts = [stem(x[:, a:b], mask_in) for stem, (a, b) in zip(self.stems, self.slices)]  # x [B, C, T]
        return self.fuse(torch.cat(parts, dim=1), mask_out)


class JawPhonemeNet(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int, n_sessions: int = 0, n_features: int = N_FEATURES):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.n_sessions = n_sessions
        if config.session_adapters and n_sessions > 0:
            # Index 0 is the shared adapter and is pinned to identity.
            self.adapter_delta = nn.Parameter(torch.zeros(n_sessions + 1, n_features, n_features))
            self.adapter_bias = nn.Parameter(torch.zeros(n_sessions + 1, n_features))
        else:
            self.register_parameter("adapter_delta", None)
            self.register_parameter("adapter_bias", None)
        self.stem = StreamStem(config)
        self.blocks = nn.ModuleList(
            TCNBlock(config.d_model, config.tcn_kernel, d, config.dropout) for d in config.tcn_dilations
        )
        if config.gru_layers > 0:
            self.gru = BiGRU(config.d_model, config.gru_hidden, config.gru_layers, config.dropout)
            self.gru_proj = nn.Linear(2 * config.gru_hidden, config.d_model)
            self.gru_norm = nn.LayerNorm(config.d_model)
        else:
            self.gru = None
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, vocab_size)

    # ------------------------------------------------------------- adapters
    def adapt(self, x: torch.Tensor, session: torch.Tensor | None) -> torch.Tensor:
        """Apply per-session affine input adapters (``session`` holds 1-based ids, 0 = shared)."""
        if self.adapter_delta is None or session is None:
            return x
        if self.training and self.config.adapter_dropout > 0:
            drop = torch.rand(session.shape, device=session.device) < self.config.adapter_dropout
            session = torch.where(drop, torch.zeros_like(session), session)
        active = (session > 0).to(x.dtype)[:, None, None]
        delta = self.adapter_delta[session] * active
        bias = self.adapter_bias[session] * active[:, :, 0]
        return x + torch.bmm(x, delta.transpose(1, 2)) + bias[:, None, :]

    def adapter_penalty(self) -> torch.Tensor:
        """L2 distance of the session adapters from identity."""
        if self.adapter_delta is None:
            return torch.zeros((), device=self.head.weight.device)
        return self.adapter_delta[1:].pow(2).mean() + self.adapter_bias[1:].pow(2).mean()

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,  # [B, T, C] normalised features
        lengths: torch.Tensor | None = None,  # [B] valid input frames
        session: torch.Tensor | None = None,  # [B] session ids (0 = shared adapter)
    ) -> dict[str, torch.Tensor | None]:
        x = self.adapt(x, session)
        frames = x.shape[1]
        steps = (frames + self.config.subsample - 1) // self.config.subsample
        if lengths is not None:
            out_lengths = output_lengths(lengths, self.config.subsample).clamp(max=steps)
            mask_in = (torch.arange(frames, device=x.device)[None, :] < lengths[:, None]).to(x.dtype)[:, None, :]
            mask = (torch.arange(steps, device=x.device)[None, :] < out_lengths[:, None]).to(x.dtype)[:, None, :]
            x = x * mask_in.transpose(1, 2)
        else:
            out_lengths, mask_in, mask = None, None, None
        h = self.stem(x.transpose(1, 2), mask_in, mask)  # [B, D, T']
        for block in self.blocks:
            h = block(h, mask)
        h = h.transpose(1, 2)  # [B, T', D]

        inter = self.head(self.dropout(h)) if (self.config.intermediate_ctc and self.training) else None
        if self.gru is not None:
            h = self.gru_norm(h + self.gru_proj(self.gru(h, out_lengths)))
        logits = self.head(self.dropout(h))
        return {"logits": logits, "lengths": out_lengths, "inter_logits": inter}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ExportWrapper(nn.Module):
    """Single-utterance inference graph: features -> phoneme log-posteriors.

    A session adapter can be baked in, which is how a model adapted to the
    current mounting is shipped without changing the runtime.
    """

    def __init__(self, model: JawPhonemeNet, session_id: int = 0):
        super().__init__()
        self.model = copy.deepcopy(model).eval()
        self.session_id = session_id
        for module in self.model.modules():  # export stock depthwise convolutions
            if isinstance(module, DepthwiseConv1d):
                module.native = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [1, T, C] -> [1, T', V]
        model = self.model
        if self.session_id > 0 and model.adapter_delta is not None:
            x = (
                x
                + torch.matmul(x, model.adapter_delta[self.session_id].transpose(0, 1))
                + model.adapter_bias[self.session_id]
            )
        h = model.stem(x.transpose(1, 2), None, None)
        for block in model.blocks:
            h = block(h, None)
        h = h.transpose(1, 2)
        if model.gru is not None:
            h = model.gru_norm(h + model.gru_proj(model.gru(h)))
        return F.log_softmax(model.head(h), dim=-1)
