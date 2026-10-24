"""Learner-side backbone slot.

A backbone is a torch module that turns a sequence of encoded observations
(optionally fused with drive values) into per-step action logits and value
estimates. Backbones are registered in the existing plugin registry under the
``backbone`` type. This module is imported only by training / torch plugin
paths; ``simlab.core`` and ``simlab.cli`` never import it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from simlab.core.registry import register

N_LOGITS = 3
MAX_N_DRIVES = 8


def _param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class Backbone(ABC):
    """Sequence-to-sequence recurrent actor/critic module."""

    state_dim: int
    obs_dim: int
    fusion: str
    n_drives: int

    @abstractmethod
    def initial_state(self, batch: int) -> torch.Tensor:
        """Zero recurrent state, shape ``(batch, state_dim)``."""

    @abstractmethod
    def frame_step(
        self, x: torch.Tensor, h: torch.Tensor, drive: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step. ``x (B, obs_dim)``, ``h (B, state_dim)``,
        ``drive (B, n_drives)`` or None. Returns ``(a (B,3), v (B,),
        f (B, state_dim))``."""

    @abstractmethod
    def param_count(self) -> int: ...

    @abstractmethod
    def config_dict(self) -> dict[str, Any]: ...

    def forward(
        self,
        obs_seq: torch.Tensor,
        drive_seq: torch.Tensor | None,
        state: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence version of ``frame_step``.

        ``obs_seq (B,T,obs_dim)``, ``drive_seq (B,T,n_drives)`` or None,
        ``state (B,state_dim)``, ``done_mask (B,T)`` (1 = keep, 0 = reset).
        Returns ``logits (B,T,3)``, ``value (B,T)``, ``new_state (B,state_dim)``.
        """
        _b, t = obs_seq.shape[0], obs_seq.shape[1]
        h = state
        logits: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for step in range(t):
            m = done_mask[:, step]
            h = h * m.unsqueeze(-1)
            drive = None if drive_seq is None else drive_seq[:, step]
            a, v, f = self.frame_step(obs_seq[:, step], h, drive)
            logits.append(a)
            values.append(v)
            h = f
        return torch.stack(logits, dim=1), torch.stack(values, dim=1), h


class GRUBackbone(Backbone):
    """The existing GRU cell, reproduced exactly (no behavior change)."""

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 64,
        fusion: str = "concat",
        n_drives: int = MAX_N_DRIVES,
        seed: int = 0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.fusion = fusion
        self.n_drives = int(n_drives)
        self.state_dim = self.hidden
        net = nn.Module()
        net.gru_ih = nn.Linear(self.obs_dim, self.hidden)
        net.gru_hh = nn.Linear(self.hidden, self.hidden)
        net.policy_head = nn.Linear(self.hidden, N_LOGITS)
        net.value_head = nn.Linear(self.hidden, 1)
        net.log_std_turn = nn.Parameter(torch.zeros(()))
        net.log_std_thrust = nn.Parameter(torch.zeros(()))
        if fusion == "film":
            net.film_scale = nn.Linear(self.n_drives, self.hidden)
            net.film_shift = nn.Linear(self.n_drives, self.hidden)
        self._net = net

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((int(batch), self.state_dim))

    def frame_step(self, x, h, drive=None):
        net = self._net
        if drive is not None and self.fusion == "film":
            f = torch.tanh(
                net.gru_ih(x)
                + net.film_scale(drive) * net.gru_hh(h)
                + net.film_shift(drive)
            )
        else:
            f = torch.tanh(net.gru_ih(x) + net.gru_hh(h))
        a = net.policy_head(f)
        v = net.value_head(f).squeeze(-1)
        return a, v, f

    def param_count(self) -> int:
        return _param_count(self._net)

    def config_dict(self) -> dict[str, Any]:
        return {
            "backbone": "gru",
            "hidden": self.hidden,
            "fusion": self.fusion,
            "n_drives": self.n_drives,
        }


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)


class TransformerBackbone(Backbone):
    """Small causal Transformer over the last ``context`` embedded obs steps.

    The recurrent state is a ring buffer of the last ``context`` embeddings.
    """

    def __init__(
        self,
        obs_dim: int,
        context: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        fusion: str = "concat",
        n_drives: int = MAX_N_DRIVES,
        seed: int = 0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.context = int(context)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.fusion = fusion
        self.n_drives = int(n_drives)
        self.state_dim = self.context * self.d_model
        net = nn.Module()
        net.in_proj = nn.Linear(self.obs_dim, self.d_model)
        net.pos = nn.Parameter(torch.zeros(1, self.context, self.d_model))
        blocks = nn.ModuleList(
            [
                _TransformerBlock(self.d_model, self.n_heads)
                for _ in range(self.n_layers)
            ]
        )
        net.blocks = blocks
        net.policy_head = nn.Linear(self.d_model, N_LOGITS)
        net.value_head = nn.Linear(self.d_model, 1)
        net.log_std_turn = nn.Parameter(torch.zeros(()))
        net.log_std_thrust = nn.Parameter(torch.zeros(()))
        if fusion == "film":
            net.film_scale = nn.Linear(self.n_drives, self.d_model)
            net.film_shift = nn.Linear(self.n_drives, self.d_model)
        self._net = net

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((int(batch), self.state_dim))

    def frame_step(self, x, h, drive=None):
        net = self._net
        b = x.shape[0]
        buf = h.view(b, self.context, self.d_model)
        e = net.in_proj(x)
        if drive is not None and self.fusion == "film":
            e = e * net.film_scale(drive) + net.film_shift(drive)
        buf = torch.cat([buf[:, 1:], e.unsqueeze(1)], dim=1)
        buf = buf + net.pos
        causal = torch.triu(
            torch.full((self.context, self.context), float("-inf"), device=x.device),
            diagonal=1,
        )
        out = buf
        for block in net.blocks:
            attn_out, _ = block.attn(out, out, out, attn_mask=causal)
            out = block.ln1(out + attn_out)
            out = block.ln2(out + block.ff(out))
        last = out[:, -1]
        a = net.policy_head(last)
        v = net.value_head(last).squeeze(-1)
        f = buf.contiguous().view(b, self.state_dim)
        return a, v, f

    def param_count(self) -> int:
        return _param_count(self._net)

    def config_dict(self) -> dict[str, Any]:
        return {
            "backbone": "transformer",
            "context": self.context,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "fusion": self.fusion,
            "n_drives": self.n_drives,
        }


class RDABackbone(Backbone):
    """Placeholder for the user-supplied RDA architecture."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "backbone 'rda' is a stub; supply your own implementation through "
            "the registered 'backbone' slot and rebuild the policy."
        )

    def initial_state(self, batch: int) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def frame_step(self, x, h, drive=None):  # pragma: no cover
        raise NotImplementedError

    def param_count(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def config_dict(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


BACKBONES: dict[str, type[Backbone]] = {
    "gru": GRUBackbone,
    "transformer": TransformerBackbone,
    "rda": RDABackbone,
}


def match_params(
    target: int,
    backbone: str,
    obs_dim: int,
    fusion: str = "concat",
    n_drives: int = MAX_N_DRIVES,
    context: int = 16,
) -> int:
    """Pick a hidden size (hidden / d_model) whose parameter count is within 5%
    of ``target``."""
    cls = BACKBONES[backbone]
    target = int(target)
    best: tuple[int, int] | None = None
    for size in range(32, 1025, 16):
        if backbone == "gru":
            bb = cls(obs_dim=obs_dim, hidden=size, fusion=fusion, n_drives=n_drives)
        else:
            bb = cls(
                obs_dim=obs_dim,
                d_model=size,
                fusion=fusion,
                n_drives=n_drives,
                context=context,
            )
        count = bb.param_count()
        if best is None or abs(count - target) < abs(best[0] - target):
            best = (count, size)
        if abs(count - target) / target <= 0.05:
            return size
    assert best is not None
    return best[1]


for _name, _cls in BACKBONES.items():
    register("backbone", _name)(_cls)
