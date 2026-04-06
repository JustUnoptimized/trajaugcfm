import itertools

import jaxtyping as jt
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        d_in:  int,
        d_out: int,
        w:     int=64,
        h:     int=2
    ) -> None:
        super().__init__()
        act = nn.SELU()  ## will get registered inside nn.Sequential()
        linears = (nn.Linear(w, w) for _ in range(h))
        self.net = nn.Sequential(
            nn.Linear(d_in, w),
            act,
            *itertools.chain.from_iterable(zip(linears, itertools.repeat(act))),
            nn.Linear(w, d_out)
        )

    def forward(
        self,
        x: jt.Real[torch.Tensor, '*batch d_in']
    ) -> jt.Real[torch.Tensor, '*batch d_out']:
        return self.net(x)


class FlowScoreMLP(nn.Module):
    def __init__(
        self,
        d_in:  int,
        d_out: int,
        w:     int=64,
        h:     int=2
    ) -> None:
        super().__init__()
        act = nn.SELU()
        linears = (nn.Linear(w, w) for _ in range(h))
        self.trunk = nn.Sequential(
            nn.Linear(d_in, w),
            act,
            *itertools.chain.from_iterable(zip(linears, itertools.repeat(act)))
        )
        self.flow_head = nn.Linear(w, d_out)
        self.score_head = nn.Linear(w, d_out)

    def forward(
        self,
        x: jt.Real[torch.Tensor, '*batch d_in']
    ) -> tuple[jt.Real[torch.Tensor, '*batch d_out'], jt.Real[torch.Tensor, '*batch d_out']]:
        z = self.trunk(x)
        return self.flow_head(z), self.score_head(z)


class AdaLNBlock(nn.Module):
    """Single residual block with Adaptive Layer Normalization (FiLM-style).

    Time embedding generates per-layer scale (gamma) and shift (beta) that
    *multiply* the hidden features instead of being concatenated. This gives
    the network a hardware gate to amplify/suppress channels at specific
    times without consuming spatial capacity.
    """
    def __init__(self, width: int, d_time: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.act = nn.SELU()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        # Project time embedding → (scale, shift) per hidden unit
        self.time_proj = nn.Sequential(
            nn.SELU(),
            nn.Linear(d_time, width * 2),
        )
        # AdaLN-Zero: init gating to identity so untrained model is pass-through
        nn.init.zeros_(self.time_proj[1].weight)
        nn.init.zeros_(self.time_proj[1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear(x))
        h = self.norm(h)
        gamma, beta = self.time_proj(t_emb).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class AdaLNMLP(nn.Module):
    """Velocity MLP with multiplicative time gating via AdaLN-Zero.

    Drop-in replacement for MLP: receives concatenated [time_emb, state] input,
    internally splits them, and uses AdaLN blocks where time modulates the
    hidden state multiplicatively instead of additively.
    """
    def __init__(
        self,
        d_in:  int,
        d_out: int,
        w:     int = 512,
        h:     int = 4,
        d_time: int = 6,
    ) -> None:
        super().__init__()
        self.d_time = d_time
        d_state = d_in - d_time
        self.proj_in = nn.Linear(d_state, w)
        self.blocks = nn.ModuleList([AdaLNBlock(w, d_time) for _ in range(h)])
        self.proj_out = nn.Linear(w, d_out)

    def forward(
        self,
        x: jt.Real[torch.Tensor, '*batch d_in']
    ) -> jt.Real[torch.Tensor, '*batch d_out']:
        t_emb = x[..., :self.d_time]
        x_state = x[..., self.d_time:]
        h = self.proj_in(x_state)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.proj_out(h)


class flowscore_wrapper(nn.Module):
    '''Wrapper to convert single-head model to double-head model'''

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: jt.Real[torch.Tensor, '*batch d_in']
    ) -> tuple[jt.Real[torch.Tensor, '*batch d_out'], None]:
        return self.model(x), None

