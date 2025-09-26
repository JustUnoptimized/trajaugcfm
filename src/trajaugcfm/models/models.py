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

