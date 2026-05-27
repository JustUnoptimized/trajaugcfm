from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from jaxtyping import Float32


# Warning: This module assumes that the input ts is on the
#          same device as the specified device. No checks
#          are made to ensure device matches.

class TimeRepr(ABC):
    def __init__(self, device: Literal['cpu', 'cuda']) -> None:
        self.device = device

    @abstractmethod
    def enrich(self, ts: Float32[Tensor, ' nt']) -> Float32[Tensor, 'nt tdims']:
        ...


class TimeRaw(TimeRepr):
    def __init__(self, device: Literal['cpu', 'cuda']) -> None:
        super().__init__(device)

    def enrich(self, ts: Float32[Tensor, ' nt']) -> Float32[Tensor, 'nt tdims']:
        return ts[:, None]


class TimeRFF(TimeRepr):
    def __init__(
        self,
        rffscale: float,
        rffdim: int,
        rffseed: int,
        device: Literal['cpu', 'cuda'],
    ) -> None:
        super().__init__(device)
        # Use NumPy to get B and convert to Torch Tensor
        self.rffscale = rffscale
        self.rffdim = rffdim
        self.rffseed = rffseed
        self.B = torch.from_numpy(
            np.random.default_rng(seed=rffseed)
            .normal(loc=0, scale=rffscale, size=(1, rffdim))
            .astype(np.float32)
        ).to(self.device)

    def enrich(self, ts: Float32[Tensor, ' nt']) -> Float32[Tensor, 'nt tdims']:
        Bt = self.B * ts[:, None]  # (nt, rffdim)
        cosBt = torch.cos(Bt)
        sinBt = torch.sin(Bt)
        return torch.cat((cosBt, sinBt), axis=1)


if __name__ == "__main__":
    device = 'cpu'
    raw = TimeRaw(device)

    rffscale = 1.0
    rffdim = 2
    rffseed = 100
    rff = TimeRFF(rffscale, rffdim, rffseed, device)

    ts = torch.rand(10)
    print(ts)

    rawts = raw.enrich(ts)
    rffts = rff.enrich(ts)

    print(rawts)
    print(rffts)
