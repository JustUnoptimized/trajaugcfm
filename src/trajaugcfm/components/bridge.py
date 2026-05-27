from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor

from jaxtyping import Float32


class Bridge(ABC):
    def __init__(self, sigma: float, device: Literal['cpu', 'cuda']) -> None:
        self.sigma = torch.tensor(sigma, device=device)
        self.device = device

    @abstractmethod
    def compute_sigma_and_log_deriv(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> tuple[Float32[Tensor, '#nt'], Float32[Tensor, ' nt'] | None]:
        '''Compute sigma_t and sigma_t_prime / sigma_t.

        d/dt log(sigma(t)) = (d/dt sigma(t)) / sigma(t).

        For the ConstantBridge, d/dt log(sigma(t)) = 0 so
        return None to signify skipping multiplication in
        regression signal ut = d/dt log(sigma(t)) * (xt - mut) + mutprime.
        '''
        ...


class ConstantBridge(Bridge):
    def __init__(self, sigma: float, device: Literal['cpu', 'cuda']) -> None:
        super().__init__(sigma, device)

    def compute_sigma_and_log_deriv(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> tuple[Float32[Tensor, ' 1'], None]:
        return self.sigma[None], None


class SchrodingerBridge(Bridge):
    def __init__(self, sigma: float, device: Literal['cpu', 'cuda']) -> None:
        super().__init__(sigma, device)

    def compute_sigma_and_log_deriv(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> tuple[Float32[Tensor, ' nt'], Float32[Tensor, ' nt']]:
        sqrt_ts = torch.sqrt(ts * (1 - ts))
        sigma_t = self.sigma * sqrt_ts
        log_deriv = self.sigma * (1 - (2 * ts)) / (2 * sqrt_ts)
        return sigma_t, log_deriv


if __name__ == "__main__":
    device = 'cpu'
    sigma = 2.0
    CB = ConstantBridge(sigma, device)
    SB = SchrodingerBridge(sigma, device)

    ts = torch.rand(5)
    CB_sigma_t, CB_log_deriv = CB.compute_sigma_and_log_deriv(ts)
    SB_sigma_t, SB_log_deriv = SB.compute_sigma_and_log_deriv(ts)

    print(CB_sigma_t, CB_log_deriv)
    print(SB_sigma_t, SB_log_deriv)
