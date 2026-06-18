from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Literal

import torch
from torch import Tensor

from jaxtyping import Float32

class Bridge(ABC):
    def __init__(self, sigma: float, device: Literal['cpu', 'cuda']) -> None:
        self.sigma = torch.tensor(sigma, device=device)
        self.device = device
        self._valuename = IntEnum(
            'Value',
            [('SIGMA', 1), ('SIGMAPRIME', 2), ('BOTH', 3)],
        )

    @property
    def valuename(self) -> IntEnum:
        return self._valuename

    @property
    @abstractmethod
    def constant_sigma(self) -> bool:
        ...

    def compute(
        self,
        value_enum: int,
        ts: Float32[Tensor, ' nt'],
    ) -> Float32[Tensor, ' #nt'] | tuple[Float32[Tensor, ' #nt'], Float32[Tensor, ' #nt']]:
        if value_enum == self.valuename.SIGMA:
            out = self.compute_sigma(ts)
        elif value_enum == self.valuename.SIGMAPRIME:
            out = self.compute_sigma_prime(ts)
        elif value_enum == self.valuename.BOTH:
            out = self.compute_sigma_and_sigma_prime(ts)
        else:
            raise ValueError('value_enum must be SIGMA, SIGMAPRIME, or BOTH')
        return out

    @abstractmethod
    def compute_sigma(
        self,
        ts: Float32[Tensor, ' nt'],
    ):
        """Compute sigma_t."""
        ...

    @abstractmethod
    def compute_sigma_prime(
        self,
        ts: Float32[Tensor, ' nt'],
    ):
        """Compute sigma_t_prime."""
        ...

    @abstractmethod
    def compute_sigma_and_sigma_prime(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> tuple[Float32[Tensor, '#nt'], Float32[Tensor, ' #nt']]:
        """Compute sigma_t and sigma_t_prime.

        d log(sigma(t)) / dt = [d log(sigma(t)) / d sigma(t)] * [d sigma(t) / dt]
        d log(sigma(t)) / d sigma(t) = 1 / sigma(t)
        """
        ...

    @abstractmethod
    def compute_lambda(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> Float32[Tensor, ' #nt']:
        ...


class ConstantBridge(Bridge):
    def __init__(self, sigma: float, device: Literal['cpu', 'cuda']) -> None:
        super().__init__(sigma, device)
        self._zero = torch.tensor(0., device=device)

    @property
    def constant_sigma(self) -> bool:
        return True

    def compute_sigma(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> Float32[Tensor, ' 1']:
        """Compute sigma_t.

        For the ConstantBridge:
        sigma(t) = sigma
        """
        return self.sigma[None]

    def compute_sigma_prime(
            self,
            ts: Float32[Tensor, ' nt'],
    ) -> Float32[Tensor, ' 1']:
        """Compute sigma_t_prime.

        For the ConstantBridge:
        sigma(t) = sigma
        d/dt sigma(t) = 0
        """
        return self._zero[None]

    def compute_sigma_and_sigma_prime(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> tuple[Float32[Tensor, ' 1'], Float32[Tensor, ' 1']]:
        """Compute sigma_t and sigma_t_prime.

        For the ConstantBridge:
        sigma(t) = sigma
        d/dt sigma(t) = 0
        """
        return self.sigma[None], self._zero[None]

    def compute_lambda(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> Float32[Tensor, ' 1']:
        del ts
        return 1 / self.sigma[None]


class SchrodingerBridge(Bridge):
    def __init__(
        self,
        sigma: float,
        device: Literal['cpu', 'cuda'],
        reg: float=1e-8,
    ) -> None:
        super().__init__(sigma, device)
        self.reg = torch.tensor(reg, device=device)

    @property
    def constant_sigma(self) -> bool:
        return False

    def compute_sigma(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> Float32[Tensor, ' nt']:
        """Compute sigma_t.

        For the SchrodingerBridge:
        sigma(t) = sigma * sqrt(t(1 - t))
        """
        return self.sigma * torch.sqrt(ts * (1 - ts))

    def compute_sigma_prime(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> Float32[Tensor, ' nt']:
        """Compute sigma_t_prime.

        For the SchrodingerBridge:
        sigma(t) = sigma * sqrt(t(1 - t))
        d/dt sigma(t) = sigma * (1 - 2t) / (2 * sqrt(t(1 - t)))
        """
        numer = self.sigma * (1 - (2 * ts))
        denom = (2 * torch.sqrt(ts * (1 - ts))) + self.reg
        return numer / denom

    def compute_sigma_and_sigma_prime(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> tuple[Float32[Tensor, ' nt'], Float32[Tensor, ' nt']]:
        """Compute sigma_t and sigma_t_prime.

        For the SchrodingerBridge:
        sigma(t) = sigma * sqrt(t(1 - t))
        d/dt sigma(t) = sigma * (1 - 2t) / (2 * sqrt(t(1 - t)))
        """
        sqrt_t1mt = torch.sqrt(ts * (1 - ts))
        sigma_t = self.sigma * sqrt_t1mt
        numer = self.sigma * (1 - (2 * ts))
        denom = (2 * sqrt_t1mt) + self.reg
        sigma_t_prime = numer / denom
        return sigma_t, sigma_t_prime

    def compute_lambda(
        self,
        ts: Float32[Tensor, ' nt'],
    ) -> Float32[Tensor, ' nt']:
        numer = 2 * torch.sqrt(ts * (1 - ts))
        denom = self.sigma
        return numer / denom



if __name__ == '__main__':
    device = 'cpu'
    sigma = 1.0
    # CB = ConstantBridge(sigma, device)
    SB = SchrodingerBridge(sigma, device)

    ts, _ = torch.sort(torch.rand(6))
    print(ts)
    # print(CB.compute(CB.valuename.BOTH, ts))
    print(SB.compute(SB.valuename.BOTH, ts))
    # CB_sigma_t, CB_log_deriv = CB.compute_sigma_and_log_deriv(ts)
    # SB_sigma_t, SB_log_deriv = SB.compute_sigma_and_log_deriv(ts)

    # print(CB_sigma_t, CB_log_deriv)
    # print(SB_sigma_t, SB_log_deriv)

