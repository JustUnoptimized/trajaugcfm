from abc import ABC, abstractmethod

import numpy as np

# Typing imports
from typing import Literal
from numpy import ndarray
from torch import Tensor
from jaxtyping import Float, Float32


class FPCABackend(ABC):
    def __init__(
        self,
        mode_cutoff_strat: Literal['var', 'mp'],
        var_thresh: float,
        m_max: int | None,
        device: Literal['cpu', 'cuda'],
        verbose: bool,
    ) -> None:
        self.mode_cutoff_strat = mode_cutoff_strat
        self.var_thresh = var_thresh
        self._m_max = m_max
        self.device = device
        self.verbose = verbose

    def _var_expl_cutoff(
        self,
        eigvals: Float[ndarray, ' M'],
    ) -> int:
        '''Fraction of variance explained cutoff.'''
        if self.verbose:
            print(f'Truncating modes using {self.var_thresh:.2f} variance explained')
        frac_var_expl = eigvals / eigvals.sum()
        return (frac_var_expl.cumsum() <= self.var_thresh).sum() + 1

    def _mp_thresh_cutoff(
        self,
        eigvals: Float[ndarray, ' M'],
        sigma2: float,
        n: int,
        d: int,
    ) -> int:
        '''Marchenko-Pastur noise bulk threshold.'''
        if self.verbose:
            print('Truncating modes using Marchenko-Pastur threshold')
        mp_ratio = d / n
        lambda_plus = sigma2 * ((1 + np.sqrt(mp_ratio)) ** 2)
        return (eigvals >= lambda_plus).sum()

    def _mode_cutoff_bound(self, m: int) -> int:
        if self.m_max is not None:
            if self.verbose:
                print(f'Upper bounding number of modes to {self.m_max}')
            return min(m, self.m_max)
        else:
            return m

    @abstractmethod
    def fit(self, fs: Float[ndarray, 'R T do']) -> Float[ndarray, 'R M']:
        '''Fits FPCA internals for querying and sampling. Returns Lambdas.'''
        ...

    @abstractmethod
    def _query(
        self,
        Lambdas_i: Float32[Tensor, 'B ns M'],
        t: Float32[Tensor, ' T'],
    ) -> tuple[Float32[Tensor, 'B ns T do'], Float32[Tensor, 'B ns T do']]:
        '''Queries FPCA for curves and derivatives at times t.

        Requires sample batch dimension ns >= 1 to allow for
        vectorized sampling.
        '''
        ...

    @property
    def m_max(self) -> int:
        return self._m_max
