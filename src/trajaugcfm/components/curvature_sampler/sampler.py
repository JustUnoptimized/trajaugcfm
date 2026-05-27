from .base import FPCABackend

import numpy as np
from scipy.spatial.distance import cdist
import torch

# Typing Imports
from typing import Literal
from numpy import ndarray
from torch import Generator, Tensor
from jaxtyping import (
    Float,
    Float32,
    Int64,
)


class ConditionalResidualSampler:
    def __init__(
        self,
        backend: FPCABackend,
        neighbors: int,
        gen: Generator,
        device: Literal['cpu', 'cuda'],
        verbose: bool,
    ) -> None:
        self.backend = backend
        self.neighbors = neighbors
        self.gen = gen
        self.device = device
        self.verbose = verbose

        # Set by fit()
        self.Lambdas = None
        self.local_sigma = None
        self._n_modes = None
        self.fitted = False

    def _knn_var(
        self,
        Lambdas: Float[ndarray, 'N M'],
    ) -> Float[ndarray, 'N M']:
        '''Compute local Lambda_i variance.'''
        if self.verbose:
            print(f'Computing Var(Lambda) based on k = {self.neighbors} nearest neighbors.')
        C = cdist(Lambdas, Lambdas, metric='sqeuclidean')
        np.fill_diagonal(C, np.inf)  # do not include self as nearest neighbor
        nearest_idxs = C.argpartition(self.neighbors, axis=1)[:, :self.neighbors]
        neighborhoods = Lambdas[nearest_idxs]  # (N knn M)
        local_sigma2 = neighborhoods.var(axis=1, ddof=1)
        assert local_sigma2.shape == Lambdas.shape  # sanity check
        return local_sigma2

    def fit(self, fs: Float[ndarray, 'R T do']) -> None:
        # Operate on numpy ndarrays
        Lambdas = self.backend.fit(fs)
        local_sigma = np.sqrt(self._knn_var(Lambdas))

        self._N, self._n_modes = Lambdas.shape

        # Downcast to torch fp32 and move to gpu
        self.Lambdas = torch.tensor(Lambdas, dtype=torch.float32, device=self.device)
        self.local_sigma = torch.tensor(local_sigma, dtype=torch.float32, device=self.device)

        # unsqueeze dim 1 for vectorized batch sampling
        self.Lambdas = self.Lambdas.unsqueeze(1)  # (R, 1, m)
        self.local_sigma = self.local_sigma.unsqueeze(1)  # (R, 1, m)

        self.fitted = True

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError('Call fit() before reconstruct() or sample().')

    def reconstruct(
        self,
        idxs: Int64[Tensor, ' B'],
        t: Float32[Tensor, ' T'],
    ) -> tuple[Float32[Tensor, 'B T do'], Float32[Tensor, 'B T do']]:
        '''Reconstruct curves and derivatives at times t.'''
        self._check_fitted()
        f, f_prime = self.backend._query(self.Lambdas[idxs], t)
        # squeeze the sample batch dimension
        return f.squeeze(1), f_prime.squeeze(1)

    def sample(
        self,
        idxs: Int64[Tensor, ' B'],
        ns: int,
        t: Float32[Tensor, ' T'],
    ) -> tuple[Float32[Tensor, 'B ns T do'], Float32[Tensor, 'B ns T do']]:
        '''Samples curves and derivatives at times t conditional on idxs.'''
        self._check_fitted()
        Lambdas_i = self.Lambdas[idxs]  # (B, 1, m)
        local_sigma_i = self.local_sigma[idxs]  # (B, 1, m)

        # sample from N(Lambdas_i, local_sigma2_i)
        eps_shape = (idxs.shape[0], ns, self.n_modes)  # (B, ns, m)
        eps = torch.randn(eps_shape, generator=self.gen, device=self.device)
        hat_Lambdas_i = (local_sigma_i * eps) + Lambdas_i  # broadcast over samples

        return self.backend._query(hat_Lambdas_i, t)

    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def N(self) -> int:
        return self._N
