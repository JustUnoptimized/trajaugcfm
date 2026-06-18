from abc import ABC, abstractmethod

import torch
from torch import Generator, Tensor

# Typing imports
from typing import Literal
from jaxtyping import Bool, Float32, Int


class Coupler(ABC):
    def __init__(
        self,
        mask: Bool[Tensor, ' do'] | None,
        gen: Generator,
        device: Literal['cpu', 'cuda'],
    ) -> None:
        self.mask = mask
        self.gen = gen
        self.device = device

    @abstractmethod
    def sample(
        self,
        Xs: Float32[Tensor, 'N snaps d'],
        refs: Float32[Tensor, 'r T do'],
        ridxs: Int[Tensor, ' B'],
        Tsnaps: Int[Tensor, ' snaps'],
        k: int,
    ) -> Float32[Tensor, 'r k snaps d']:
        """Conditional on each ref, sample k endpoint pairs."""
        ...

# TODO: Make OTCoupler

class RBFCoupler(Coupler):
    def __init__(
        self,
        scale: float,
        mask: Bool[Tensor, ' do'] | None,
        gen: Generator,
        device: Literal['cpu', 'cuda'],
    ) -> None:
        super().__init__(mask, gen, device)
        self.scale = scale
        self.denom = -2 * (scale ** 2)

    def sample(
        self,
        Xs: Float32[Tensor, 'N snaps d'],
        refs: Float32[Tensor, 'B T do'],
        Tsnaps: Int[Tensor, ' snaps'],
        k: int,
    ) -> Float32[Tensor, 'B k snaps d']:
        # if d != do then select the features for similarity comparison
        if self.mask is not None:
            Xos = Xs[:, :, self.mask]
        else:
            Xos = Xs

        B = refs.shape[0]
        snaps = Tsnaps.shape[0]
        refs_snaps = refs[:, Tsnaps, :]  # (B, snaps, do)

        # Compute pairwise squared euclidean distances
        # || x - y ||^2 = || x ||^2 - 2<x, y> + || y ||^2
        Xos2 = (Xos ** 2).sum(dim=2, keepdim=True).transpose(1, 2)  # (N, 1, snaps)
        refs2 = (refs_snaps ** 2).sum(dim=2, keepdim=True).transpose(1, 2)  # (B, 1, snaps)
        inner = torch.einsum('nsi,rsi->rns', Xos, refs_snaps)  # (B, N, snaps)
        # Ds[i, j, s] contains ||xi - yj||^2 at snapshot s
        Ds = Xos2.transpose(0, 1) - (2 * inner) + refs2  # broadcast sum to (B, N, snaps)

        # RBF weight each distance and normalize using logsumexp trick
        logRBFs = Ds / self.denom
        logRBFs -= logRBFs.max(dim=1, keepdim=True)
        RBFs = torch.exp(logRBFs)  # (B, N, snaps)
        normconst = RBFs.sum(dim=1, keepdim=True)  # (B, 1, snaps)
        RBFs /= normconst

        # Inverse CDF sampling
        # First sample u ~ Unif(0, 1)
        # Then pick the first sample x where CDF(x) >= u
        RBFs_CDF = RBFs.cumsum(dim=1)
        u = torch.rand(
            (B, k, snaps),
            generator=self.gen,
            device=self.device
        )  # (B, k, snaps)
        z_idxs = torch.searchsorted(
            RBFs_CDF.transpose(1, 2).contiguous(),  # (B, snaps, N)
            u.transpose(1, 2).contiguous(),  # (B, snaps, k)
        ).transpose(1, 2)  # (B, k, snaps)
        z_idxs = z_idxs.clamp(max=Xs.shape[0]-1)  # z_idxs in [0, N-1]
        z = Xs[z_idxs, torch.arange(snaps)[None, None, :]]

        # sanity check after fancy indexing
        assert z.shape == (B, k, snaps, Xs.shape[-1])

        return z

