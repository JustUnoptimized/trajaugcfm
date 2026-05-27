from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor

from jaxtyping import (
    Bool,
    Float,
    Int,
)


class Interpolant(ABC):
    def __init__(self, device: Literal['cpu', 'cuda']) -> None:
        self.device = device

    @abstractmethod
    def compute_mu_t(
        self,
        a: Float[Tensor, '... d'],
        b: Float[Tensor, '... d'],
        ts: Float[Tensor, ' nt'],
        oidx: Bool[Tensor, ' do'],
        hidx: Bool[Tensor, ' dh'],
    ) -> Float[Tensor, '... nt d']:
        ...

    def _lerp(
        self,
        a: Float[Tensor, '... d'],
        b: Float[Tensor, '... d'],
        ts: Float[Tensor, ' nt'],
    ) -> Float[Tensor, '... nt d']:
        '''Computes linear interpolation from a to b at times ts.

        lerp = t * b + (1 - t) * a

        Endpoints a and b must have shape (*batch, d).
        Times ts must have shape (nt).
        Output has shape (*batch, nt, d).
        '''
        batch_dims = a.ndim - 1
        ts = ts.view((1,) * len(batch_dims) + (-1, 1))  # (*batch, nt, 1)
        return ts * b.unsqueeze(-2) + (1 - ts) * a.unsqueeze(2)  # (*batch, nt, d)


class LinearInterpolant(Interpolant):
    def __init__(self, device: Literal['cpu', 'cuda'], **kwargs) -> None:
        super().__init__(device)

    def compute_mu_t(
        self,
        a: Float[Tensor, '... d'],
        b: Float[Tensor, '... d'],
        ts: Float[Tensor, ' nt'],
        oidx: Bool[Tensor, ' do'],
        hidx: Bool[Tensor, ' dh'],
    ) -> Float[Tensor, '... nt d']:
        '''Interpolate from a to b using linear interpolation.

        oidx, hidx are ignored.
        '''
        del oidx, hidx
        return self._lerp(a, b, ts)


class CurvatureTransferInterpolant(Interpolant):
    def __init__(
        self,
        device: Literal['cpu', 'cuda'],
        refs_query: Literal['snap', 'interp']='interp',
        Delta_M_strategy: Literal['swap', 'shift', 'fpca']='fpca',
        refs: Float[ndarray, 'R T do'] | None=None,
        fpca_expl_var_thresh: float=0.9,
        **kwargs,
    ) -> None:
        super().__init__(device)
        BT_strategies = {
            'cpu': self._compute_BT_cpu,
            'cuda': self._compute_BT_cuda,
        }
        self._compute_BT = BT_strategies[device]
        self.refs_query = refs_query
        query_strategy = {
            'snap': self._grid_query_snap,
            'interp': self._grid_query_interp,
        }
        self._grid_query = query_strategy[refs_query]
        self.fpca_expl_var_thresh = fpca_expl_var_thresh
        Delta_M_strategies = {
            'swap': self._compute_Delta_M_o_swap,
            'shift': self._compute_Delta_M_o_shift,
            'fpca': self._compute_Delta_M_o_fpca,
        }
        self._compute_Delta_M_o = Delta_M_strategies[Delta_M_strategy]
        # the following three will be set by _fit_fpca()
        self.modes = None
        self.eigvals = None
        self.Lambdas = None
        if Delta_M_strategy == 'fpca':
            if refs is None:
                # this if statement should never be called
                # since the sampler should always pass in refs
                raise ValueError('refs required if Delta_M_strategy == "fpca"')
            # self._fit_fpca(refs, fpca_expl_var_thresh)

    def _grid_query_snap(
        self,
        refs: Float[Tensor, 'nr T do'],
        ts: Float[Tensor, ' nt'],
    ) -> Float[Tensor, 'nr nt do']:
        T = refs.shape[1]
        ts_grid_idxs = (ts * (T - 1)).round().long()
        return refs[:, ts_grid_idxs, :]

    def _grid_query_interp(
        self,
        refs: Float[Tensor, 'nr T do'],
        ts: Float[Tensor, ' nt'],
    ) -> Float[Tensor, 'nr nt do']:
        T = refs.shape[1]
        ts_scaled = ts * (T - 1)
        lo = ts_scaled.floor().long()
        hi = (lo + 1).clamp(max=T-1)
        w = (ts_scaled - lo.float()).unsqueeze(0).unsqueeze(1)  # floats (1, nt, 1)
        return (1 - w) * refs[:, lo, :] + w * refs[:, hi, :]

    def _compute_BT_cpu(
        self,
        a: Float[Tensor, '... k d'],
        b: Float[Tensor, '... k d'],
        oidx : Bool[Tensor, ' do'],
        hidx : Bool[Tensor, ' dh'],
    ) -> Float[Tensor, '... do dh']:
        '''Computes B using least squares.

        [ B @ linslope_M_o = linslope_M_h ]
        but our data contains (..., k, d) which means we have
        [ linslope_M_o.mT @ B.mT = linslope_M_h.mT ]

        returns B.mT
        '''
        linslope_M = b - a
        res = torch.linalg.lstsq(linslope_M[..., oidx], linslope_M[..., hidx])
        return res.solution.mT

    def _compute_BT_cuda(
        self,
        a: Float[Tensor, '... k d'],
        b: Float[Tensor, '... k d'],
        oidx : Bool[Tensor, ' do'],
        hidx : Bool[Tensor, ' dh'],
    ) -> Float[Tensor, '... do dh']:
        '''Computes B using the pinv.

        The cuda driver for torch.linalg.lstsq() assumes full-rank linslope_M_o
        but we cannot make that assumption.

        B.mT = pinv(linslope_M_o.mT) @ linslope_M_h

        returns B.mT
        '''
        linslope_M = b - a
        return torch.linalg.pinv(linslope_M[..., oidx]) @ linslope_M[..., hidx].mT

    def _compute_Delta_M_o_swap(
        self,
        a: Float[Tensor, 'nr k d'],
        b: Float[Tensor, 'nr k d'],
        ts: Float[Tensor, ' nt'],
        oidx : Bool[Tensor, ' do'],
        refs : Float[Tensor, 'nr T do'],
        T : Float[Tensor, ' T'],
    ) -> Float[Tensor, 'nr k T do']:
        '''Compute Delta M_o by swapping mu_o(t) with x_o(t).

        We need the curve x_o(t) with endpoints x_o(0), x_o(1) but
        that is missing information. Sidestep the issue by replacing
        x_o(t) with mu_o(t) where mu_o(t) is the reference trajectory.

        This introduces a mismatch where Delta M_o(t) != x_o(t) at t = 0, 1.
        '''
        del ts
        aobs = a[..., oidx]
        bobs = b[..., oidx]
        abobs_lerp = self._lerp(aobs, bobs, T)  # (nr, k, T, do)
        return refs.unsqueeze(1) - abobs_lerp

    def _compute_Delta_M_o_shift(
        self,
        a: Float[Tensor, 'nr k d'],
        b: Float[Tensor, 'nr k d'],
        ts: Float[Tensor, ' nt'],
        oidx : Bool[Tensor, ' do'],
        refs : Float[Tensor, 'nr T do'],
        T : Float[Tensor, ' T'],
    ) -> Float[Tensor, 'nr 1 T do']:
        '''Compute Delta M_o by shifting deviations Delta mu_o(t) onto xbar_o(t).

        Using nu_o(t) = Delta mu_o(t) + xbar_o(t) shifts deviations
        from mubar_o(t) onto xbar_o(t) preserving the endpoint conditions.
        The resulting Delta N_o(t) is rank 1 at any fixed time t because
        the curvature residuals of nu_o(t) are now identical across
        endpoint pairs.

        The return shape (nr, 1, T, do) reflects the rank 1 structure
        via broadcasting.
        '''
        del a, b, ts, oidx
        refs_lerp = self._lerp(refs[:, 0], refs[:, -1], T)  # (nr, T, do)
        return (refs - refs_lerp).unsqueeze(1)

    def _compute_Delta_M_o_fpca(
        self,
        a: Float[Tensor, 'nr k d'],
        b: Float[Tensor, 'nr k d'],
        ts: Float[Tensor, ' nt'],
        oidx : Bool[Tensor, ' do'],
        refs : Float[Tensor, 'nr T do'],
        T : Float[Tensor, ' T'],
    ) -> Float[Tensor, 'nr 1 T do']:
        pass

    def _fit_fpca(
        self,
        refs: Float[ndarray, 'R T do'],
        var_thresh: float,
    ):
        '''Fit fpca on all curvature residuals of reference trajectories.

        Current implementation is a rough discretized version
        which essentially computes PCA on the flattened
        (R, T*do) matrix.

        Using specific basis functions (e.g. Fourier) may be
        added later.

        Will use finite difference for now for d/dt.
        '''
        print('compute lerps and residuals')
        R, T, do = refs.shape
        Tspan = np.linspace(0, 1, T)
        # self._lerp() requires torch.Tensor so recode it here
        Tspan_bc = Tspan.reshape((1, T, 1))  # for broadcasting
        refs_lerp = Tspan_bc * refs[:, [-1], :] + Tspan_bc * refs[:, [0], :]

        # Get curvature residuals and perform FPCA via SVD
        resid = refs - refs_lerp  # (R, T, do)
        resid_flat = resid.reshape(R, T*do)
        resid_flat_centered = resid_flat - resid_flat.mean(axis=0, keepdims=True)
        # U : (R, k)
        # S : (k,) for k = min(R, T*do). Usually R < T*do
        # VT: (k, T*do)
        U, S, VT = np.linalg.svd(resid_flat_centered, full_matrices=False)

        # Truncate to top variance modes and coefficient vectors Lambda
        eigvals = (S ** 2)
        frac_var_expl = eigvals / eigvals.sum()
        # add 1 to cutoff so total explained var exceeds var_thresh
        cutoff = (frac_var_expl.cumsum() < fpca_expl_var_thresh).sum() + 1
        modes = VT[:cutoff].reshape((cutoff, T, do))
        eigvals = eigvals[:cutoff]
        Lambdas = U[:, :cutoff] * S[None, :cutoff]

        self.modes = torch.tensor(modes, dtype=torch.float32, device=self.device)
        self.eigvals = torch.tensor(eigvals, dtype=torch.float32, device=self.device)
        self.Lambdas = torch.tensor(Lambdas, dtype=torch.float32, device=self.device)
        # Bessel correction is (R - 1)

    def _reconstruct_fpca(
        self,
        refidxs: Int[Tensor, ' nr'],
    ):
        pass

    def compute_mu_t(
        self,
        a: Float[Tensor, '... k d'],
        b: Float[Tensor, '... k d'],
        ts: Float[Tensor, ' nt'],
        oidx: Bool[Tensor, ' do'],
        hidx: Bool[Tensor, ' dh'],
    ) -> Float[Tensor, '... k nt d']:
        B = self._compute_BT(a, b, oidx, hidx)

