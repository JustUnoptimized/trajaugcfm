from .base import FPCABackend

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import torch

# Typing imports
from typing import Literal
from numpy import ndarray
from torch import Tensor
from jaxtyping import Float, Float32


class SVDBackend(FPCABackend):
    """SVD implementation of FPCA on uniform time grid trajectories.
    Trajectories are smoothed using a Savitzky-Golay Filter.
    """

    def __init__(
        self,
        # SVDBackend Args
        window: int,
        # Parent Args
        mode_cutoff_strat: Literal['var', 'mp'],
        var_thresh: float,
        m_max: int | None,
        device: Literal['cpu', 'cuda'],
        verbose: bool,
    ) -> None:
        super().__init__(
            mode_cutoff_strat,
            var_thresh,
            m_max,
            device,
            verbose,
        )
        self._window = window  # savgol window

        # set by fit()
        self.fs_sg_mean = None
        self.fs_sg_mean_prime = None
        self.modes = None
        self.modes_prime = None

    def fit(self, fs: Float[ndarray, 'R T do']) -> Float[ndarray, 'R M']:
        """Fits SVD FPCA."""
        if self.verbose:
            print('Fitting SVD FPCA...')

        # Smoothing
        if self.verbose:
            print(f'Smoothing with SavGol window = {self.window}')
        R, T, do = fs.shape
        Tspan = np.linspace(0, 1, T)
        fs_sg = savgol_filter(
            fs,
            self.window,
            polyorder=3,  # cubic
            axis=1,
            mode='constant',
            cval=0.0,
        )

        # Centering and SVD
        if self.verbose:
            print('Computing SVD. May take a few minutes...')
        fs_sg_mean = fs_sg.mean(axis=0, keepdims=True)
        fs_sg_centered = fs_sg - fs_sg_mean
        U, S, VT = np.linalg.svd(
            fs_sg_centered.reshape((R, T*do)), full_matrices=False
        )
        eigvals = S ** 2

        # Cutoff number of modes
        if self.mode_cutoff_strat == 'var':
            bessel = R - 1
            m = self._var_expl_cutoff(eigvals / bessel)
        elif self.mode_cutoff_strat == 'mp':
            sigma2 = (fs - fs_sg).var()  # noise variance
            m = self._mp_thresh_cutoff(eigvals / R, sigma2, R, T*do)
        else:
            # should never see this
            raise ValueError(f'mode_cutoff_strat must be "var" or "mp" but found {self.mode_cutoff_strat}')
        m = self._mode_cutoff_bound(m)
        if self.verbose:
            print(f'Using m = {m} modes')

        # Get scores and eigenmodes
        Lambdas = U[:, :m] * S[None, :m]
        modes = VT.reshape((-1, T, do))[:m]
        modes_spline = CubicSpline(Tspan, modes, axis=1, bc_type='not-a-knot')
        modes_prime = modes_spline(Tspan, nu=1)
        assert modes.shape == modes_prime.shape  # sanity check

        # Store necessary fp32 tensors on device for reconstruction
        self.fs_sg_mean = torch.tensor(fs_sg_mean, dtype=torch.float32, device=self.device)
        fs_sg_mean_spline = CubicSpline(Tspan, fs_sg_mean, axis=1, bc_type='not-a-knot')
        fs_sg_mean_prime = fs_sg_mean_spline(Tspan, nu=1)
        self.fs_sg_mean_prime = torch.tensor(fs_sg_mean_prime, dtype=torch.float32, device=self.device)
        self.modes = torch.tensor(modes, dtype=torch.float32, device=self.device)
        self.modes_prime = torch.tensor(modes_prime, dtype=torch.float32, device=self.device)

        # TODO: only for snapping t. Remove if manual cubic spline implemented later
        self.T = T
        return Lambdas

    def _query(
        self,
        Lambdas_i: Float32[Tensor, 'B ns M'],
        t: Float32[Tensor, ' T'],
    ) -> tuple[Float32[Tensor, 'B ns T do'], Float32[Tensor, 'B ns T do']]:
        # TODO: update later to full torch impl?
        # currently snap t to nearest computed gridpoint
        t_idx = torch.round(t * (self.T - 1)).long().clamp(0, self.T - 1).to(self.device)
        f = torch.einsum('isj,jtd->istd', Lambdas_i, self.modes[:, t_idx, :])
        f += self.fs_sg_mean[:, None, t_idx, :]  # un-center
        f_prime = torch.einsum('isj,jtd->istd', Lambdas_i, self.modes_prime[:, t_idx, :])
        f_prime += self.fs_sg_mean_prime[:, None, t_idx, :]  # un-center
        return f, f_prime

    @property
    def window(self) -> int:
        return self._window

