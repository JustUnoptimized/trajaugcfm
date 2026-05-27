from .base import FPCABackend

import numpy as np
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis
import torch

# Typing imports
from typing import Literal
from numpy import ndarray
from torch import Tensor
from jaxtyping import Float, Float32


class PSplineBackend(FPCABackend):
    """P-Spline basis FPCA."""

    def __init__(
        self,
        # BSplineBackend Args
        n_basis: int,
        smoothing_lambda: float | None,
        p_order: int,
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
        self._n_basis = n_basis
        self._n_basis_eff = n_basis - 2  # only use middle basis
        self._smoothing_lambda = smoothing_lambda
        self._p_order = p_order

        # Set by fit()
        self.A = None
        self.B = None
        self.B_prime = None
        self.C_mean = None

    def fit(self, fs: Float[ndarray, 'R T do']) -> Float[ndarray, 'R M']:
        if self.verbose:
            print('Fitting B-spline FPCA...')

        R, T, do = fs.shape
        Tspan = np.linspace(0, 1, T)

        # B-Spine basis
        if self.verbose:
            print('Computing B-spline basis coefficients.')
        bsb = BSplineBasis(
            domain_range=(0., 1.),
            n_basis=self.n_basis,
            order=4,  # fix to cubic
        )
        B = bsb(Tspan)[1:-1, :, 0]
        if self.smoothing_lambda is not None:
            # use penalized B-Spline
            if self.verbose:
                print(f'Order {self.p_order} smoothing penalty with lambda = {self.smoothing_lambda:.4f}')
            reg = L2Regularization(LinearDifferentialOperator(self.p_order))
            Pen = self.smoothing_lambda * reg.penalty_matrix(bsb)[1:-1, 1:-1]
        else:
            # use ordinary B-Spline
            Pen = 0
        BBTplusPen = (B @ B.T) + Pen
        BBTplusPen_inv_B = np.linalg.solve(BBTplusPen, B)
        C = BBTplusPen_inv_B[None, ...] @ fs
        C_mean = C.mean(axis=0, keepdims=True)
        C_centered = C - C_mean
        G_basis = bsb.gram_matrix()[1:-1, 1:-1]

        # Centering and G_sample = XX^T
        if self.verbose:
            print('Computing sample Gram matrix.')
        L = np.linalg.cholesky(G_basis)
        D = L.T[None, ...] @ C_centered
        D_flat = D.reshape((R, -1))
        G_sample = D_flat @ D_flat.T
        bessel = R - 1
        G_sample /= bessel

        # U @ Sigma @ U^T = XX^T
        if self.verbose:
            print('Computing eigendecomposition. May take a few minutes...')
        Sigma, U = np.linalg.eigh(G_sample)
        # reorder by decreasing eigenvalue
        Sigma = np.flip(Sigma, axis=0)
        U = np.flip(U, axis=1)

        # Cutoff number of modes
        if self.mode_cutoff_strat == 'var':
            m = self._var_expl_cutoff(Sigma)
        elif self.mode_cutoff_strat == 'mp':
            fs_bsb = B.T[None, ...] @ C
            sigma2 = (fs - fs_bsb).var()  # noise variance
            m = self._mp_thresh_cutoff(Sigma * bessel / R, sigma2, R, T*do)
        else:
            # should never see this
            raise ValueError(f'mode_cutoff_strat must be "var" or "mp" but found {self.mode_cutoff_strat}')
        m = self._mode_cutoff_bound(m)
        if self.verbose:
            print(f'Using m = {m} modes')

        # Solve for scores and eigenmodes V^T = S^{-1} U^T X
        if self.verbose:
            print('Solve for eigenmodes')
        S = np.sqrt(Sigma[:m])
        Lambdas = U[:, :m] * S[None, :]
        A = np.einsum('j,ij,ipd->jpd', 1/S, U[:, :m], C_centered)

        # Store necessary tensors for reconstruction
        self.A = torch.tensor(A, dtype=torch.float32, device=self.device)
        self.B = torch.tensor(B, dtype=torch.float32, device=self.device)
        B_prime = bsb.derivative()(Tspan)[1:-1, :, 0]
        self.B_prime = torch.tensor(B_prime, dtype=torch.float32, device=self.device)
        self.C_mean = torch.tensor(C_mean, dtype=torch.float32, device=self.device)

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
        coeffs = torch.einsum('isj,jpd->ispd', Lambdas_i, self.A)
        coeffs += self.C_mean[:, None, ...]  # un-center
        f = torch.einsum('pt,ispd->istd', self.B[:, t_idx], coeffs)
        f_prime = torch.einsum('pt,ispd->istd', self.B_prime[:, t_idx], coeffs)
        return f, f_prime

    @property
    def n_basis(self) -> int:
        return self._n_basis

    @property
    def n_basis_eff(self) -> int:
        return self._n_basis_eff

    @property
    def smoothing_lambda(self) -> float | None:
        return self._smoothing_lambda

    @property
    def p_order(self) -> float:
        return self._p_order

