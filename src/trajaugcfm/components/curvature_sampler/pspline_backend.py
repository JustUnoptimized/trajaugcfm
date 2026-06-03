from .base import FPCABackend

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq
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
        sobolev_weight: float | None,
        smoothing_lambda: float | None,
        edf_lambda: float | None,
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
        self._sobolev_weight = sobolev_weight
        assert smoothing_lambda is None or edf_lambda is None, \
            'Cannot set both smoothing lambda and edf lambda together.'
        self._smoothing_lambda = smoothing_lambda
        self._edf_lambda = edf_lambda
        self._use_penalty = smoothing_lambda is not None or edf_lambda is not None
        self._p_order = p_order

        if edf_lambda is not None:
            assert p_order < edf_lambda and edf_lambda < (n_basis - 2), \
                f'edf lambda must be in ({p_order}, {n_basis-2}) but got {edf_lambda}'

        # Set by fit()
        self.A = None
        self.B = None
        self.B_prime = None
        self.C_mean = None
        self.G0 = None
        self.G1 = None
        self.G1_scale = None
        # if using smoothing penalty, the unspecified {smoothing lambda, edf lambda}
        # will be computed and set during fit()

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
        BBT = (B @ B.T)

        # use penalized B-Spline
        if self.use_penalty:
            if self.verbose:
                print(f'Using order {self.p_order} smoothing penalty')
            # Pen = \int D_p B(t) D_p B(t)^\top dt
            Pen = (
                L2Regularization(LinearDifferentialOperator(self.p_order))
                .penalty_matrix(bsb)[1:-1, 1:-1]
            )
            # Pen = reg.penalty_matrix(bsb)[1:-1, 1:-1]
            # ridge regularization. for eigh stability
            w = sla.eigvalsh(BBT, Pen + 1e-6 * np.eye(Pen.shape[0]))
            if self.smoothing_lambda is not None:
                # use the provided lambda directly
                self._edf_lambda = self._compute_edf(w, self.smoothing_lambda)
            elif self.edf_lambda is not None:
                # search for lambda corresponding to edf in log space
                sl_log, rootres = brentq(
                    lambda x: self._compute_edf(w, np.exp(x)) - self.edf_lambda,
                    np.log(1e-8),
                    np.log(1e8),
                    full_output=True,
                )
                assert rootres.converged, 'Rootfinding did not converge'
                self._smoothing_lambda = np.exp(sl_log)
            Pen *= self.smoothing_lambda
        else:
            # use ordinary B-spline
            Pen = 0

        BBTplusPen_inv_B = np.linalg.solve(BBT + Pen, B)
        C = BBTplusPen_inv_B[None, ...] @ fs
        C_mean = C.mean(axis=0, keepdims=True)
        C_centered = C - C_mean
        G0 = bsb.gram_matrix()[1:-1, 1:-1]
        if self._sobolev_weight is not None:
            # G1 = \int \dot{B}(t) \dot{B}(t)^\top dt
            G1 = (
                L2Regularization(LinearDifferentialOperator(1))
                .penalty_matrix(bsb)[1:-1, 1:-1]
            )
            G1_scale = np.trace(G0) / np.trace(G1)
            G_basis = G0 + (self.sobolev_weight * G1_scale * G1)
        else:
            G1 = None
            G1_scale = None
            G_basis = G0

        # Centering and G_sample = XX^T
        if self.verbose:
            print('Computing sample Gram matrix.')
        # Defensive symmetrization in case of numerical error
        G_basis = (G_basis + G_basis.T) / 2
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
            # Use variance of resids between raw trajs and unpenalized projection
            # as estimate of noise variance.
            # Heuristic is that unfittable deviations = noise.
            if self.use_penalty:
                # refit coeffs C for best ambient space B-spline projection
                Cproj = np.linalg.solve(BBT, B)[None, ...] @ fs
            else:
                # current coeffs are best ambient space B-spline projection
                Cproj = C
            fs_proj = B.T[None, ...] @ Cproj
            # compute noise variance; ddof = 0 because MP assumes population stats
            sigma2 = (fs - fs_proj).var()
            m = self._mp_thresh_cutoff(Sigma * bessel / R, sigma2, R, T*do)
        else:
            # should never see this
            raise ValueError(f'mode_cutoff_strat must be "var" or "mp" but found {self.mode_cutoff_strat}')
        m = self._mode_cutoff_bound(m)
        if self.verbose:
            print(f'Using m = {m} modes')

        # TODO: if using logging module, log a warning and then silently clip?
        assert Sigma[:m].min() > 0, 'Found non-positive eigvals of G_sample.'

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
        self.G0 = torch.tensor(G0, dtype=torch.float32, device=self.device)
        self.G1 = None if G1 is None else torch.tensor(G1, dtype=torch.float32, device=self.device)
        self.G1_scale = G1_scale

        # TODO: only for snapping t. Remove if manual cubic spline implemented later
        self.T = T
        return Lambdas

    def _query(
        self,
        Lambdas_i: Float32[Tensor, 'B ns M'],
        t: Float32[Tensor, ' T'],
    ) -> tuple[Float32[Tensor, 'B ns T do'], Float32[Tensor, 'B ns T do']]:
        # TODO: update later to full torch B-spline impl?
        # currently use lerp to bridge between T grid on B-spline values
        pos = t * (self.T - 1)
        lo = torch.floor(pos).long().clamp(0, self.T - 2)
        hi = lo + 1
        t_frac = (pos - lo.to(pos.dtype)).clamp(0, 1)[None, :]
        B_t = torch.lerp(self.B[:, lo], self.B[:, hi], t_frac)
        B_prime_t = torch.lerp(self.B_prime[:, lo], self.B_prime[:, hi], t_frac)
        coeffs = torch.einsum('isj,jpd->ispd', Lambdas_i, self.A)
        coeffs += self.C_mean[:, None, ...]  # un-center
        f = torch.einsum('pt,ispd->istd', B_t, coeffs)
        f_prime = torch.einsum('pt,ispd->istd', B_prime_t, coeffs)
        return f, f_prime

    def _compute_edf(self, w: Float[ndarray, ' j'], lmda: float) -> float:
        """Computes edf(lambda) = tr([BB.T + lambda*Pen].inv BB.T)

        w_j is gamma_j / r_j where gamma_j, r_j are respectively the
        eigenvalues of BB.T and Pen.
        """
        return (w / (w + lmda)).sum()


    @property
    def n_basis(self) -> int:
        return self._n_basis

    @property
    def n_basis_eff(self) -> int:
        return self._n_basis_eff

    @property
    def sobolev_weight(self) -> float | None:
        return self._sobolev_weight

    @property
    def smoothing_lambda(self) -> float | None:
        return self._smoothing_lambda

    @property
    def edf_lambda(self) -> float | None:
        return self._edf_lambda

    @property
    def use_penalty(self) -> bool:
        return self._use_penalty

    @property
    def p_order(self) -> int:
        return self._p_order

