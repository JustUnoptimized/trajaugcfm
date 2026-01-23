from abc import abstractmethod
import os
from time import time
from typing import (
    Literal,
    overload,
    Self,
    TypedDict,
)

import jaxtyping as jt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import (
    cho_factor,
    cho_solve,
    issymmetric,
    solve_triangular,
)  # noqa: F401
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF as RBFKernel,
    WhiteKernel,
)
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from trajaugcfm.eigen_orient import orient_eigenvectors
from trajaugcfm.utils import (
    batch_eigval_replace_neg,
    batch_sqrtm,
    batch_interp,
    batch_inv,
    batch_inv_sqrtm,
    build_indexer,
    np_sigmoid,
    roundrobin_split_idxs,
)

## TYPING DECLARATIONS
## TODO: Double check all typing in annotations
type RBFK_Bounds = tuple[int | float, int | float] | Literal['fixed']
type Times = (
    jt.Float64[np.ndarray, 'nt rff_dims*2']
    | jt.Float64[np.ndarray, 'nt 1']
)
type Sigma_T = (
    jt.Float64[np.ndarray, 'k nt dims dims']
    | jt.Float64[np.ndarray, ' nt']
    | float
)
type Aux = (
    jt.Float64[np.ndarray, 'k dims dims']
    | jt.Float64[np.ndarray, ' nt']
    | None
)
type A_T_Prime_A_T_Inv = (
    jt.Float64[np.ndarray, 'k nt dims dims']
    | jt.Float[np.ndarray, ' nt']
    | None
)
type GCFMBatch = tuple[
    jt.Float32[Tensor, 'batch rff_dims*2'] | jt.Float32[Tensor, 'batch 1'],  ## ts
    jt.Float32[Tensor, 'batch dims'],                                        ## xt
    jt.Float32[Tensor, 'batch dims'],                                        ## ut
    jt.Float32[Tensor, 'batch dims'],                                        ## eps
    jt.Float32[Tensor, 'batch dims dims'] | None,                            ## lt
]
## nrefs = nbatches * k
## except when nrefs % k != 0 in which case
## nrefs = (nbatches - 1) * k + k_last
class Diagnostics(TypedDict):
    ts_history_all:            jt.Real[np.ndarray, 'nepochs nbatches nt']
    gain_history_all:          jt.Real[np.ndarray, 'nepochs nrefs nt']
    mu_correction_history_all: jt.Real[np.ndarray, 'nepochs nrefs b nt']
    eigvals_obs_history_all:   jt.Real[np.ndarray, 'nepochs nbatches nt obs']
    eigvecs_obs_history_all:   jt.Real[np.ndarray, 'nepochs nbatches nt obs obs']
    eigvals_hid_history_all:   jt.Real[np.ndarray, 'nepochs nrefs nt hid']
    eigvecs_hid_history_all:   jt.Real[np.ndarray, 'nepochs nrefs nt hid hid']
    batch_split_idxs:          jt.Int[np.ndarray, ' nbatches-1']


## TODO: Refactor to use protocols. Could maybe cleanup method signature boilerplate code
## TODO: How to parallellize? Lots of numpy matrix operations is slow for single thread...
class GCFMSamplerBase(IterableDataset):
    def __init__(
        self,
        prng:        np.random.Generator,
        Xs:          jt.Real[np.ndarray, 'N margidx dims'],
        Xrefs:       jt.Real[np.ndarray, 'Nrefs T obs'],
        obsmask:     list[bool],
        tidxs:       list[int],
        k:           int,
        n:           int,
        b:           int,
        nt:          int,
        delta:       float=0.01,
        rbfk_scale:  float=0.1,
        rbfk_bounds: RBFK_Bounds=(0.05, 5),
        whitenoise:  float=0.1,
        gpr_nt:      int=8,
        rbfd_scale:  float=1.,
        cc_impute:   bool=False,
        orient:      Literal['pre'] | Literal['post']='pre',
        reg:         float=1e-8,
        sigma:       float=1.0,
        sb_reg:      float=1e-8,
        tau:         float=1.,
        spectral:    Literal['maxgain'] | Literal['robust']='maxgain',
        beta_a:      float=2.0,
        rff_seed:    int=2000,
        rff_scale:   float=1.0,
        rff_dim:     int=1,
        diagnostics: bool=False
    ) -> None:
        '''Builds sampler for Guided Conditional Flow.

        Is a subclass of torch.utils.data.IterableDataset so can be
        passed into a torch.utils.data.DataLoader. This class already batches
        so the DataLoader must be created with the kwarg batch_size=None.

        Batch size is k * b * nt.
        Can be an iterable where a full iteration one cycle through the Xrefs
        Currently only implemented for augmentation via a GP regression
        with a sum kernel of RBF + White.

        Hyperparams for all mixins are precomputed on class init but
        are ignored during sampling depending on the chosen mixins.
        E.g. beta_a is saved but ignored if using the UniformTimeMixin.

        If using the TimeRFFMixin, standardize the random features across runs or validation
        by keeping the same rff_seed, rff_scale, and rff_dim.

        Most (all?) operations are vectorized.

        Args:
            prng:        NumPy Generator for reproducability
            Xs:          All snapshot data
            Xrefs:       All reference trajectories
            obsmask:     Mask to recover only the observed (reference) variables
            tidxs:       Time indices into Xrefs recovering the snapshot time points
            k:           Number of refs per batch
            n:           Number of samples per snapshot for weighted minibatch sampling
            b:           Minibatch size per ref
            nt:          Number of timepoints per sample in minibatch
            rbfk_scale:  Initial scale for GPR
            rbfk_bounds: Optimization bounds for GPR
            whitenoise:  Fixed white noise level for GPR
            gpr_nt:      Number of time points into ref used for GPR
            rbfd_scale:  Scale for RBF distance when resampling batch conditional on ref
            reg:         Regularization to prevent singular matrices
            sigma:       Sigma scaler for isotropic flow conditional prob. path
            sb_reg:      Regularizer to prevent small sigma_t for Schrodinger bridge
            beta_a:      Shape param. if using beta dist as time sampler
            rff_seed:    Seed for generating random frequencies
            rff_scale:   Scale for freq ~ N(0, rff_scale**2)
            rff_dim:     Number of rff dimensions for each cos and sin transform
        '''
        ## Reproducability
        self.prng = prng

        ## Data
        self.Xs = Xs
        self.Xrefs = Xrefs
        self.tidxs = tidxs

        ## Batch size along dimension
        self.b = b
        self.n = n
        self.k = k
        self.nt = nt

        ## Time delta for finite difference using (f(t+d/2) - f(t-d/2)) / d
        ## so need to scale t in [delta/2, 1-(delta/2)]
        self.delta_half = delta / 2
        self.delta = delta
        self.t_scale = 1 - self.delta

        ## Pre-compute some masks
        ## TODO: permute so var list is [obsvars, hidvars]?
        ##       could help with any regularization?
        self.obsmask = obsmask
        self.hidmask = ~obsmask
        self.obsobsmask = np.ix_(self.obsmask, self.obsmask)
        self.obshidmask = np.ix_(self.obsmask, self.hidmask)
        self.hidobsmask = np.ix_(self.hidmask, self.obsmask)
        self.hidhidmask = np.ix_(self.hidmask, self.hidmask)

        ## Dimensions
        self.nobs = int(self.obsmask.sum())
        self.nhid = int(self.hidmask.sum())
        self.dim = self.Xs.shape[-1]
        self.padlen = self.nhid - self.nobs

        ## Spline interpolator for Sigma_oot, used in anisotropic covariance
        self.Sigma_t_obs = self.SigmaInterpolator(Xrefs.shape[1], self.nobs, reg=reg)

        ## Time sampler params (Currently Beta only)
        self.beta_a = beta_a

        ## Time RFF enrichment fixed features (ignored if not enhancing time)
        self.B = np.random.default_rng(seed=rff_seed).normal(
            loc=0, scale=rff_scale, size=(1, rff_dim)
        )
        ## pre-scale by 2pi to avoid recomputation in _enrich_ts()
        self.B *= 2 * np.pi

        ## Variance schedule (IFMixins only)
        self.sigma = sigma

        ## Regularization
        self.reg = reg
        self.obsreg = np.eye(self.nobs)[None, None, ...] * self.reg
        self.hidreg = np.eye(self.nhid)[None, None, ...] * self.reg
        self.sb_reg = sb_reg

        ## Cross-Cov Imputation
        self.cc_impute = cc_impute
        self.preorient = orient == 'pre'
        if not self.preorient and orient != 'post':
            ## received neither pre nor post
            raise ValueError(
                f'argument orient must be "pre" or "post" but got {orient}'
            )
        ## Spectral Filtering
        self.tau = tau
        self.maxgain = spectral == 'maxgain'
        if not self.maxgain and spectral != 'robust':
            ## received neither maxgain nor robust
            raise ValueError(
                f'argument spectral must be "maxgain" or "robust" but got {spectral}'
            )

        ## RBF params for endpoint sampling
        self.rbfd_scale = rbfd_scale
        self.rbfd_denom = - 2 * (rbfd_scale ** 2)

        ## Gaussian Process Regression params
        self.rbfk_scale = rbfk_scale
        self.rbfk_bounds = rbfk_bounds
        self.whitenoise = whitenoise
        self.gpr_nt = gpr_nt
        ## TODO: change to be random intervals and recompute every epoch?
        self.gpr_ts_idxs = roundrobin_split_idxs(Xrefs.shape[1], gpr_nt)
        tspan = np.linspace(0, 1, num=Xrefs.shape[1]).reshape((-1, 1))
        self.gpr_ts = tspan[self.gpr_ts_idxs]
        self.gprs = self._precompute_gprs()

        ## Set up iterator state and compute len
        nrefs = Xrefs.shape[0]
        self._len, r = divmod(nrefs, k)
        if r > 0:
            self._len += 1
        self._sentinel = self._len - 1  ## needed for some indexing issues
        # self._iteridx = 0
        self._Xrefidxs = np.arange(nrefs).astype(int)

        ## For debugging/monitoring
        self.diagnostics = diagnostics
        if diagnostics:
            # full training logs
            self.ts_history_all = []
            self.gain_history_all = []
            self.mu_correction_history_all = []
            self.eigvals_obs_history_all = []
            self.eigvecs_obs_history_all = []
            self.eigvals_hid_history_all = []
            self.eigvecs_hid_history_all = []

            # training logs per epoch
            self.ts_history = np.zeros((self._len, nt))
            self.gain_history = np.zeros((nrefs, self.nt))
            self.mu_correction_history = np.zeros((nrefs, self.b, self.nt))
            self.eigvals_obs_history = np.zeros((self._len, self.nt, self.nobs))
            self.eigvecs_obs_history = np.zeros((self._len, self.nt, self.nobs, self.nobs))
            self.eigvals_hid_history = np.zeros((nrefs, self.nt, self.nhid))
            self.eigvecs_hid_history = np.zeros((nrefs, self.nt, self.nhid, self.nhid))
            self.batch_split_idxs = np.arange(k, nrefs, k, dtype=int)
            self.prev_batch_idx = -1  ## for bookkeeping

    class SigmaInterpolator:
        def __init__(
            self,
            T: int,
            nobs: int,
            reg: float
        ) -> None:
            '''Nested inner class to access interpolated Sigma as a function call'''
            self.nobs = nobs
            self.Lt_idxs = np.tril_indices(nobs)
            self.tspan = np.linspace(0, 1, T)
            self.reg = reg
            self.cs = None

        def fit(
            self,
            refs: jt.Real[np.ndarray, 'k T obs']
        ) -> None:
            covs_t = np.einsum('nti,ntj->tij', refs, refs)  ## (T, obs obs)
            covs_t /= refs.shape[0] - 1
            covs_t += np.eye(self.nobs)[None] * self.reg
            L_t = np.linalg.cholesky(covs_t)
            L_t_vals = L_t[:, *self.Lt_idxs]  ## (T, (obs+1)*obs/2)
            self.cs = CubicSpline(self.tspan, L_t_vals, axis=0, bc_type='natural')

        def __call__(
            self,
            ts: jt.Real[np.ndarray, ' nt'],
            derivative: bool=False,
            chol: bool=False
        ) -> jt.Real[np.ndarray, 'nt obs obs']:
            '''Compute Sigma_t or Sigma_t_prime

            We consider the Cholesky factorization Sigma_t = L_t @ L_t^T
            for a lower triangular L_t.

            If derivative == False
                then return Sigma_t = L_t @ L_t^T

            If derivative == True
                then return Sigma_t_prime = L_t_prime @ L_t^T
                                            + L_t @ L_t_prime^T

            If chol == True then return L_t or L_t_prime depending on derivative
            '''
            L_t = np.zeros((ts.shape[0], self.nobs, self.nobs))
            if derivative:
                if chol:
                    L_t[..., *self.Lt_idxs] = self.cs(ts, nu=1)
                    ## this is actually L_t_prime because ^^^^
                    ## Don't make new array L_t_prime to save memory
                    return L_t
                else:
                    ## Sigma = L @ L.T
                    ## Sigma_prime = L_prime @ L.T + L @ L_prime.T
                    L_t_prime = np.zeros_like(L_t)
                    L_t[..., *self.Lt_idxs] = self.cs(ts, nu=0)
                    L_t_prime[..., *self.Lt_idxs] = self.cs(ts, nu=1)
                    Sigma_t_prime = (L_t_prime @ L_t.swapaxes(-1, -2)) \
                                    + (L_t @ L_t_prime.swapaxes(-1, -2))
                    return Sigma_t_prime
            else:
                if chol:
                    L_t[..., *self.Lt_idxs] = self.cs(ts, nu=0)
                    return L_t
                else:
                    L_t[..., *self.Lt_idxs] = self.cs(ts, nu=0)
                    return L_t @ L_t.swapaxes(-1, -2)
            # return L_t if chol else L_t @ L_t.swapaxes(-1, -2)

    @classmethod
    def get_mixin_names(cls) -> list[str]:
        '''Return list of mixin class names'''
        bases = {object, GCFMSamplerBase}
        return [mixin.__name__ for mixin in cls.__bases__ if mixin not in bases]

    def _precompute_gprs(self) -> list[GaussianProcessRegressor]:
        '''Pre-compute GPRs on Xrefs

        Use RBFKernel + WhiteKernel to prevent small sigma near train times.
        '''
        gprs = [
            GaussianProcessRegressor(
                kernel=RBFKernel(
                    length_scale=self.rbfk_scale,
                    length_scale_bounds=self.rbfk_bounds
                )+WhiteKernel(
                    noise_level=self.whitenoise,
                    noise_level_bounds='fixed'
                ),
                copy_X_train=False
            ) for _ in range(self.Xrefs.shape[0])
        ]

        print('Pre-computing GPRs...')
        t0 = time()
        for i, xref in enumerate(self.Xrefs):
            gprs[i].fit(self.gpr_ts, xref[self.gpr_ts_idxs])
        t1 = time()
        print(f'Pre-computed GPRs in {t1-t0:.2f}s')

        return gprs

    def _get_xs_minibatch(self) -> jt.Real[np.ndarray, 'n margidx dims']:
        '''Sample minibatch w/o replacement from each marginal snapshot'''
        idxs = np.empty((self.n, self.Xs.shape[1]), dtype=int)
        for i in range(self.Xs.shape[1]):
            idxs[:, i] = self.prng.choice(self.Xs.shape[0], size=self.n, replace=False)
        return self.Xs[idxs, np.arange(self.Xs.shape[1])[None, :]]

    def _sample_z_given_refs(
        self,
        xs: jt.Real[np.ndarray, 'n margidx dims'],
        refs: jt.Real[np.ndarray, 'k T refdims']
    ) -> jt.Real[np.ndarray, 'k b margidx dims']:
        r'''Samples z = (x_0, ..., x_M) from prod_i^M pi(x_i | ref)

        Uses the RBF distance from ref as the unnormalized probabilities.
        Sampling is vectorized using a discretized version of inverse transform sampling.
        '''
        xsobs = xs[:, :, self.obsmask]
        k = refs.shape[0]  ## k < self.k possible for final batch
        RBFs = np.zeros((k, self.n, xs.shape[1]))

        ## Get prob tensor based on RBF dist
        for i in range(xs.shape[1]):
            ## cdist(metric=sqeuclidean) returns D where D[i, j] = ||x_i - x_j||^2
            RBFs[:, :, i] = cdist(
                refs[:, self.tidxs[i]], xsobs[:, i], metric='sqeuclidean'
            )
        RBFs /= self.rbfd_denom
        RBFs = np.exp(RBFs)
        normconst = np.sum(RBFs, axis=1, keepdims=True)  ## shape (k, 1, xs.shape[1])
        ## each RBF[i, :, j] should be a vector of probs into xs at snapshot j cond on ref i
        RBFs /= normconst

        ## for each ref and time marginal, sample endpoints z
        ## Batch sample using batched inverse transform sampling
        ## RBFs_cumsum[i, :, j] contains CDF vector for ref i, marginal j
        RBFs_cumsum = np.cumsum(RBFs, axis=1)                 ## compute CDF
        u = self.prng.random((k, self.b, xs.shape[1]))        ## sample u ~ Unif(0, 1)
        ## Compute CDF_inv by finding idxs where CDF > u and take first occasion
        comp = RBFs_cumsum[:, :, None, :] > u[:, None, :, :]  ## compute CDF_inv
        sample_idxs = np.argmax(comp, axis=1)                 ## (k, b, xs.shape[1])
        z = xs[sample_idxs, np.arange(xs.shape[1])[None, None, :]]

        return z

    def _compute_marginal_mu_sigma(
        self,
        z: jt.Real[np.ndarray, 'k b margidx dims'],
    ) -> tuple[jt.Real[np.ndarray, 'k margidx dims'], jt.Real[np.ndarray, 'k margidx dims dims']]:
        '''Compute mu and Sigma based on sampled z'''
        mus = z.mean(axis=1, keepdims=True)  ## (k, 1, margidx, dims)
        ## get covs over batch dim
        centered = z - mus                   ## (k, b, margidx, dims)
        covs = np.einsum('kbti,kbtj->ktij', centered, centered)
        covs /= z.shape[1] - 1               ## (k, margidx, dims, dims)
        mus = np.squeeze(mus, axis=1)        ## (k, margidx, dims)
        return mus, covs

    def _compute_mu_t_sigma_t_gpr(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        ts: jt.Real[np.ndarray, ' nt']
    ) -> tuple[jt.Real[np.ndarray, 'k nt obs'], jt.Real[np.ndarray, 'k nt obs']]:
        '''Compute mu_t and sigma_t from GPR'''
        mu_t_gpr = np.zeros((refidxs.shape[0], ts.shape[0], self.Xrefs.shape[-1]))
        sigma_t_gpr = np.zeros_like(mu_t_gpr)  ## (k, nt, obs)
        ts = ts.reshape((-1, 1))               ## (nt, 1)
        for i, idx in enumerate(refidxs):
            mu_i, std_i  = self.gprs[idx].predict(ts, return_std=True)
            mu_t_gpr[i] = mu_i.reshape(mu_i.shape[0], self.nobs)
            sigma_t_gpr[i] = std_i.reshape(std_i.shape[0], self.nobs)
        return mu_t_gpr, sigma_t_gpr

    def _compute_gpr_dmudt(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        ts: jt.Real[np.ndarray, ' nt']
    ) -> jt.Real[np.ndarray, 'k nt obs']:
        '''Compute time derivative of GPR mean function

        Rasmussen and Williams, Gaussian Processes for Machine Learning, 2006
        Formula (2.25)

        dmu_dt = d/dt Kstar^T @ alpha
        alpha = K^{-1} y
        Kstar = K(xtrain, ts)
        K = K(xtrain, xtrain)
        y = GPR(xtrain)
        '''
        Xtrain = np.zeros((refidxs.shape[0], self.gpr_nt, 1))
        Alpha = np.zeros((refidxs.shape[0], self.gpr_nt, self.nobs))
        ts = ts.reshape((-1, 1))

        ## dKstar_dt = rbf(xtrain, xtest) * (xtrain - xtest) / ell^2
        Scales = np.zeros((refidxs.shape[0]))
        Kstar = np.zeros((refidxs.shape[0], self.gpr_nt, ts.shape[0]))
        for i, idx in enumerate(refidxs):
            gpr = self.gprs[idx]
            kernel = gpr.kernel_.k1  ## kernel is RBFKernel + WhiteKernel
            Xtrain[i] = gpr.X_train_
            Alpha[i] = gpr.alpha_
            Scales[i] = kernel.length_scale
            Kstar[i] = kernel(Xtrain[i], ts)
        chainrule_mult = (Xtrain - ts.T[None, ...])  ## (k, gpr_nt, nt)
        chainrule_mult /= (Scales ** 2)[:, None, None]
        Kstar *= chainrule_mult                      ## (k, gpr_nt, nt)
        dmu_dt = Kstar.swapaxes(1, 2) @ Alpha        ## (k, nt, obs)
        return dmu_dt

    # def _compute_mu_t_aug(
        # self,
        # mu_t_obs: jt.Real[np.ndarray, 'k nt obs'],
        # mu_t_hid: jt.Real[np.ndarray, 'k b nt hid']
    # ) -> jt.Real[np.ndarray, 'k b nt obs+hid']:
        # '''Compute mu_t augmented with ref data
#
        # mu_t_aug = (mu_t_aug_obs, mu_t_aug_hid)
        # '''
        # k = mu_t_obs.shape[0]
        # mu_t_aug = np.zeros((k, self.b, self.nt, self.dim))
        # mu_t_aug[..., self.obsmask] = mu_t_obs[:, None]
        # mu_t_aug[..., self.hidmask] = mu_t_hid
        # return mu_t_aug
#
    # def _compute_mu_t_aug_prime(
        # self,
        # mu_t_obs_prime: jt.Real[np.ndarray, 'k nt obs'],
        # mu_t_hid_prime: jt.Real[np.ndarray, 'k b nt hid'],
    # ) -> jt.Real[np.ndarray, 'k nt dims']:
        # '''Compute mu_t_prime augmented with ref data
#
        # mu_t_aug_prime = (mu_t_aug_prime_obs, mu_t_aug_prime_hid)
        # '''
        # mu_t_aug_prime = np.zeros((refidxs.shape[0], self.b, self.nt, self.dim))
        # mu_t_gpr_prime = self._compute_gpr_dmudt(refidxs, ts)
        # mu_t_aug_prime[..., self.obsmask] = mu_t_gpr_prime[:, None]
        # mu_t_aug_prime[..., self.hidmask] = (cond_mu_tpd - cond_mu_tmd) / self.delta
        # return mu_t_aug_prime

    def _concat_mu_t_obshid(
        self,
        mu_t_obs: jt.Real[np.ndarray, 'k nt obs'],
        mu_t_hid: jt.Real[np.ndarray, 'k b nt hid'],
    ) -> jt.Real[np.ndarray, 'k b nt obs+hid']:
        '''Concat mu batches in feature dim.

        Broadcasts mu_t_obs to batch dim.
        '''
        concat_mu_t = np.zeros((*mu_t_hid.shape[:-1], self.dim))
        concat_mu_t[..., self.obsmask] = mu_t_obs[:, None]
        concat_mu_t[..., self.hidmask] = mu_t_hid
        return concat_mu_t

    # def _compute_xt_diff(
        # self,
        # xt: jt.Real[np.ndarray, 'k b nt dims'],
        # mu_t: jt.Real[np.ndarray, 'k nt dims']
    # ) -> jt.Real[np.ndarray, 'k b nt dims']:
        # '''Convenience method to compute xt - mu_t'''
        # return xt - mu_t[:, None]

    ## All mixin methods, ordered by call order in __next__()
    ## Time Sampling Mixin Method Signatures
    @abstractmethod
    def _sample_ts(self) -> jt.Float64[np.ndarray, ' nt']:
        '''Samples batch of times using TimeMixin'''
        raise NotImplementedError

    ## TimeRFFMixin
    @overload
    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, ' nt']
    ) -> jt.Float64[np.ndarray, 'nt rff_dim*2']:
        ...

    ## TimeNoEnrichMixin
    @overload
    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, ' nt']
    ) -> jt.Float64[np.ndarray, 'nt 1']:
        ...

    @abstractmethod
    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, ' nt']
    ) -> Times:
        raise NotImplementedError

    ## Flow Matching Mixin Method Signatures
    @abstractmethod
    def _compute_mu_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        raise NotImplementedError

    ## AFMixin
    @overload
    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[jt.Real[np.ndarray, 'k nt dims dims'], jt.Real[np.ndarray, 'k dims dims']]:
        ...

    ## IFCBMixin
    @overload
    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[float, None]:
        ...

    ## IFSBMixin
    @overload
    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[jt.Real[np.ndarray, ' nt'], jt.Real[np.ndarray, ' nt']]:
        ...

    @abstractmethod
    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[Sigma_T, Aux]:
        raise NotImplementedError

    ## AFMixin
    @overload
    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: jt.Real[np.ndarray, 'k nt dims dims'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        eps: jt.Real[np.ndarray, 'k b nt dims'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        ...

    ## IFCBMixin
    @overload
    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: float,
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        eps: jt.Real[np.ndarray, 'k b nt dims'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        ...

    ## IFSBMixin
    @overload
    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: jt.Real[np.ndarray, ' nt'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        eps: jt.Real[np.ndarray, 'k b nt dims'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        ...

    @abstractmethod
    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: Sigma_T,
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        eps: jt.Real[np.ndarray, 'k b nt dims'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        raise NotImplementedError

    ## AFMixin
    @overload
    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        aux: jt.Real[np.ndarray, 'k dims dims']
    ) -> jt.Real[np.ndarray, 'k nt dims dims']:
        ...

    ## IFCBMixin
    @overload
    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        aux: None
    ) -> None:
        ...

    ## IFSBMixin
    @overload
    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        aux: jt.Real[np.ndarray, ' nt']
    ) -> jt.Real[np.ndarray, ' nt']:
        ...

    @abstractmethod
    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        aux: Aux
    ) -> A_T_Prime_A_T_Inv:
        raise NotImplementedError

    ## AFMixin
    @overload
    def _compute_ut(
        self,
        xt_diff: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime_A_t_inv: jt.Real[np.ndarray, 'k nt dims dims']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        ...

    ## IFCBMixin
    @overload
    def _compute_ut(
        self,
        xt_diff: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime_A_t_inv: None
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        ...

    ## IFSBMixin
    @overload
    def _compute_ut(
        self,
        xt_diff: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime_A_t_inv: jt.Real[np.ndarray, ' nt']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        ...

    @abstractmethod
    def _compute_ut(
        self,
        xt_diff: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime_A_t_inv: A_T_Prime_A_T_Inv
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        raise NotImplementedError

    ## Score Matching Mixin Method Signatures
    ## NSMixin
    @overload
    def _compute_lambda(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> None:
        ...

    ## ASMixin
    @overload
    def _compute_lambda(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> jt.Float64[np.ndarray, 'k nt dims dims']:
        ...

    @abstractmethod
    def _compute_lambda(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> jt.Float64[np.ndarray, 'k nt dims dims'] | None:
        raise NotImplementedError

    def _log_batch_internals(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        So: jt.Real[np.ndarray, 'nt obs'],
        Qo: jt.Real[np.ndarray, 'nt obs obs'],
        Sh: jt.Real[np.ndarray, 'k nt hid'],
        Qh: jt.Real[np.ndarray, 'k nt hid hid'],
        mu_correction: jt.Real[np.ndarray, 'k b nt hid'],
    ) -> None:
        '''Logs internal computation diagnostics for current batch.'''
        curr_idx = self.prev_batch_idx + 1
        k = Qh.shape[0]
        i = self.batch_split_idxs[self.prev_batch_idx]
        j = i + k
        QhSh = Qh[..., :self.nobs] * Sh[..., None, :self.nobs]
        SoQo = (Qo * (1 / (So[..., None, :] + self.reg))).swapaxes(-1, -2)
        gain = QhSh @ SoQo  ## Sigma_h^(1/2) @ [I 0]^T @ Sigma_o^(1/2])

        self.ts_history[curr_idx] = ts
        self.gain_history[i:j] = np.linalg.norm(gain, ord='fro', axis=(-1, -2))
        self.mu_correction_history[i:j] = np.linalg.norm(mu_correction, axis=-1)
        self.eigvals_obs_history[curr_idx] = np.square(So)
        self.eigvecs_obs_history[curr_idx] = Qo
        self.eigvals_hid_history[i:j] = np.square(Sh)
        self.eigvecs_hid_history[i:j] = Qh
        self.prev_batch_idx = curr_idx

    def _log_epoch_internals(self) -> None:
        '''Log internal computation diagnostics for current epoch.'''
        self.ts_history_all.append(self.ts_history.copy())
        self.gain_history_all.append(self.gain_history.copy())
        self.mu_correction_history_all.append(self.mu_correction_history.copy())
        self.eigvals_obs_history_all.append(self.eigvals_obs_history.copy())
        self.eigvecs_obs_history_all.append(self.eigvecs_obs_history.copy())
        self.eigvals_hid_history_all.append(self.eigvals_hid_history.copy())
        self.eigvecs_hid_history_all.append(self.eigvecs_hid_history.copy())

    def get_logs(self) -> Diagnostics | None:
        '''Returns diagnostic logs up to last full epoch.

        The following arrays have 1st dim (zero-indexed)
        representing nbatches:
            ts_history_all
            eigvals_obs_history_all
            eigvecs_obs_history_all

        The following arrays have 1st dim (zero-indexed)
        representing nrefs:
            gain_history_all
            mu_correction_history_all
            eigvals_hid_history_all
            eigvecs_hid_history_all

        You can transform nrefs to nbatches in the following
        2-stage transformation:
            1) Split nrefs into [nbatches, k]
            2) Reduce to just nbatches

        ```python
        # arr := ndarray of shape (nepochs nrefs [ b ] nt ...)
        # where [ b ] denotes an optional dimension that may not be present
        arr = np.split(arr, batch_split_idxs, axis=1)
        # arr := list (len=nbatches) of ndarray of shape (nepochs k [ b ] nt ...)
        klast = arr[-1].shape[1]
        if klast == 0:  # np.split() creates empty trailing chunk when batch_split_idxs[-1] == nrefs
            arr = arr[:-1]  # arr[-1] has degenerate shape (nepochs 0 [ b ] nt ...)
        nbatches = len(arr)
        # take reduction over k [and b] then swap axis order to (nepochs nbatches nt ...)
        # for example reduction could be arr.mean()
        # reduce_b := bool indicating whether to also reduce over dim b
        # this argument should only be True for mu_correction_history_all
        reduce_axis = (1, 2) if reduce_b else 1
        arr = np.stack([arr[i].<reduction>(axis=reduce_axis) for i in range(nbatches)])
        # arr := ndarray of shape (nbatches nepochs nt ...)
        arr = arr.swapaxes(0, 1)
        # arr := ndarray of shape (nepochs nbatches nt ...)
        ```
        '''
        if self.diagnostics:
            return {
                'ts_history_all':            np.array(self.ts_history_all),
                'gain_history_all':          np.array(self.gain_history_all),
                'mu_correction_history_all': np.array(self.mu_correction_history_all),
                'eigvals_obs_history_all':   np.array(self.eigvals_obs_history_all),
                'eigvecs_obs_history_all':   np.array(self.eigvecs_obs_history_all),
                'eigvals_hid_history_all':   np.array(self.eigvals_hid_history_all),
                'eigvecs_hid_history_all':   np.array(self.eigvecs_hid_history_all),
                'batch_split_idxs':          self.batch_split_idxs
            }
        else:
            return None

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Self:
        '''First resets iteration state, then returns self'''
        self.prng.shuffle(self._Xrefidxs)
        self._batch_iter = (
            self._Xrefidxs[i*self.k:(i+1)*self.k]
            for i in range(self._len)
        )
        return self

    def __next__(self) -> GCFMBatch:
        try:
            refidxs = next(self._batch_iter)
        except StopIteration:
            if self.diagnostics:
                self._log_epoch_internals()
                ## reset prev batch idx for diagnostics
                self.prev_batch_idx = -1  # updated in _log_batch_internals()
            raise

        ## Sample k refs
        refs = self.Xrefs[refidxs]
        self.Sigma_t_obs.fit(refs)

        ## Independently sample minibatch
        xs = self._get_xs_minibatch()

        ## Resample according to refs
        z = self._sample_z_given_refs(xs, refs)
        mus, covs = self._compute_marginal_mu_sigma(z)

        ## Sample t according to chosen distribution
        ts = self.delta_half + (self._sample_ts() * self.t_scale)
        tspd = ts + self.delta_half  ## tspd -> ts plus delta/2
        tsmd = ts - self.delta_half  ## tsmd -> ts minus delta/2

        batch_shape = (refidxs.shape[0], self.b, self.nt)

        ## Main algorithm
        eps = self.prng.normal(size=(*batch_shape, self.dim))
        mu_t = self._compute_mu_t(ts, mus)
        # Sigma_t, aux = self._compute_sigma_t(ts, covs)
        # Sigma_tdelta, auxdelta = self._compute_sigma_t(tdelta, covs)
        Sigma_ot, Sigma_ht, *aux = self._compute_sigma_t(ts, covs)
        Sigma_otpd, Sigma_htpd, *auxtpd = self._compute_sigma_t(tspd, covs)
        Sigma_otmd, Sigma_htmd, *auxtmd = self._compute_sigma_t(tsmd, covs)
        mu_t_gpr, Sigma_t_gpr = self._compute_mu_t_sigma_t_gpr(refidxs, ts)
        So, Qo, Sh, Qh = self._process_Sigma_t(Sigma_ot, Sigma_ht)
        cond_mu_t, cond_A_t, mu_correction_t = self._compute_cond_params(
            mu_t[:, :, self.hidmask], Qh, Sh, eps[:, :, :, self.obsmask]
        )
        Sopd, Qopd, Shpd, Qhpd = self._process_Sigma_t(Sigma_otpd, Sigma_htpd)
        cond_mu_tpd, cond_A_tpd, mu_correction_tpd = self._compute_cond_params(
            mu_t[:, :, self.hidmask], Qhpd, Shpd, eps[:, :, :, self.obsmask]
        )
        Somd, Qomd, Shmd, Qhmd = self._process_Sigma_t(Sigma_otmd, Sigma_htmd)
        cond_mu_tmd, cond_A_tmd, mu_correction_tmd = self._compute_cond_params(
            mu_t[:, :, self.hidmask], Qhmd, Shmd, eps[:, :, :, self.obsmask]
        )
        # xt = self._sample_xt(refidxs, mu_t, Sigma_t, mu_t_gpr, Sigma_t_gpr, eps)
        Ao = self.Sigma_t_obs(ts, derivative=False, chol=True)
        ## TODO: align IFCB and IFSB Mixins to follow this method signature?
        xt = self._sample_xt(refidxs, mu_t_gpr, Ao, cond_mu_t, cond_A_t, eps)
        mu_t_aug = self._concat_mu_t_obshid(mu_t_gpr, cond_mu_t)
        mu_t_gpr_prime = self._compute_gpr_dmudt(refidxs, ts)
        cond_mu_t_prime = (cond_mu_tpd - cond_mu_tmd) / self.delta
        # mu_t_aug_prime = self._compute_mu_t_aug_prime(refidxs, ts, cond_mu_tpd, cond_mu_tmd)
        mu_t_aug_prime = self._concat_mu_t_obshid(mu_t_gpr_prime, cond_mu_t_prime)
        # xt_diff = self._compute_xt_diff(xt, mu_t_aug)
        xt_diff = xt - mu_t_aug
        # A_t_prime_A_t_inv = self._compute_A_t_prime_A_t_inv(ts, aux)
        ## TODO: align IFCB and IFSB Mixins to follow this method signature?
        A_t_prime_A_t_inv, Sigma_t = self._compute_A_t_prime_A_t_inv(  ## k nt dim dim
            ts,
            Sigma_ot,
            Sigma_ht,
            So,
            Qo,
            Sh,
            Qh,
            *aux,  ## Sigma_0, C, C_t for AFMixin
            Sopd,
            Qopd,
            Shpd,
            Qhpd,
            Somd,
            Qomd,
            Shmd,
            Qhmd
        )
        ut = self._compute_ut(xt_diff, mu_t_aug_prime, A_t_prime_A_t_inv)
        lt = self._compute_lambda(ts, covs)

        if self.diagnostics:
            self._log_batch_internals(ts, So, Qo, Sh, Qh, mu_correction_t)

        ## Flatten and cast into Tensors of shape (k*b*nt, dims)
        ## and cast to float32 for compatibility with default torch float operations
        ts = self._enrich_ts(ts)
        ts = np.broadcast_to(ts[None, None, ...], (*batch_shape, ts.shape[-1]))
        ts = torch.from_numpy(ts.reshape((-1, ts.shape[-1])).astype(np.float32))
        xt = torch.from_numpy(xt.reshape((-1, xt.shape[-1])).astype(np.float32))
        ut = torch.from_numpy(ut.reshape((-1, ut.shape[-1])).astype(np.float32))
        eps = torch.from_numpy(eps.reshape((-1, eps.shape[-1])).astype(np.float32))
        if lt is not None:
            lt = np.broadcast_to(lt[:, None, ...], (*batch_shape, self.dim, self.dim))
            lt = torch.from_numpy(lt.reshape((-1, *lt.shape[-2:])).astype(np.float32))

        return ts, xt, ut, eps, lt


class UniformTimeMixin:
    '''Samples batch of times from Unif(0, 1)'''

    def _sample_ts(self) -> jt.Float64[np.ndarray, ' nt']:
        return self.prng.random(size=self.nt)


class BetaTimeMixin:
    '''Samples batch of times from Beta(a, a)'''

    def _sample_ts(self) -> jt.Float64[np.ndarray, ' nt']:
        return self.prng.beta(self.beta_a, self.beta_a, size=self.nt)


class TimeRFFMixin:
    '''Enriches time with Random Fourier Features

    Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains

    Tancik et al.

    arxiv.org/pdf/2006.10739
    '''

    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, ' nt']
    ) -> jt.Float64[np.ndarray, 'nt rff_dim*2']:
        Bt = self.B * ts[:, None]  ## (nt, rff_dim)
        cosBt = np.cos(Bt)
        sinBt = np.sin(Bt)
        return np.concatenate((cosBt, sinBt), axis=1)


class TimeNoEnrichMixin:
    '''Dummy class with no time enrichment'''

    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, ' nt']
    ) -> jt.Float64[np.ndarray, 'nt 1']:
        return ts[:, None]


## CURRENTLY DOES NOT CONVERGE!
class AFMixin:
    '''Anisotropic Flow Mixin

    All methods are coupled!
    '''

    def _compute_mu_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        r'''Compute mu_t for W2 geodesic between MVNs

        \mu_t = t \mu_1 + (1 - t) \mu_0
        '''
        return batch_interp(mus[:, 0], mus[:, 1], ts)         ## (k, nt, dims)

    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[jt.Real[np.ndarray, 'k nt dims dims'], jt.Real[np.ndarray, 'k dims dims']]:
        r'''Compute Sigma_t for W2 geodesic between MVNs

        C = \Sigma_1^{1/2} (\Sigma_1^{1/2} \Sigma_0 \Sigma_1^{1/2})^{-1/2} \Sigma_1^{1/2}
        C_t = tC + (1 - t)I
        \Sigma_t = C_t \Sigma_0 C_t

        Beware numerical errors resulting in non-symmetric matrices!
        '''
        k, nt = covs.shape[0], ts.shape[0]

        ## compute interpolated observed cov
        Sigma_ot = self.Sigma_t_obs(ts) + (np.eye(self.nobs)[None] * self.reg)  ## (nt, obs, obs)

        ## Compute W2 Geodesic for hidden vars
        Ih = np.eye(self.nhid)
        Sigma_1_sqrt = batch_sqrtm(covs[:, 1, *self.hidhidmask])  ## (k, hid, hid)
        Sigma_0_h = covs[:, 0, *self.hidhidmask]  ## for convenience later in Atprime @ At
        Sigma_101 = Sigma_1_sqrt @ Sigma_0_h @ Sigma_1_sqrt
        # Regularize to avoid bad matrix
        Sigma_101 += Ih[None] * self.reg
        Sigma_101_inv_sqrt = batch_inv_sqrtm(Sigma_101)
        C = Sigma_1_sqrt @ Sigma_101_inv_sqrt @ Sigma_1_sqrt      ## (k, hid, hid)
        C_t = batch_interp(Ih[None], C, ts)                              ## (k, nt, hid, hid)
        Sigma_ht = C_t @ Sigma_0_h[:, None] @ C_t
        Sigma_ht += np.eye(self.nhid)[None, None] * self.reg

        ## RETURNS CROSS-COV BLOCKS AS 0!
        ## FILL IN CROSS-COV INSIDE SAMPLE_XT() TO REDUCE REDUNDANT COMPUTATIONS
        # return Sigma_t, C
        return Sigma_ot, Sigma_ht, Sigma_0_h, C, C_t

    def _process_Sigma_t(
        self,
        Sigma_ot: jt.Real[np.ndarray, 'nt obs obs'],
        Sigma_ht: jt.Real[np.ndarray, 'k nt hid hid']
    ) -> tuple[jt.Real[np.ndarray, 'nt obs'], jt.Real[np.ndarray, 'nt obs obs'], jt.Real[np.ndarray, 'k nt hid'], jt.Real[np.ndarray, 'k nt hid hid']]:
        '''Compute eigendecompositions and precompute some products'''
        Lh, Qh = np.linalg.eigh(Sigma_ht)  ## Qh is (k, nt, hid, hid)
        Lo, Qo = np.linalg.eigh(Sigma_ot)  ## Qo is (nt, obs, obs)
        # Ao = Qo * np.sqrt(Lo)[..., None, :]

        ## Clip eigvals to min positive
        Lo = batch_eigval_replace_neg(Lo)
        Lh = batch_eigval_replace_neg(Lh)

        ## Order Lh elements, Qh columns in DESCending order
        Qh = np.flip(Qh, axis=-1)
        Lh = np.flip(Lh, axis=-1)

        if self.cc_impute:
            ## TODO: Pre-orient or post score-sort orient?
            ## Orient eigvecs to consistent signs
            # Qh, _, _ = orient_eigenvectors(Qh)
            # Qo, _, _ = orient_eigenvectors(Qo)

            k = Sigma_ht.shape[0]
            noise_floor = np.median(Lo, axis=-1, keepdims=True)  ## (nt, 1)
            lambda_bound = noise_floor * np.square(1 + np.sqrt(self.nobs / k))  ## (nt, 1)
            z = (Lo - lambda_bound) / self.tau  ## (nt, nobs)
            w = np_sigmoid(z)  ## (nt, nobs)

            ## if score then Strategy A: Signal Precision Maximization (High Gain)
            ## else Strategy B: Principal Component Matching (Robust)
            score = w / (Lo + self.reg) if self.maxgain else w * Lo
            sortidx = np.flip(np.argsort(score, axis=-1), axis=-1)

            ## sort Qo and Lo
            Qo = np.take_along_axis(Qo, sortidx[:, None], axis=-1)
            Lo = np.take_along_axis(Lo, sortidx, axis=-1)

            ## Orient eigvecs to consistent signs
            Qh, _, _ = orient_eigenvectors(Qh)
            Qo, _, _ = orient_eigenvectors(Qo)

        return np.sqrt(Lo), Qo, np.sqrt(Lh), Qh

    def _compute_cond_params(
        self,
        mu_t_hid: jt.Real[np.ndarray, 'k nt hid'],
        Qh: jt.Real[np.ndarray, 'k nt hid hid'],
        Sh: jt.Real[np.ndarray, 'k nt hid'],
        eps_obs: jt.Real[np.ndarray, 'k b nt obs']
    ) -> tuple[jt.Real[np.ndarray, 'k b nt hid'], jt.Real[np.ndarray, 'k b nt hid hid'], jt.Real[np.ndarray, 'k b nt hid']]:
        '''Compute params for mu_t_hid and Sigma_t_hid given obs.

        Note that cross-covariance Sigma_oht and gain Kt
        are never explicitly computed!
        '''
        if self.cc_impute:
            Qh_hi = Qh[..., :self.nobs]
            Sh_hi = Sh[..., :self.nobs]
            Ah_hi = Qh_hi * Sh_hi[..., None, :]  ## Multiply cols of Qh by Lh^{1/2}

            Qh_lo = Qh[..., self.nobs:]
            Sh_lo = Sh[..., self.nobs:]

            mu_correction = np.matvec(Ah_hi[:, None], eps_obs)
            cond_mu_t = mu_t_hid[:, None] + mu_correction
            cond_A_t = Qh_lo * Sh_lo[..., None, :]
        else:
            mu_correction = np.zeros((mu_t_hid.shape[0], self.b, self.nt, self.nhid))
            cond_mu_t = mu_t_hid[:, None]
            cond_A_t = Qh * Sh[..., None, :]

        return cond_mu_t, cond_A_t, mu_correction

    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        # mu_t: jt.Real[np.ndarray, 'k nt dims'],
        # Sigma_t: jt.Real[np.ndarray, 'k nt dims dims'],
        # Sigma_ot,
        # Sigma_ht,
        # Sigma_otdelta,
        # Sigma_htdelta,
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        # Sigma_ot,
        Ao: jt.Real[np.ndarray, 'nt obs obs'],
        cond_mu_t: jt.Real[np.ndarray, 'k b nt hid'],
        # Ao,
        # Lo,
        # Qo,
        cond_A_t: jt.Real[np.ndarray, 'k nt hid hid'],
        # Sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        eps: jt.Real[np.ndarray, 'k b nt dims'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Sample xt

        First sample xt_obs using mu_t and sigma_t from GPR
        Then compute conditional mu_t_hid|obs and Sigma_t_hid|obs
        Use conditional params to sample xt_hid
        Return xt = (xt_obs, xt_hid)

        Beware numerical errors resulting in non-symmetric matrices!
        '''
        xt = np.zeros_like(eps)

        ## First compute xt_obs from mu_gpr, sigma_gpr
        ## Assume sigma_gpr is list of stddevs
        ## ==> ref vars have 0 covariance == indep
        # xt_obs = (sigma_t_gpr[:, None] * eps[:, :, :, self.obsmask]) + mu_t_gpr[:, None]

        ## Compute xt_obs from gpr and Sigma_ot
        # Ao = Qo * np.sqrt(Lo)[..., None, :]
        # Ao = np.linalg.cholesky(Sigma_ot)
        xt_obs = (
            np.matvec(Ao[None, None], eps[:, :, :, self.obsmask])
            + mu_t_gpr[:, None]
        )

        ## Conditional sampling. NOTE: Sigma_oht and Kt never explicitly computed
        # cond_mu_t = mu_t[..., self.hidmask] \
                    # + np.matvec(QhLh_hi @ Qo.swapaxes(-1, -2), eps[:, :, :, self.hidmask])
        # cond_A_t = Qh_lo * Lh_lo[..., None, :]
        # print('cond_A_t shape', cond_A_t.shape)
        # print('cond_mu_t shape', cond_mu_t.shape)
        # xt_hid = np.matvec(cond_A_t[:, None], eps[:, :, :, self.hidmask]) + cond_mu_t

        ## Only take eps vector of length nhid - nobs, reflecting reduced rank of cond_A_t
        effective_eps = eps[:, :, :, self.hidmask]
        if self.cc_impute:  # to handle reduced rank for cross-cov imputation
            effective_eps = effective_eps[..., self.nobs:]
        ## when nobs == nhid the below matvec returns 0
        ## which handles the case that xt_hid is deterministic
        ## given xt_obs (consequence of cross-covariance structure)
        # print('cond a shape', cond_A_t[:, None].shape)
        # print('eps shape', effective_eps.shape)
        # print('matvec shape', np.matvec(cond_A_t[:, None], effective_eps).shape)
        # print('cond mu t shape', cond_mu_t.shape)
        xt_hid = (np.matvec(cond_A_t[:, None], effective_eps) + cond_mu_t)
        # print('xt hid == cond mu t?', np.all(xt_hid == cond_mu_t), np.allclose(xt_hid, cond_mu_t))
        # print('xt_hid shape', xt_hid.shape)

        # obs_diff = xt_obs - mu_t[:, None, :, self.obsmask]
        # cond_mu_t = mu_t[:, None, :, self.hidmask] + np.matvec(Kt[:, None], obs_diff)
        # cond_Sigma_t = Sigma_t[..., *self.hidhidmask] - (Kt @ Sigma_t[..., *self.obshidmask])
        # cond_A_t = np.linalg.cholesky(cond_Sigma_t)
        # xt_hid = np.matvec(cond_A_t[:, None], eps[:, :, :, self.hidmask]) + cond_mu_t

        ## Next compute conditional mu_t_hid|obs and sigma_t_hid|obs
        # Sigma_t_hidobs = Sigma_t[:, :, *self.hidobsmask]                  ## (k, nt, hid, obs)
        # Sigma_t_obsobs = Sigma_t[:, :, *self.obsobsmask]              ## (k, nt, obs, obs)
        # Sigma_t_obsobs_inv = batch_inv(Sigma_t_obsobs)                ## (k, nt, obs, obs)
        ## TODO: Test more stable inverse
        # L_t_obsobs, lower = cho_factor(Sigma_t_obsobs)
        # L_t_obsobs = self.Sigma_t_obs(self.ts_history[-1], chol=True)
        # Sigma_t_obsobs_inv = cho_solve((L_t_obsobs, lower), np.eye(self.nobs)[None, None])
        # B = Sigma_t_hidobs @ Sigma_t_obsobs_inv                           ## (k, nt, hid, obs)

        # obs_diff = xt_obs - mu_t[:, None, :, self.obsmask]                ## (k, b, nt, obs)
        # cond_mu_t = np.matvec(B[:, None], obs_diff)                       ## (k, b, nt, hid)
#
        # cond_Sigma_t = B @ Sigma_t[:, :, *self.obshidmask]
        # cond_Sigma_t = Sigma_t[:, :, *self.hidhidmask] - cond_Sigma_t     ## (k, nt, hid, hid)
        # cond_Sigma_t += np.eye(cond_Sigma_t.shape[-1])[None, ...] * self.reg

        ## TODO: use cholesky here instead of eigen decomp?
        # cond_A_t = batch_sqrtm(cond_Sigma_t)                              ## (k, nt, hid, hid)
        # cond_A_t = np.linalg.cholesky(cond_Sigma_t)

        ## Sample xt_hid|obs
        ## xt_hid vars have nonzero covariance
        # xt_hid = np.matvec(cond_A_t[:, None], eps[:, :, :, self.hidmask]) ## (k, b, nt, hid)
        # xt_hid += cond_mu_t

        xt[:, :, :, self.obsmask] = xt_obs
        xt[:, :, :, self.hidmask] = xt_hid
        return xt

    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        # aux: jt.Real[np.ndarray, 'k dims dims']
        Sigma_ot,
        Sigma_ht,
        So,
        Qo,
        Sh,
        Qh,
        Sigma_0_h,
        C: jt.Real[np.ndarray, 'k hid hid'],
        C_t,
        So_tpd,
        Qo_tpd,
        Sh_tpd,
        Qh_tpd,
        So_tmd,
        Qo_tmd,
        Sh_tmd,
        Qh_tmd,
    ) -> jt.Real[np.ndarray, 'k nt dims dims']:
        r'''Compute A_t_prime @ A_t_inv for A_t A_t^T = Sigma_t

        C \gets aux

        A_t = C_t Q_C \Lambda_C^{1/2}

        A_t^\prime = C_t^\prime Q_0 \Lambda_0^{1/2}

        A_t^{-1} = \Lambda_0^{-1/2} Q_0^{-1} C_t^{-1}

        A_t^\prime A_t^{-1} = C_t^\prime Q_0 \Lambda_0^{1/2} \Lambda_0^{-1/2} Q_0^{-1} C_t^{-1}
                            = C_t^\prime C_t^{-1}

        C_t^\prime = C - I

        C_t^{-1} = Q_C (t \Lambda_C + (1 - t)I)^{-1} Q_C^{-1}
        '''
        # I = np.eye(self.dim)  # noqa: E741
        # C = aux
        # C_t_prime = C - I[None, ...]
#
        # L_C, Q_C = np.linalg.eigh(C)
        # L_C_t_inv = 1 / batch_interp(np.ones(self.dim)[None, ... ], L_C, ts)
        # C_t_inv = Q_C[:, None] \
                  # @ np.apply_along_axis(np.diag, -1, L_C_t_inv) \
                  # @ Q_C[:, None].swapaxes(-1, -2)
        # return C_t_prime[:, None, ...] @ C_t_inv

        ## Construct Sigma_t in blocks
        k = C.shape[0]
        Sigma_t = np.zeros((k, self.nt, self.dim, self.dim))
        ## First do obs and hid covariance blocks
        Sigma_t[..., *self.obsobsmask] = Sigma_ot[None]
        Sigma_t[..., *self.hidhidmask] = Sigma_ht

        if self.cc_impute:
            ## Sigma_oh = Lo @ Do^{1/2} [I 0] Dh^{1/2} Vh.T
            QoSo = Qo * So[..., None, :]  ## Multiply Qo[:, i] by So[i]
            ShQh = (Qh[..., :self.nobs] * Sh[..., None, :self.nobs]).swapaxes(-1, -2)
            Sigma_oh = QoSo[None] @ ShQh  ## (k, nt, obs, hid)
            Sigma_t[..., *self.obshidmask] = Sigma_oh
            Sigma_t[..., *self.hidobsmask] = Sigma_oh.swapaxes(-1, -2)

        I = np.eye(self.dim)
        Sigma_t += I[None, None] * self.reg  ## regularize just in case
        A_t = np.linalg.cholesky(Sigma_t)

        ## Construct Sigma_t_prime in blocks
        Sigma_t_prime = np.zeros_like(Sigma_t)
        ## First do obs and hid covariance blocks
        Sigma_oo_prime = self.Sigma_t_obs(ts, derivative=True)
        ## Sigma_hh_prime = d/dt [C_t @ Sigma_0 @ C_t]
        ##                = C_t_prime @ Sigma_0 @ C_t + C_t @ Sigma_0 @ C_t_prime
        ## where C_t_prime = C - I
        I_h = np.eye(self.nhid)
        C_t_prime = C - I_h[None]
        Sigma_hh_prime = (C_t_prime[:, None] @ Sigma_0_h[:, None] @ C_t) \
                          + (C_t @ Sigma_0_h[:, None] @ C_t_prime[:, None])
        if self.cc_impute:
            ## Cross-covariance block (finite difference)
            ## Sigma_oh = Lo @ Do^{1/2} [I 0] Dh^{1/2} Vh.T
            ## Do ts + delta
            QoSo_tpd = Qo_tpd * So_tpd[..., None, :]  ## Multiply Qo[:, i] by So[i]
            ShQh_tpd = (Qh_tpd[..., :self.nobs] * Sh_tpd[..., None, :self.nobs]).swapaxes(-1, -2)
            Sigma_oh_tpd = QoSo_tpd[None] @ ShQh_tpd

            ## ts - delta
            QoSo_tmd = Qo_tmd * So_tmd[..., None, :]  ## Multiply Qo[:, i] by So[i]
            ShQh_tmd = (Qh_tmd[..., :self.nobs] * Sh_tmd[..., None, :self.nobs]).swapaxes(-1, -2)
            Sigma_oh_tmd = QoSo_tmd[None] @ ShQh_tmd

            ## Central difference
            Sigma_oh_prime = (Sigma_oh_tpd - Sigma_oh_tmd) / self.delta

            ## Reconstruct Sigma_t_prime
            Sigma_t_prime[..., *self.obsobsmask] = Sigma_oo_prime[None]
            Sigma_t_prime[..., *self.hidhidmask] = Sigma_hh_prime
            Sigma_t_prime[..., *self.obshidmask] = Sigma_oh_prime
            Sigma_t_prime[..., *self.hidobsmask] = Sigma_oh_prime.swapaxes(-1, -2)

        ## Compute A_t_prime @ A_t_inv = A @ phi(A_inv @ Sigma_t_prime @ A_inv.T) @ A_inv
        A_t_inv = solve_triangular(A_t, I[None, None], lower=True)
        M = A_t_inv @ Sigma_t_prime @ A_t_inv.swapaxes(-1, -2)
        A_t_prime_A_t_inv = A_t @ self._compute_phi(M) @ A_t_inv

        return A_t_prime_A_t_inv, Sigma_t

    def _compute_phi(
        self, M: jt.Real[np.ndarray, '... d d']
    ) -> jt.Real[np.ndarray, '... d d']:
        '''Auxiliary function to compute phi(M)'''
        tril_idxs = np.tril_indices(self.dim, k=-1)
        diag_idxs = np.diag_indices(self.dim)
        phi = np.zeros_like(M)
        phi[..., *tril_idxs] = M[..., *tril_idxs]
        phi[..., *diag_idxs] = M[..., *diag_idxs] / 2
        return phi

    def _compute_ut(
        self,
        xt_diff: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime_A_t_inv: jt.Real[np.ndarray, 'k nt dims dims']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Compute ut = A_t_prime @ A_t_inv (xt - mu_t) + mu_t_prime'''
        Ax = np.matvec(A_t_prime_A_t_inv[:, None], xt_diff)  ## (k, b, nt, dims)
        # print('Ax shape', Ax.shape)  ## k b nt dims
        # return Ax + mu_t_prime[:, None]
        return Ax + mu_t_prime


class IFCBMixin:
    '''Isotropic Flow Constant Bridge Mixin

    All exposed methods are coupled!
    '''

    def _compute_mu_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        '''Compute mu_t of constant bridge'''
        return batch_interp(mus[:, 0], mus[:, 1], ts)  ## (k, nt, dims)

    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[float, None]:
        '''Compute sigma_t of constant bridge

        Args ts, covs are ignored. Only kept for consistent method signature.'''
        del ts, covs

        return self.sigma, None

    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: float,
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        eps: jt.Real[np.ndarray, 'k b nt dims'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Sample xt from N(mu_t_aug, sigma_t * I)

        Use mu_t and mu_t_gpr instead of precomputing mu_t_aug
        to keep method call order consistent with AFMixin.

        Args refidxs, sigma_t_gpr are ignored.
        '''
        del refidxs, sigma_t_gpr

        xt = np.zeros_like(eps)
        sigma_eps = Sigma_t * eps
        xt[:, :, :, self.obsmask] = sigma_eps[:, :, :, self.obsmask] \
                                    + mu_t_gpr[:, None]
        xt[:, :, :, self.hidmask] = sigma_eps[:, :, :, self.hidmask] \
                                    + mu_t[:, None, :, self.hidmask]
        return xt

    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        aux: None,
    ) -> None:
        '''Compute sigma_t_prime * sigma_t_inv

        sigma_t_prime = 0 so the product = 0.
        No variance correction so return None
        '''
        del ts, aux
        return None

    def _compute_ut(
        self,
        xt_diff: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime_A_t_inv: None
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Compute ut = sigma_t_prime * sigma_t_inv (xt - mu_t) + mu_t_prime

        For the constant bridge, sigma_t_prime == 0
        so only need to return mu_t_prime.

        Args xt_diff, A_t_prime_A_t_inv are ignored.
        '''
        del xt_diff, A_t_prime_A_t_inv

        k = mu_t_prime.shape[0]
        return np.broadcast_to(mu_t_prime[:, None], (k, self.b, self.nt, self.dim))


class IFSBMixin:
    '''Isotropic Flow Schrodinger Bridge Mixin

    All exposed methods are coupled!
    '''

    def _compute_mu_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        '''Compute mu_t of Schrodinger bridge'''
        return batch_interp(mus[:, 0], mus[:, 1], ts)  ## (k, nt, dims)

    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[jt.Real[np.ndarray, ' nt'], jt.Real[np.ndarray, ' nt']]:
        '''Compute sigma_t of Schrodinger bridge

        sigma_t = sigma * sqrt(t * (1 - t))

        Additionally returns t * (1 - t) for reuse in A_t_prime_A_t_inv
        Arg covs is ignored.
        '''
        del covs

        aux = ts * (1 - ts)
        sigma_t = self.sigma * np.sqrt(aux)
        return sigma_t, aux

    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, ' k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: jt.Real[np.ndarray, ' nt'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        eps: jt.Real[np.ndarray, 'k b nt dims'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Sample xt from N(mu_t_aug, sigma_t * I)

        Use mu_t and mu_t_gpr instead of precomputing mu_t_aug
        to keep method call order consistent with AFMixin.

        Args refidxs, sigma_t_gpr are ignored.
        '''
        del refidxs, sigma_t_gpr

        xt = np.zeros_like(eps)
        sigma_eps = Sigma_t[None, None, :, None] * eps
        xt[:, :, :, self.obsmask] = sigma_eps[:, :, :, self.obsmask] \
                                    + mu_t_gpr[:, None]
        xt[:, :, :, self.hidmask] = sigma_eps[:, :, :, self.hidmask] \
                                    + mu_t[:, None, :, self.hidmask]
        return xt

    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        aux: jt.Real[np.ndarray, ' nt']
    ) -> jt.Real[np.ndarray, ' nt']:
        '''Compute sigma_t_prime * sigma_t_inv

        sigma_t = sigma * sqrt(t * (1 - t))
        sigma_t_prime = sigma * 0.5 * (1 - 2t) / sqrt(t * (1 - t))
        sigma_t_prime * sigma_t_inv = 0.5 * (1 - 2t) / (t * (1 - t))

        aux <- t * (1 - t) computed at self._compute_sigma_t()
        '''
        return 0.5 * (1 - (2 * ts)) / (aux + self.sb_reg)

    def _compute_ut(
        self,
        xt_diff: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime_A_t_inv: jt.Real[np.ndarray, ' nt']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Compute ut = sigma_t_prime * sigma_t_inv (xt - mu_t) + mu_t_prime'''
        Ax = A_t_prime_A_t_inv[None, None, :, None] * xt_diff
        return Ax + mu_t_prime[:, None]


class ASMixin:
    '''Anisotropic Score Mixin

    Currently only compatible with IFMixins

    Returns lambda(t) schedule for stable computation of
    scaled score target
    '''

    def _compute_lambda(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> jt.Float64[np.ndarray, 'k nt dims dims']:
        '''Compute lambda for stable scaled score loss

        Set score loss weighting function lambda(t) to
        Sigma_t^{1/2} which allows for a stable (scaled) score loss of

        L_score = || lambda(t) @ st + eps ||^2

        where st is the output from the neural network
        and eps is the eps used in sampling xt.
        '''
        Sigma_1_sqrt = batch_sqrtm(covs[:, 1])                ## (k, dims, dims)
        Sigma_101 = Sigma_1_sqrt @ covs[:, 0] @ Sigma_1_sqrt
        ## Regularize to avoid bad matrix
        Sigma_101 += np.eye(Sigma_101.shape[-1])[None, ...] * self.reg
        Sigma_101_inv_sqrt = batch_inv_sqrtm(Sigma_101)
        C = Sigma_1_sqrt @ Sigma_101_inv_sqrt @ Sigma_1_sqrt  ## (k, dims, dims)
        I = np.eye(C.shape[-1])[None, ...]                    ## (1, dims, dims)  # noqa: E741
        C_t = batch_interp(I, C, ts)                          ## (k, nt, dims, dims)
        Sigma_t = C_t @ covs[:, 0][:, None] @ C_t             ## (k, nt, dims, dims)
        ## Regularize to avoid bad matrix
        Sigma_t += np.eye(self.dim)[None, None, ...] * self.reg
        return 2 * batch_sqrtm(Sigma_t) / (self.sigma ** 2)


class NSMixin:
    '''No Score Mixin'''

    def _compute_lambda(
        self,
        ts: jt.Real[np.ndarray, ' nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> None:
        return None


def build_sampler_class(
    time_sampler: Literal['uniform', 'beta'],
    use_time_enrich: bool,
    time_enrich: Literal['rff'],
    flow: Literal['isotropic', 'anisotropic'],
    flow_bridge: Literal['constant', 'schrodinger'],
    score: bool,
    score_shape: Literal['anisotropic']
) -> GCFMSamplerBase:
    '''Dynamically creates GCFMSampler with appropriate mixins'''
    if time_sampler == 'uniform':
        time_mixin = UniformTimeMixin
    elif time_sampler == 'beta':
        time_mixin = BetaTimeMixin
    else:
        raise ValueError(f'Unsupported time sampler "{time_sampler}"')

    if use_time_enrich:
        if time_enrich == 'rff':
            time_enrich_mixin = TimeRFFMixin
        else:
            raise ValueError(f'Unsupported time enricher "{time_enrich}"')
    else:
        time_enrich_mixin = TimeNoEnrichMixin

    if flow == 'isotropic':
        if flow_bridge == 'constant':
            flow_mixin = IFCBMixin
        elif flow_bridge == 'schrodinger':
            flow_mixin = IFSBMixin
        else:
            raise ValueError(f'Unsupported flow bridge "{flow_bridge}"')
    elif flow == 'anisotropic':
        flow_mixin = AFMixin
    else:
        raise ValueError(f'Unsupported flow shape "{flow}"')

    if score:
        if score_shape == 'anisotropic':
            score_mixin = ASMixin
        else:
            raise ValueError(f'Unsupported score shape "{score_shape}"')
    else:
        score_mixin = NSMixin

    bases = (time_mixin, time_enrich_mixin, flow_mixin, score_mixin, GCFMSamplerBase)
    return type('GCFMSampler', bases, {})


def main() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from torch.utils.data import DataLoader

    from trajaugcfm.constants import (
        CONSTOBS,
        DYNOBS,
        OBS,
        DATADIR
    )

    experiment = 'easy'
    data = np.load(os.path.join(DATADIR, experiment, 'data.npy'))  ## (drugcombs, N, T, *dims)
    # dynmask = build_indexer(OBS, dropvars=CONSTOBS)
    # data = data[:, :, :, dynmask]
#
    # dyn_if_vars = [dynvar for dynvar in DYNOBS if '_IF' in dynvar]
    # dynifmask = build_indexer(DYNOBS, dropvars=dyn_if_vars)
    # data = data[:, :, :, dynifmask]
    # data = data[0]

    obsmask = np.zeros(data.shape[-1], dtype=bool)
    # obsidxs = [0, 1, 2]
    obsidxs = [0, 1]
    obsmask[obsidxs] = True
    tidxs = [0, 400]
    # _nsplit = 250
    # _nsplit = 400
    _nsplit = 50

    Xrefs = data[:_nsplit][:, :, obsmask]
    Xs = data[_nsplit:, tidxs]
    Xobs = Xs[:, :, obsmask]
    print('Xobs shape', Xobs.shape)
    print('Xs shape', Xs.shape)
    print('Xrefs shape', Xrefs.shape)

    seed = 1000
    prng = np.random.default_rng(seed=seed)
    k = 16
    n = 128
    b = 16
    nt = 8
    rbfk_scale = 0.1
    # rbfk_bounds = (0.05, 5)
    rbfk_bounds = 'fixed'
    whitenoise = 0.1
    gpr_nt = 10
    rbfd_scale = 1.
    reg = 1e-8
    sigma = 1.0
    sb_reg = 1e-8
    beta_a = 2.0
    rff_seed = 2000
    rff_scale = 1.0
    rff_dim = 3
    diagnostics = True
    # diagnostics = False

    GCFMSampler = build_sampler_class(
        time_sampler='uniform',
        use_time_enrich=False,
        time_enrich='rff',
        flow='anisotropic',
        flow_bridge='schrodinger',
        score=False,
        score_shape='anisotropic'
    )
    gcfm_sampler = GCFMSampler(
        prng,
        Xs,
        Xrefs,
        obsmask,
        tidxs,
        k,
        n,
        b,
        nt,
        rbfk_scale=rbfk_scale,
        rbfk_bounds=rbfk_bounds,
        whitenoise=whitenoise,
        gpr_nt=gpr_nt,
        rbfd_scale=rbfd_scale,
        reg=reg,
        sigma=sigma,
        sb_reg=sb_reg,
        beta_a=beta_a,
        rff_seed=rff_seed,
        rff_scale=rff_scale,
        rff_dim=rff_dim,
        diagnostics=diagnostics,
    )
    print(gcfm_sampler.get_mixin_names())
    print('sampler len', len(gcfm_sampler))
    # ts = np.linspace(0, 1, 5)
    # print(ts)
    # Sigma_t = gcfm_sampler.Sigma_t_obs(ts)
    # Lt, Qt = np.linalg.eigh(Sigma_t)
    # print(np.all(Lt > 0))
    # return

    batch_size = None
    gcfm_loader = DataLoader(gcfm_sampler, batch_size=batch_size)
    for i, (ts, xt, ut, eps, lt) in enumerate(gcfm_loader):
        pass
        # print('ts shape', ts.shape)
        # print('xt shape', xt.shape)
        # print('ut shape', ut.shape)
        # print('eps shape', eps.shape)
        # if lt is not None:
            # print('lt shape', lt.shape)
        # break

    covs_history, Sigma_t_history, ts_history, mu_correction_t_history, batch_split_idxs = gcfm_sampler.get_logs()

    dims = covs_history.shape[-1]
    nhid = mu_correction_t_history.shape[-1]
    nobs = dims - nhid
    hidmask = ~obsmask
    obsobsmask = np.ix_(obsmask, obsmask)
    obshidmask = np.ix_(obsmask, hidmask)
    hidobsmask = np.ix_(hidmask, obsmask)
    hidhidmask = np.ix_(hidmask, hidmask)

    print('covs history shape', covs_history.shape)
    print('Sigma_t_history shape', Sigma_t_history.shape)
    print('ts history shape', ts_history.shape)
    print('mu correction t history shape', mu_correction_t_history.shape)

    def summarize_covs(covs):
        eigvals, eigvecs = np.linalg.eigh(covs)
        print('  mean trace', np.trace(covs, axis1=1, axis2=2).mean())
        print('  mean det', np.linalg.det(covs).mean())
        print('  mean logdet', np.linalg.slogdet(covs)[1].mean())
        print('  mean condition', (eigvals[:, -1] / eigvals[:, 0]).mean())
        print('  mean max eigval', eigvals[:, -1].mean())
        print('  mean min eigval', eigvals[:, 0].mean())
        print('  mean eigvec norm', np.sqrt(np.square(eigvecs).sum(axis=-2)).mean())
        print('  mean max eigvec norm', np.sqrt(np.square(eigvecs[..., -1]).sum(axis=-1)).mean())
        print('  num near singular', (eigvals[:, 0] < 1e-6).sum())

    print('covs0 history summary')
    summarize_covs(covs_history[:, 0])
    print('covs1 history summary')
    summarize_covs(covs_history[:, 1])
    print('Sigma t history')
    summarize_covs(Sigma_t_history.reshape(-1, dims, dims))
    print('Sigma t oo history')
    summarize_covs(Sigma_t_history[..., *obsobsmask].reshape(-1, nobs, nobs))
    print('Sigma t hh history')
    summarize_covs(Sigma_t_history[..., *hidhidmask].reshape(-1, nhid, nhid))


    print(Sigma_t_history[32, 7])

    corrnorm = np.sqrt(np.square(mu_correction_t_history).sum(axis=-1)).flatten()

    print('corr uniques', np.unique(corrnorm).shape)
    print('corr mean', corrnorm.mean())
    print('corr stddev', corrnorm.std())
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    sns.histplot(data=pd.DataFrame(data=corrnorm, columns=['norm']), x='norm', stat='density', kde=True, ax=ax)
    ax.set_xlabel(r'$\| V_{h,t}^\dagger D_{h,t}^{\dagger \, 1/2} \varepsilon_0 \|$')
    fig.tight_layout()
    fig.savefig('corrnorm.png')
    return
    Sigma_t_history, ts_history, split_idxs = gcfm_sampler.get_logs()
    print('Sigma_t_history shape', Sigma_t_history.shape)
    print('ts_history shape', ts_history.shape)
    # for cov in Sigma_t_history:
        # print(cov.shape)
    # split_idxs = np.cumsum([arr.shape[0] for arr in gcfm_sampler.Sigma_t_history])[:-1]
    # Sigmas = np.concatenate(Sigma_t_history, axis=0)
    # Sigma_t_history = np.array(Sigmas)
    # ts_history = np.array(gcfm_sampler.ts_history)
    print('sigma t type', type(Sigma_t_history))
    print('ts hist type', type(ts_history))
    ## (len, k, nt, d)
    print('Sigma t hist shape', Sigma_t_history.shape)
    print('Sigma t hist num', np.prod(Sigma_t_history.shape))
    ## (len, nt)
    print('ts history shape', ts_history.shape)
    print('split idxs', split_idxs)
    print('Sigma t history batches shape')
    for Sigma_t_batch in np.split(Sigma_t_history, split_idxs):
        print(Sigma_t_batch.shape)
    return
    hidmask = ~obsmask
    obsobsmask = np.ix_(obsmask, obsmask)
    obshidmask = np.ix_(obsmask, hidmask)
    hidobsmask = np.ix_(hidmask, obsmask)
    hidhidmask = np.ix_(hidmask, hidmask)

    oo = Sigma_t_history[..., *obsobsmask]
    hh = Sigma_t_history[..., *hidhidmask]
    oo_eigvals = np.linalg.eigvalsh(oo)
    hh_eigvals = np.linalg.eigvalsh(hh)
    oo_pds = oo_eigvals > 0
    hh_pds = hh_eigvals > 0
    print('oo not pds', (~oo_pds).sum())
    print('hh not pds', (~hh_pds).sum())
    print('oo eigval range', f'({oo_eigvals.min():.4f}, {oo_eigvals.max():.4f})')
    print('hh eigval range', f'({hh_eigvals.min():.4f}, {hh_eigvals.max():.4f})')


    # a = 1
    a = np.arange(-4, 10)[:, None, None]
    alpha_t = (-a * (ts_history**2)) + (a * ts_history) + 1
    print('use quadratic alpha(t)')
    print('alpha t type', type(alpha_t))
    print('alpha t shape', alpha_t.shape)

    ## Add slight reg to Sigma_t_history
    print('add small reg to Sigma t hist')
    Sigma_t_history += np.eye(Sigma_t_history.shape[-1]) * 1e-8
    ## dims summary: (alpha, len(dataloader), k, nt, d, d)
    tmp = np.empty((a.shape[0], *Sigma_t_history.shape), dtype=Sigma_t_history.dtype)
    tmp[:] = Sigma_t_history
    Sigma_t_history = tmp
    print('expanded sigma t shape', Sigma_t_history.shape)
    Sigma_t_history[:, :, :, :, *hidobsmask] *= alpha_t[:, :, None, :, None, None]
    Sigma_t_history[:, :, :, :, *obshidmask] *= alpha_t[:, :, None, :, None, None]
    # L_t_history = np.linalg.cholesky(Sigma_t_history)
    print('tmp and sigma t different?', np.all(Sigma_t_history == tmp))
    syms = issymmetric(Sigma_t_history, atol=1e-12)
    print('all sym', np.all(syms))
    print('num not sym', (~syms).sum())

    eigval_hist = np.linalg.eigvalsh(Sigma_t_history)
    pds = eigval_hist[..., 0] > 0
    print('pds shape', pds.shape)
    idx_a, idx_b, idx_k, idx_t = np.nonzero(~pds)
    print('not pds num nonzero', idx_a.shape[0])
    not_pds_ts = ts_history.squeeze(0)[(idx_a, idx_t)]
    print('not pds ts shape', not_pds_ts.shape)
    # print('pds ts examples', pds_ts[:10])


    A = tmp[..., *obsobsmask]
    B = tmp[..., *obshidmask]
    C = tmp[..., *hidobsmask]
    D = tmp[..., *hidhidmask]
    A_cho, lower = cho_factor(A)
    A_inv = cho_solve((A_cho, lower), np.eye(A.shape[-1]))
    print('A shape', A.shape)
    print('B shape', B.shape)
    print('C shape', C.shape)
    print('D shape', D.shape)
    CAinvB = C @ A_inv @ B
    schur = D - ((alpha_t ** 2)[:, :, None, :, None, None] * CAinvB)
    print('schur shape', schur.shape)

    return
    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols, 6*nrows))
    n_alpha = np.prod(pds.shape[1:])
    fig.suptitle(
        fr'Counts of cases (total {n_alpha} per $\alpha$) where ' \
        + r'$\hat{\Sigma}_t \nsucc 0$ per $\alpha(t) = (-a)t^2 + at + 1$',
        fontsize=36
    )
    fig.supxlabel('Sampled ts', fontsize=24)
    fig.supylabel('Counts', fontsize=24)
    for i, aa in enumerate(np.squeeze(a, (1, 2))):
        a_cond = idx_a == i
        idx_aa = idx_a[a_cond]
        idx_tt = idx_t[a_cond]
        ts_a = ts_history[0][(idx_aa, idx_tt)]
        uniqs, counts = np.unique(ts_a, return_counts=True)
        uniqs_sorted_idx = np.argsort(uniqs)
        uniqs_sorted = uniqs[uniqs_sorted_idx]
        uniqs_sorted_str = [np.format_float_positional(_x, precision=6) for _x in uniqs_sorted]
        counts_sorted = counts[uniqs_sorted_idx]

        ax = axs[*divmod(i, ncols)]
        counts_a = counts.sum()
        ax.set_title(f'a = {aa}, total = {counts_a} ({100 * counts_a / n_alpha}%)')
        ax.axhline(0, alpha=0.3, c='k')
        bars = ax.bar(uniqs_sorted_str, counts_sorted)
        ax.bar_label(bars)
        ax.set_xticks(ax.get_xticks(), uniqs_sorted_str, rotation=45)
    for j in range(i+1, nrows*ncols-1):
        ax = axs[*divmod(j, ncols)]
        ax.set_visible(False)

    ax = axs[*divmod(j+1, ncols)]
    ax.set_title(r'$\alpha(t)$')
    ax.set_xlim((-0.2, 1.2))
    # ax.set_ylim((-0.2, 2.2))
    ax.axvline(0, alpha=0.3, c='k')
    ax.axhline(0, alpha=0.3, c='k')
    alphaspan = np.linspace(0, 1, 101)
    for aa in [-4, -2, 0, 2, 4, 6]:
        alphai = (-aa) * (alphaspan ** 2) + (aa * alphaspan) + 1
        ax.plot(alphaspan, alphai, label=rf'$a = {aa}$')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, left=0.05)
    fig.savefig('pd_times.png')
    fig.savefig('pd_times.pdf')

    print('check posdef by looking at sign of smallest eigenvalue')
    print('all pd', np.all(pds))
    print('alpha coef', a.reshape(-1))
    print('num not pd per alpha', (~pds).sum(axis=(1, 2, 3)))
    print(f'percentage out of {np.prod(pds.shape[1:])}', (~pds).sum(axis=(1, 2, 3)) / np.prod(pds.shape[1:]))

    print('check schur complement pd')
    A = Sigma_t_history[..., *obsobsmask]
    B = Sigma_t_history[..., *obshidmask]
    C = Sigma_t_history[..., *hidobsmask]
    D = Sigma_t_history[..., *hidhidmask]

    print('A posdef?', np.all(np.linalg.eigvalsh(A) > 0))
    print('D posdef?', np.all(np.linalg.eigvalsh(D) > 0))

    A_cho, lower = cho_factor(A)
    A_inv = cho_solve((A_cho, lower), np.eye(A.shape[-1]))
    assert np.allclose(A @ A_inv, np.broadcast_to(np.eye(A.shape[-1]), A.shape))

    sc = D - (C @ A_inv @ B)
    print('schur comp shape', sc.shape)
    sc_eigval_hist = np.linalg.eigvalsh(sc)
    sc_pds = sc_eigval_hist[..., 0] > 0
    print('sc pds shape', sc_pds.shape)
    print('check posdef by looking at sign of smallest eigenvalue')
    print('sc all pd', np.all(sc_pds))
    print('alpha coef', a.reshape(-1))
    print('sc num not pd per alpha', (~sc_pds).sum(axis=(1, 2, 3)))
    print(f'percentage out of {np.prod(sc_pds.shape[1:])}', (~sc_pds).sum(axis=(1, 2, 3)) / np.prod(sc_pds.shape[1:]))
    # print('all pd', np.all(L_t_history[:, :, :, 0] > 0))
    # print('num not pd', (Sigma_t_history[:, :, :, 0] <= 0).sum())

    return


if __name__ == "__main__":
    main()
