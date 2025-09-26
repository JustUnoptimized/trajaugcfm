from abc import ABC, abstractmethod
import os
from itertools import product
from time import time
from typing import (
    Literal,
    overload,
    Self
)

import jaxtyping as jt
import numpy as np
from scipy.linalg import issymmetric
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF as RBFKernel,
    WhiteKernel
)
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from trajaugcfm.utils import (
    batch_sqrtm,
    batch_interp,
    batch_inv,
    batch_inv_sqrtm,
    build_indexer,
    roundrobin_split_idxs
)

## TYPING DECLARATIONS
type RBFK_Bounds = tuple[int | float, int | float] | Literal['fixed']
type Times = jt.Float64[np.ndarray, 'nt rff_dims*2'] \
             | jt.Float64[np.ndarray, 'nt 1']
type Sigma_T = jt.Float64[np.ndarray, 'k nt dims dims'] \
               | jt.Float64[np.ndarray, 'nt'] \
               | float
type Aux = jt.Float64[np.ndarray, 'k dims dims'] \
           | jt.Float64[np.ndarray, 'nt'] \
           | None
type A_T_Prime_A_T_Inv = jt.Float64[np.ndarray, 'k nt dims dims'] \
                         | jt.Float[np.ndarray, 'nt'] \
                         | None
type GCFMBatch = tuple[
    jt.Float32[Tensor, 'batch rff_dims*2'] | jt.Float32['batch 1'],  ## ts
    jt.Float32[Tensor, 'batch dims'],                                ## xt
    jt.Float32[Tensor, 'batch dims'],                                ## ut
    jt.Float32[Tensor, 'batch dims'],                                ## eps
    jt.Float32[Tensor, 'batch dims dims'] | None,                    ## lt
]


## TODO: How to parallellize? Lots of numpy matrix operations is slow for single thread...
class GCFMSamplerBase(IterableDataset):
    def __init__(
        self,
        prng:        np.random.Generator,
        Xs:          jt.Real[np.ndarray, 'N margidx dims'],
        Xrefs:       jt.Real[np.ndarray, 'Nrefs T refdims'],
        obsmask:     list[bool],
        tidxs:       list[int],
        k:           int,
        n:           int,
        b:           int,
        nt:          int,
        rbfk_scale:  float=0.1,
        rbfk_bounds: RBFK_Bounds=(0.05, 5),
        whitenoise:  float=0.1,
        gpr_nt:      int=8,
        rbfd_scale:  float=1.,
        reg:         float=1e-8,
        sigma:       float=1.0,
        sb_reg:      float=1e-8,
        beta_a:      float=2.0,
        rff_seed:    int=2000,
        rff_scale:   float=1.0,
        rff_dim:     int=1,
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
        self.nobs = self.obsmask.sum()
        self.nhid = self.hidmask.sum()
        self.dim = self.Xs.shape[-1]

        ## Time sampler params (Currently Beta only)
        self.beta_a = beta_a

        ## Time RFF enrichment fixed features (ignored if not enhancing time)
        # self.B = self.prng.normal(loc=0, scale=rff_scale, size=(1, rff_dim))
        self.B = np.random.default_rng(seed=rff_seed).normal(loc=0, scale=rff_scale, size=(1, rff_dim))
        ## pre-scale by 2pi to avoid recomputation in _enrich_ts()
        self.B *= 2 * np.pi

        ## Variance schedule (IFMixins only)
        self.sigma = sigma

        ## Regularization
        self.reg = reg
        self.obsreg = np.eye(self.nobs)[None, None, ...] * self.reg
        self.hidreg = np.eye(self.nhid)[None, None, ...] * self.reg
        self.sb_reg = sb_reg

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
        self._len, r = divmod(Xrefs.shape[0], k)
        if r > 0:
            self._len += 1
        self._sentinel = self._len - 1  ## needed for some indexing issues
        self._iteridx = 0
        self._Xrefidxs = np.arange(Xrefs.shape[0]).astype(int)

    def __len__(self) -> int:
        return self._len

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
        '''Sample minibatch from each marginal snapshot'''
        idxs = self.prng.integers(self.Xs.shape[0], size=(self.n, self.Xs.shape[1]))
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
        refidxs: jt.Int[np.ndarray, 'k'],
        ts: jt.Real[np.ndarray, 'nt']
    ) -> tuple[jt.Real[np.ndarray, 'k nt obs'], jt.Real[np.ndarray, 'k nt obs']]:
        '''Compute mu_t and sigma_t from GPRs'''
        mu_t_gpr = np.zeros((refidxs.shape[0], ts.shape[0], self.Xrefs.shape[-1]))
        sigma_t_gpr = np.zeros_like(mu_t_gpr)  ## (k, nt, obs)
        ts = ts.reshape((-1, 1))               ## (nt, 1)
        for i, idx in enumerate(refidxs):
            mu_i, std_i = self.gprs[idx].predict(ts, return_std=True)
            mu_t_gpr[i] = mu_i
            sigma_t_gpr[i] = std_i
        return mu_t_gpr, sigma_t_gpr

    def _compute_gpr_dmudt(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
        ts: jt.Real[np.ndarray, 'nt']
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

        ## dKstar_dt = rbf(xtrain, xtest) * (xtrain - xtest) // ell^2
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

    def _compute_mu_t_aug(
        self,
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        '''Compute mu_t augmented with ref data

        mu_t_aug_obs = mu_t_gpr
        mu_t_aug_hid = t mu_1 + (1 - t) mu_0
        mu_t_aug = (mu_t_aug_obs, mu_t_aug_hid)
        '''
        mu_t_aug = np.zeros_like(mu_t)
        mu_t_aug[:, :, self.obsmask] = mu_t_gpr
        mu_t_aug[:, :, self.hidmask] = mu_t[:, :, self.hidmask]
        return mu_t_aug

    def _compute_mu_t_aug_prime(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
        ts: jt.Real[np.ndarray, 'nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims']
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        r'''Compute \mu_t_prime augmented with ref data

        mu_t_aug_prime_obs = mu_t_gpr_prime
        mu_t_aug_prime_hid = mu_1 - mu_0
        mu_t_aug_prime = (mu_t_aug_prime_obs, mu_t_aug_prime_hid)
        '''
        mu_t_aug_prime = np.zeros((refidxs.shape[0], ts.shape[0], mus.shape[-1]))
        mu_t_gpr_prime = self._compute_gpr_dmudt(refidxs, ts)
        mu_t_aug_prime[:, :, self.obsmask] = mu_t_gpr_prime
        mu_t_hid_prime = mus[:, 1, self.hidmask] - mus[:, 0, self.hidmask]
        mu_t_aug_prime[:, :, self.hidmask] = mu_t_hid_prime[:, None]
        return mu_t_aug_prime

    def _compute_xt_diff(
        self,
        xt: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t: jt.Real[np.ndarray, 'k nt dims']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Convenience method to compute xt - mu_t'''
        return xt - mu_t[:, None]

    ## All mixin methods, ordered by call order in __next__()
    ## Time Sampling Mixin Method Signatures
    @abstractmethod
    def _sample_ts(self) -> jt.Float64[np.ndarray, 'nt']:
        '''Samples batch of times using TimeMixin'''
        raise NotImplementedError

    ## TimeRFFMixin
    @overload
    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, 'nt']
    ) -> jt.Float64[np.ndarray, 'nt rff_dim*2']:
        ...

    ## TimeNoEnrichMixin
    @overload
    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, 'nt']
    ) -> jt.Float64[np.ndarray, 'nt 1']:
        ...

    @abstractmethod
    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, 'nt']
    ) -> Times:
        raise NotImplementedError

    ## Flow Matching Mixin Method Signatures
    @abstractmethod
    def _compute_mu_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        raise NotImplementedError

    ## AFMixin
    @overload
    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[jt.Real[np.ndarray, 'k nt dims dims'], jt.Real[np.ndarray, 'k dims dims']]:
        ...

    ## IFCBMixin
    @overload
    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[float, None]:
        ...

    ## IFSBMixin
    @overload
    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[jt.Real[np.ndarray, 'nt'], jt.Real[np.ndarray, 'nt']]:
        ...

    @abstractmethod
    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[Sigma_T, Aux]:
        raise NotImplementedError

    ## AFMixin
    @overload
    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
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
        refidxs: jt.Int[np.ndarray, 'k'],
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
        refidxs: jt.Int[np.ndarray, 'k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: jt.Real[np.ndarray, 'nt'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        eps: jt.Real[np.ndarray, 'k b nt dims'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        ...

    @abstractmethod
    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
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
        ts: jt.Real[np.ndarray, 'nt'],
        aux: jt.Real[np.ndarray, 'k dims dims']
    ) -> jt.Real[np.ndarray, 'k nt dims dims']:
        ...

    ## IFCBMixin
    @overload
    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        aux: None
    ) -> None:
        ...

    ## IFSBMixin
    @overload
    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        aux: jt.Real[np.ndarray, 'nt']
    ) -> jt.Real[np.ndarray, 'nt']:
        ...

    @abstractmethod
    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
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
        A_t_prime_A_t_inv: jt.Real[np.ndarray, 'nt']
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
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> None:
        ...

    ## ASMixin
    @overload
    def _compute_lambda(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> jt.Float64[np.ndarray, 'k nt dims dims']:
        ...

    @abstractmethod
    def _compute_lambda(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> jt.Float64[np.ndarray, 'k nt dims dims'] | None:
        raise NotImplementedError

    def __iter__(self) -> Self:
        '''First resets iteration state, then returns self'''
        self._iteridx = -1  ## -1 instead of 0 because next() increments first
        self.prng.shuffle(self._Xrefidxs)
        return self

    def __next__(self) -> GCFMBatch:
        if self._iteridx < self._sentinel:
            self._iteridx += 1
            ## Hacky iteration to get idxs[i:i+k]
            ## TODO: use internal bound generator rather than this clunky expr?
            refidxs = self._Xrefidxs[self._iteridx*self.k:(self._iteridx+1)*self.k]
            ## Sample k refs
            refs = self.Xrefs[refidxs]

            ## Independently sample minibatch
            xs = self._get_xs_minibatch()

            ## Resample according to refs
            z = self._sample_z_given_refs(xs, refs)
            mus, covs = self._compute_marginal_mu_sigma(z)

            ## Sample t according to chosen distribution
            ts = self._sample_ts()

            batch_shape = (refidxs.shape[0], self.b, self.nt)

            ## Main algorithm
            mu_t = self._compute_mu_t(ts, mus)
            Sigma_t, aux = self._compute_sigma_t(ts, covs)
            mu_t_gpr, sigma_t_gpr = self._compute_mu_t_sigma_t_gpr(refidxs, ts)
            eps = self.prng.normal(size=(*batch_shape, self.dim))
            xt = self._sample_xt(refidxs, mu_t, Sigma_t, mu_t_gpr, sigma_t_gpr, eps)
            mu_t_aug = self._compute_mu_t_aug(mu_t, mu_t_gpr)
            mu_t_aug_prime = self._compute_mu_t_aug_prime(refidxs, ts, mus)
            xt_diff = self._compute_xt_diff(xt, mu_t_aug)
            A_t_prime_A_t_inv = self._compute_A_t_prime_A_t_inv(ts, aux)
            ut = self._compute_ut(xt_diff, mu_t_aug_prime, A_t_prime_A_t_inv)
            lt = self._compute_lambda(ts, covs)

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

        else:
            raise StopIteration


class UniformTimeMixin:
    '''Samples batch of times from Unif(0, 1)'''

    def _sample_ts(self) -> jt.Float64[np.ndarray, 'nt']:
        return self.prng.random(size=self.nt)


class BetaTimeMixin:
    '''Samples batch of times from Beta(a, a)'''

    def _sample_ts(self) -> jt.Float64[np.ndarray, 'nt']:
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
        ts: jt.Float64[np.ndarray, 'nt']
    ) -> jt.Float64[np.ndarray, 'nt rff_dim*2']:
        Bt = self.B * ts[:, None]  ## (nt, rff_dim)
        cosBt = np.cos(Bt)
        sinBt = np.sin(Bt)
        return np.concatenate((cosBt, sinBt), axis=1)


class TimeNoEnrichMixin:
    '''Dummy class with no time enrichment'''

    def _enrich_ts(
        self,
        ts: jt.Float64[np.ndarray, 'nt']
    ) -> jt.Float64[np.ndarray, 'nt 1']:
        return ts[:, None]


## CURRENTLY DOES NOT CONVERGE!
class AFMixin:
    '''Anisotropic Flow Mixin

    All methods are coupled!
    '''

    def _compute_mu_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        r'''Compute mu_t for W2 geodesic between MVNs

        \mu_t = t \mu_1 + (1 - t) \mu_0
        '''
        return batch_interp(mus[:, 0], mus[:, 1], ts)         ## (k, nt, dims)

    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[jt.Real[np.ndarray, 'k nt dims dims'], jt.Real[np.ndarray, 'k dims dims']]:
        r'''Compute Sigma_t for W2 geodesic between MVNs

        C = \Sigma_1^{1/2} (\Sigma_1^{1/2} \Sigma_0 \Sigma_1^{1/2})^{-1/2} \Sigma_1^{1/2}
        C_t = tC + (1 - t)I
        \Sigma_t = C_t \Sigma_0 C_t

        Beware numerical errors resulting in non-symmetric matrices!
        '''
        Sigma_1_sqrt = batch_sqrtm(covs[:, 1])                ## (k, dims, dims)
        Sigma_101 = Sigma_1_sqrt @ covs[:, 0] @ Sigma_1_sqrt
        ## Regularize to avoid bad matrix
        Sigma_101 += np.eye(Sigma_101.shape[-1])[None, ...] * self.reg
        Sigma_101_inv_sqrt = batch_inv_sqrtm(Sigma_101)
        C = Sigma_1_sqrt @ Sigma_101_inv_sqrt @ Sigma_1_sqrt  ## (k, dims, dims)
        I = np.eye(C.shape[-1])[None, ...]                    ## (1, dims, dims)
        C_t = batch_interp(I, C, ts)                          ## (k, nt, dims, dims)
        Sigma_t = C_t @ covs[:, 0][:, None] @ C_t             ## (k, nt, dims, dims)
        return Sigma_t, C

    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: jt.Real[np.ndarray, 'k nt dims dims'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
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
        xt_obs = (sigma_t_gpr[:, None] * eps[:, :, :, self.obsmask]) + mu_t_gpr[:, None]

        ## Next compute conditional mu_t_hid|obs and sigma_t_hid|obs
        Sigma_t_hidobs = Sigma_t[:, :, *self.hidobsmask]                  ## (k, nt, hid, obs)
        Sigma_t_obsobs = Sigma_t[:, :, *self.obsobsmask]              ## (k, nt, obs, obs)
        Sigma_t_obsobs_inv = batch_inv(Sigma_t_obsobs)                ## (k, nt, obs, obs)
        B = Sigma_t_hidobs @ Sigma_t_obsobs_inv                           ## (k, nt, hid, obs)

        obs_diff = xt_obs - mu_t[:, None, :, self.obsmask]                ## (k, b, nt, obs)
        cond_mu_t = np.matvec(B[:, None], obs_diff)                       ## (k, b, nt, hid)

        cond_Sigma_t = B @ Sigma_t[:, :, *self.obshidmask]
        cond_Sigma_t = Sigma_t[:, :, *self.hidhidmask] - cond_Sigma_t     ## (k, nt, hid, hid)
        cond_Sigma_t += np.eye(cond_Sigma_t.shape[-1])[None, ...] * self.reg

        cond_A_t = batch_sqrtm(cond_Sigma_t)                              ## (k, nt, hid, hid)

        ## Sample xt_hid|obs
        ## xt_hid vars have nonzero covariance
        xt_hid = np.matvec(cond_A_t[:, None], eps[:, :, :, self.hidmask]) ## (k, b, nt, hid)

        xt[:, :, :, self.obsmask] = xt_obs
        xt[:, :, :, self.hidmask] = xt_hid
        return xt

    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        aux: jt.Real[np.ndarray, 'k dims dims']
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
        I = np.eye(self.dim)
        C = aux
        C_t_prime = C - I[None, ...]

        L_C, Q_C = np.linalg.eigh(C)
        L_C_t_inv = 1 / batch_interp(np.ones(self.dim)[None, ... ], L_C, ts)
        C_t_inv = Q_C[:, None] \
                  @ np.apply_along_axis(np.diag, -1, L_C_t_inv) \
                  @ Q_C[:, None].swapaxes(-1, -2)
        return C_t_prime[:, None, ...] @ C_t_inv

    def _compute_ut(
        self,
        xt_diff: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime_A_t_inv: jt.Real[np.ndarray, 'k nt dims dims']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Compute ut = A_t_prime @ A_t_inv (xt - mu_t) + mu_t_prime'''
        Ax = np.matvec(A_t_prime_A_t_inv[:, None], xt_diff)  ## (k, b, nt, dims)
        return Ax + mu_t_prime[:, None]


class IFCBMixin:
    '''Isotropic Flow Constant Bridge Mixin

    All exposed methods are coupled!
    '''

    def _compute_mu_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        '''Compute mu_t of constant bridge'''
        return batch_interp(mus[:, 0], mus[:, 1], ts)  ## (k, nt, dims)

    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[float, None]:
        '''Compute sigma_t of constant bridge

        Args ts, covs are ignored. Only kept for consistent method signature.'''
        del ts, covs

        return self.sigma, None

    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
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

        k = mu_t.shape[0]
        xt = np.zeros_like(eps)
        sigma_eps = Sigma_t * eps
        xt[:, :, :, self.obsmask] = sigma_eps[:, :, :, self.obsmask] \
                                    + mu_t_gpr[:, None]
        xt[:, :, :, self.hidmask] = sigma_eps[:, :, :, self.hidmask] \
                                    + mu_t[:, None, :, self.hidmask]
        return xt

    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
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
        ts: jt.Real[np.ndarray, 'nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        '''Compute mu_t of Schrodinger bridge'''
        return batch_interp(mus[:, 0], mus[:, 1], ts)  ## (k, nt, dims)

    def _compute_sigma_t(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims']
    ) -> tuple[jt.Real[np.ndarray, 'nt'], jt.Real[np.ndarray, 'nt']]:
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
        refidxs: jt.Int[np.ndarray, 'k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: jt.Real[np.ndarray, 'nt'],
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

        k = mu_t.shape[0]
        xt = np.zeros_like(eps)
        sigma_eps = Sigma_t[None, None, :, None] * eps
        xt[:, :, :, self.obsmask] = sigma_eps[:, :, :, self.obsmask] \
                                    + mu_t_gpr[:, None]
        xt[:, :, :, self.hidmask] = sigma_eps[:, :, :, self.hidmask] \
                                    + mu_t[:, None, :, self.hidmask]
        return xt

    def _compute_A_t_prime_A_t_inv(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        aux: jt.Real[np.ndarray, 'nt']
    ) -> jt.Real[np.ndarray, 'nt']:
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
        A_t_prime_A_t_inv: jt.Real[np.ndarray, 'nt']
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
        ts: jt.Real[np.ndarray, 'nt'],
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
        I = np.eye(C.shape[-1])[None, ...]                    ## (1, dims, dims)
        C_t = batch_interp(I, C, ts)                          ## (k, nt, dims, dims)
        Sigma_t = C_t @ covs[:, 0][:, None] @ C_t             ## (k, nt, dims, dims)
        ## Regularize to avoid bad matrix
        Sigma_t += np.eye(self.dim)[None, None, ...] * self.reg
        return 2 * batch_sqrtm(Sigma_t) / (self.sigma ** 2)


class NSMixin:
    '''No Score Mixin'''

    def _compute_lambda(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
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
    from torch.utils.data import DataLoader

    from trajaugcfm.constants import (
        CONSTOBS,
        DYNOBS,
        OBS,
        DATADIR
    )


    experiment = 'mix_ics'
    data = np.load(os.path.join(DATADIR, experiment, 'data.npy'))  ## (drugcombs, N, T, *dims)
    dynmask = build_indexer(OBS, dropvars=CONSTOBS)
    data = data[:, :, :, dynmask]

    dyn_if_vars = [dynvar for dynvar in DYNOBS if '_IF' in dynvar]
    dynifmask = build_indexer(DYNOBS, dropvars=dyn_if_vars)
    data = data[:, :, :, dynifmask]
    dmso = data[0]

    obsmask = np.zeros(data.shape[-1], dtype=bool)
    obsidxs = [0, 1, 2]
    obsmask[obsidxs] = True
    hidmask = ~obsmask
    tidxs = [0, 400]
    # _nsplit = 250
    _nsplit = 50

    Xrefs = dmso[:_nsplit][:, :, obsmask]
    Xs = dmso[_nsplit:, tidxs]
    Xobs = Xs[:, :, obsmask]
    print('Xobs shape', Xobs.shape)
    print('Xs shape', Xs.shape)
    print('Xrefs shape', Xrefs.shape)

    seed = 1000
    prng = np.random.default_rng(seed=seed)
    k = 2
    n = 16
    b = 4
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

    GCFMSampler = build_sampler_class(
        time_sampler='uniform',
        use_time_enrich=True,
        time_enrich='rff',
        flow='isotropic',
        flow_bridge='schrodinger',
        score=True,
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
        rff_dim=rff_dim
    )
    print(gcfm_sampler.get_mixin_names())

    batch_size = None
    gcfm_loader = DataLoader(gcfm_sampler, batch_size=batch_size)
    for i, (ts, xt, ut, eps, lt) in enumerate(gcfm_loader):
        print('ts shape', ts.shape)
        print('xt shape', xt.shape)
        print('ut shape', ut.shape)
        print('eps shape', eps.shape)
        if lt is not None:
            print('lt shape', lt.shape)
        break
    return


if __name__ == "__main__":
    main()
