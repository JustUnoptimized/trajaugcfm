import os
from itertools import product
from time import time
from typing import Self

import jaxtyping as jt
import numpy as np
from scipy.linalg import issymmetric
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF as RBFKernel
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


## TODO: Clean up function annotations and other comments
class TrajAugCFMSampler(IterableDataset):
    def __init__(
        self,
        Xs:          jt.Real[np.ndarray, 'N margidx dims'],
        Xrefs:       jt.Real[np.ndarray, 'Nrefs T refdims'],
        obsmask:     list[bool],
        tidxs:       list[int],
        k:           int,
        n:           int,
        b:           int,
        nt:          int,
        rbfk_scale:  float=1.,
        rbfk_bounds: tuple[int|float, int|float]=(0.05, 5),
        gpr_nt:      int=10,
        rbfd_scale:  float=1.,
        reg:         float=1e-8,
        seed:        int | None=None,
        ## FIX(GIF): feature flags for Guided Isotropic Flow
        fixgif: bool=False,
        fixgif_sigma: str='bb',
        fixgif_sigma_scale: float=1.0,
        fixgif_sigma_eps: float=0.0,
        ## Time sampling options
        time_sample: str='uniform',
        time_beta_a: float=2.0,
        ## Score target construction
        score_gauss: bool=False,
    ) -> None:
        '''Builds sampler for Anisotropic per-sample conditional flow.

        Is a subclass of torch.utils.data.IterableDataset so can be
        passed into a torch.utils.data.DataLoader. This class already batches
        so the DataLoader should be created with the kwarg batch_size=None.

        Batch size is k * b * nt.
        Can be an iterable where a full iteration one cycle through the Xrefs
        Currently only implemented for augmentation via a GP regression with an RBF kernel.

        FIX(GIF): When `fixgif=True`, switches the target path to a Guided
        Isotropic Flow with isotropic covariance schedule and drift
        u(t,x)=mu'(t)+[sigma'(t)/sigma(t)](x-mu(t)).

        Args:
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
            gpr_nt:      Number of time points into ref used for GPR
            rbfd_scale:  Scale for RBF distance when resampling batch conditional on ref
            reg:         Regularization to prevent singular matrices
            seed:        NumPy Generator seed for reproducibility
            fixgif:      Enable Guided Isotropic Flow drift/target when True
            fixgif_sigma: 'const' or 'bb' (Brownian-bridge) sigma schedule
            fixgif_sigma_scale: base sigma scale for schedule
        '''
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

        ## Regularization
        self.reg = reg
        self.obsreg = np.eye(self.nobs)[None, None, ...] * self.reg
        self.hidreg = np.eye(self.nhid)[None, None, ...] * self.reg

        ## RBF params for endpoint sampling
        self.rbfd_scale = rbfd_scale
        self.rbfd_denom = - 2 * (rbfd_scale ** 2)

        ## Gaussian Process Regression params
        self.rbfk_scale = rbfk_scale
        self.rbfk_bounds = rbfk_bounds
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

        ## Reproducability
        self.prng = np.random.default_rng(seed=seed)

        ## FIX(GIF): store feature flag + schedule config
        self.fixgif = fixgif
        self.fixgif_sigma = fixgif_sigma
        self.fixgif_sigma_scale = fixgif_sigma_scale
        self.fixgif_sigma_eps = fixgif_sigma_eps
        self.time_sample = time_sample
        self.time_beta_a = time_beta_a
        self.score_gauss = score_gauss

    def __len__(self) -> int:
        return self._len

    def _precompute_gprs(self) -> list[GaussianProcessRegressor]:
        '''Pre-compute GPRs on Xrefs using a RBF Kernel'''
        gprs = [
            GaussianProcessRegressor(
                kernel=RBFKernel(
                    length_scale=self.rbfk_scale,
                    length_scale_bounds=self.rbfk_bounds
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
        '''Sample minibatch from each marginal snapshot

        First select indep indices for each marginal i using choice()
        where len(indices) == b.
        Then stack Xs[indices, i] in the time dimension.
        '''
        return np.stack(
            [self.Xs[self.prng.choice(self.Xs.shape[0], size=self.n), i] \
             for i in range(self.Xs.shape[1])],
            axis=1
        )

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
        # RBFs = np.zeros((self.b, k, xs.shape[1]))
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
        centered = z - mus  ## (k, b, margidx, dims)
        covs = np.einsum('kbti,kbtj->ktij', centered, centered)
        covs /= z.shape[1] - 1  ## (k, margidx, dims, dims)
        mus = np.squeeze(mus, axis=1)  ## (k, margidx, dims)
        return mus, covs

    def _compute_mu_t_sigma_t_gpr(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
        ts: jt.Real[np.ndarray, 'nt']
    ) -> tuple[jt.Real[np.ndarray, 'k nt obs'], jt.Real[np.ndarray, 'k nt obs']]:
        '''Compute mu_t and sigma_t from GPRs'''
        mu_t_gpr = np.zeros((refidxs.shape[0], ts.shape[0], self.Xrefs.shape[-1]))
        sigma_t_gpr = np.zeros_like(mu_t_gpr)  ## (k, nt, obs)
        ts = ts.reshape((-1, 1))  ## (nt, 1)
        for i, idx in enumerate(refidxs):
            mu_i, std_i = self.gprs[idx].predict(ts, return_std=True)
            mu_t_gpr[i] = mu_i
            sigma_t_gpr[i] = std_i
        return mu_t_gpr, sigma_t_gpr

    def _compute_interpolants(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims'],
    ) -> tuple[jt.Real[np.ndarray, 'k nt dims'], jt.Real[np.ndarray, 'k dims dims'], jt.Real[np.ndarray, 'k nt dims dims']]:
        r'''Compute interpolants for W2 geodesic between MVNs

        \mu_t = t \mu_1 + (1 - t) \mu_0
        C = \Sigma_1^{1/2} (\Sigma_1^{1/2} \Sigma_0 \Sigma_1^{1/2})^{-1/2} \Sigma_1^{1/2}
        C_t = tC + (1 - t)I
        \Sigma_t = C_t \Sigma_0 C_t

        Beware numerical errors resulting in non-symmetric matrices!
        '''
        mu_t = batch_interp(mus[:, 0], mus[:, 1], ts)         ## (k, nt, dims)
        Sigma_1_sqrt = batch_sqrtm(covs[:, 1])                ## (k, dims, dims)
        Sigma_101 = Sigma_1_sqrt @ covs[:, 0] @ Sigma_1_sqrt
        ## Regularize to avoid bad matrix
        Sigma_101 += np.eye(Sigma_101.shape[-1])[None, ...] * self.reg
        Sigma_101_inv_sqrt = batch_inv_sqrtm(Sigma_101)
        C = Sigma_1_sqrt @ Sigma_101_inv_sqrt @ Sigma_1_sqrt  ## (k, dims, dims)
        I = np.eye(C.shape[-1])[None, ...]                    ## (1, dims, dims)
        C_t = batch_interp(I, C, ts)                          ## (k, nt, dims, dims)
        Sigma_t = C_t @ covs[:, 0][:, None] @ C_t             ## (k, nt, dims, dims)
        return mu_t, C, Sigma_t

    ## FIX(GIF): linear mean-only interpolation (no W2)
    def _compute_mu_t_linear(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        '''Compute mu_t = (1-t) mu_0 + t mu_1 for each k without W2 terms'''
        return batch_interp(mus[:, 0], mus[:, 1], ts)

    ## FIX(GIF): isotropic sigma schedule and derivative
    def _sigma_schedule(
        self,
        ts: jt.Real[np.ndarray, 'nt']
    ) -> tuple[jt.Real[np.ndarray, 'nt'], jt.Real[np.ndarray, 'nt']]:
        '''Return sigma_t and sigma_t_prime arrays for given times ts

        Schedules:
        - 'const': sigma(t) = c; sigma'(t) = 0
        - 'bb'   : Brownian-bridge style sigma^2(t) = c^2 t (1 - t)
        '''
        t = np.clip(ts, 1e-4, 1-1e-4)
        c = float(self.fixgif_sigma_scale)
        if self.fixgif_sigma == 'const':
            sigma_t = np.full_like(t, fill_value=c, dtype=float)
            sigma_t_prime = np.zeros_like(t, dtype=float)
        else:  ## 'bb'
            # sigma(t) = c * sqrt(eps + t(1-t))
            q = self.fixgif_sigma_eps + t * (1 - t)
            sigma_t = c * np.sqrt(q)
            # derivative: d/dt sqrt(q) = (1/(2 sqrt(q))) * (1 - 2t)
            sigma_t_prime = c * (0.5 / np.sqrt(q)) * (1 - 2*t)
        return sigma_t, sigma_t_prime

    ## FIX(GIF): isotropic sampling
    def _sample_xt_isotropic(
        self,
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        sigma_t: jt.Real[np.ndarray, 'nt']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Sample x_t ~ N(mu_t, sigma(t)^2 I) with shared sigma across dims'''
        k, nt, d = mu_t.shape
        eps = self.prng.normal(size=(k, self.b, nt, d))
        sigma = sigma_t.reshape((1, 1, nt, 1))
        mu = mu_t[:, None, :, :]  ## (k, 1, nt, d)
        return mu + sigma * eps

    ## FIX(GIF): isotropic drift
    def _compute_ut_isotropic(
        self,
        xt: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        sigma_t: jt.Real[np.ndarray, 'nt'],
        sigma_t_prime: jt.Real[np.ndarray, 'nt']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''u(t,x) = mu'(t) + (sigma'(t)/sigma(t)) (x - mu(t))'''
        ratio = (sigma_t_prime / np.maximum(sigma_t, 1e-12)).reshape((1, 1, -1, 1))
        return mu_t_prime[:, None] + ratio * (xt - mu_t[:, None])

    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: jt.Real[np.ndarray, 'k nt dims dims'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Sample xt

        First sample xt_obs using mu_t and sigma_t from GPR
        Then compute conditional mu_t_hid|obs and Sigma_t_hid|obs
        Use conditional params to sample xt_hid
        Return xt = (xt_obs, xt_hid)

        Beware numerical errors resulting in non-symmetric matrices!
        '''
        xt = np.zeros((refidxs.shape[0], self.b, *mu_t.shape[1:]))        ## (k, b, nt, dims)

        ## First compute x_t_obs from mu_gp, sigma_gp
        eps = self.prng.normal(size=(*xt.shape[:3], mu_t_gpr.shape[-1]))  ## (k, b, nt, obs)
        ## Assume sigma_gp is list of stddevs
        ## ==> ref vars have 0 covariance == indep
        xt_obs = (sigma_t_gpr[:, None] * eps) + mu_t_gpr[:, None]         ## (k, b, nt, obs)

        ## Next compute conditional mu_t_hid|obs and sigma_t_hid|obs
        Sigma_t_hidobs = Sigma_t[:, :, *self.hidobsmask]                  ## (k, nt, hid, obs)
        Sigma_t_obsobs_inv = Sigma_t[:, :, *self.obsobsmask]              ## (k, nt, obs, obs)
        Sigma_t_obsobs_inv = batch_inv(Sigma_t_obsobs_inv)                ## (k, nt, obs, obs)
        B = Sigma_t_hidobs @ Sigma_t_obsobs_inv                           ## (k, nt, hid, obs)

        obs_diff = xt_obs - mu_t[:, None, :, self.obsmask]                ## (k, b, nt, obs)
        cond_mu_t = B[:, None] @ obs_diff[:, :, :, :, None]               ## (k, b, nt, hid, 1)
        cond_mu_t = cond_mu_t.squeeze(axis=4)                             ## (k, b, nt, hid)

        cond_Sigma_t = B @ Sigma_t[:, :, *self.obshidmask]
        cond_Sigma_t = Sigma_t[:, :, *self.hidhidmask] - cond_Sigma_t     ## (k, nt, hid, hid)
        cond_Sigma_t += np.eye(cond_Sigma_t.shape[-1])[None, ...] * self.reg

        cond_A_t = batch_sqrtm(cond_Sigma_t)                              ## (k, nt, hid, hid)

        ## Sample xt_hid|obs
        eps = self.prng.normal(size=cond_mu_t.shape)                      ## (k, b, nt, hid)
        ## xt_hid vars have nonzero covariance
        xt_hid = cond_A_t[:, None] @ eps[:, :, :, :, None]                ## (k, b, nt, hid, 1)
        xt_hid = xt_hid.squeeze(axis=4) + cond_mu_t                       ## (k, b, nt, hid)

        xt[:, :, :, self.obsmask] = xt_obs
        xt[:, :, :, self.hidmask] = xt_hid
        return xt

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

        if isinstance(self.gprs[0].kernel_, RBFKernel):
            ## dKstar_dt = rbf(xtrain, xtest) * (xtrain - xtest) // ell^2
            Scales = np.zeros((refidxs.shape[0]))
            Kstar = np.zeros((refidxs.shape[0], self.gpr_nt, ts.shape[0]))
            for i, idx in enumerate(refidxs):
                gpr = self.gprs[idx]
                kernel = gpr.kernel_
                Xtrain[i] = gpr.X_train_
                Alpha[i] = gpr.alpha_
                Scales[i] = kernel.length_scale
                Kstar[i] = kernel(Xtrain[i], ts)
            chainrule_mult = (Xtrain - ts.T[None, ...])          ## (k, gpr_nt, nt)
            chainrule_mult /= (Scales ** 2)[:, None, None]
            Kstar *= chainrule_mult                              ## (k, gpr_nt, nt)
            dmu_dt = Kstar.swapaxes(1, 2) @ Alpha                ## (k, nt, obs)
            return dmu_dt
        else:
            raise ValueError(f'dmu_dt not implemented for GPR with kernel {self.gprs[0].kernel_}')

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

    def _compute_A_t_prime(
        self,
        L_0_sqrt: jt.Real[np.ndarray, 'k dims'],
        Q_0: jt.Real[np.ndarray, 'k dims dims'],
        C: jt.Real[np.ndarray, 'k dims dims']
    ) -> jt.Real[np.ndarray, 'k dims dims']:
        r'''Compute A_t_prime = (C - I) @ Q_0 \Lambda_0^{1/2}

        Derived from Sigma_t = C_t @ Sigma_0 @ C_t
        where Sigma_0 = Q_0 @ \Lambda_0 @ Q_0
        which implies A_t = C_t @ Q_0 \ Lambda_0^{1/2}
        '''
        I = np.eye(self.dim)[None, ...]
        A_t_prime = (C - I) @ Q_0 @ np.apply_along_axis(np.diag, -1, L_0_sqrt)
        return A_t_prime

    def _compute_A_t_inv(
        self,
        L_0_sqrt: jt.Real[np.ndarray, 'k dims'],
        Q_0: jt.Real[np.ndarray, 'k dims dims'],
        C: jt.Real[np.ndarray, 'k dims dims'],
        ts: jt.Real[np.ndarray, 'nt']
    ) -> jt.Real[np.ndarray, 'k nt dims dims']:
        r'''Compute A_t_inv = Q_C (t \Lambda_C + (1 - t)I)^{-1} Q_C^{-1}

        Derived from Sigma_t = C_t @ Sigma_0 @ C_t
        where Sigma_0 = Q_0 @ \Lambda_0 @ Q_0
        which implies A_t = C_t @ Q_0 \ Lambda_0^{1/2}
        '''
        L_0_inv_sqrt = 1 / L_0_sqrt
        Q_0_T = Q_0.swapaxes(1, 2)
        L_C, Q_C = np.linalg.eigh(C)
        I = np.ones(L_C.shape[-1]).reshape((1, -1))
        L_C_t = batch_interp(I, L_C, ts)
        C_t_inv = Q_C[:, None] @ np.apply_along_axis(np.diag, -1, 1 / L_C_t) @ Q_C[:, None].swapaxes(-1, -2)
        A_t_inv = np.apply_along_axis(np.diag, -1, L_0_inv_sqrt)[:, None] @ Q_0_T[:, None] @ C_t_inv
        return A_t_inv

    def _compute_ut(
        self,
        xt: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime: jt.Real[np.ndarray, 'k dims dims'],
        A_t_inv: jt.Real[np.ndarray, 'k nt dims dims']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Compute ut = A_t_prime @ A_t_inv (xt - mu_t) + mu_t_prime'''
        xt_diff = xt - mu_t[:, None]                                     ## (k, b, nt, dims)
        A_t_prime_inv = A_t_prime[:, None] @ A_t_inv                     ## (k, nt, dims, dims)
        A_t_xtdiff = A_t_prime_inv[:, None] @ xt_diff[:, :, :, :, None]  ## (k, b, nt, dims, 1)
        A_t_xtdiff = A_t_xtdiff.squeeze(axis=4)                          ## (k, b, nt, dims)
        ut = A_t_xtdiff + mu_t_prime[:, None]                            ## (k, b, nt, dims)
        return ut

    def __iter__(self) -> Self:
        '''First resets iteration state, then returns self'''
        self._iteridx = -1  ## -1 instead of 0 because next() increments first
        self.prng.shuffle(self._Xrefidxs)
        return self

    def __next__(self) -> tuple[jt.Float32[Tensor, 'batch 1'], jt.Float32[Tensor, 'batch dims'], jt.Float32[Tensor, 'batch dims']]:
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

            ## Sample t according to chosen law
            if self.time_sample == 'uniform':
                ts = self.prng.random(size=self.nt)
            else:  ## symmetric Beta(a,a)
                a = float(self.time_beta_a)
                ts = self.prng.beta(a, a, size=self.nt)

            if self.fixgif:
                ## FIX(GIF) path: isotropic sampling and drift
                mu_t_linear = self._compute_mu_t_linear(ts, mus)
                mu_t_gpr, _sigma_obs = self._compute_mu_t_sigma_t_gpr(refidxs, ts)
                mu_t_aug = self._compute_mu_t_aug(mu_t_linear, mu_t_gpr)
                mu_t_aug_prime = self._compute_mu_t_aug_prime(refidxs, ts, mus)
                sigma_t, sigma_t_prime = self._sigma_schedule(ts)
                xt = self._sample_xt_isotropic(mu_t_aug, sigma_t)
                ut = self._compute_ut_isotropic(xt, mu_t_aug, mu_t_aug_prime, sigma_t, sigma_t_prime)

                if self.score_gauss:
                    ## Build low-rank+diag Gaussian score target using Woodbury
                    # U per k: dims x r, with r = nobs; rows: [I_obs; B], B = Sigma_hid,obs Sigma_obs,obs^{-1}
                    r = int(self.nobs)
                    U = np.zeros((refidxs.shape[0], self.dim, r))
                    # Average cov over marginals for stability
                    cov_avg = mus.shape[1]  # unused; keep to show intention
                    for i in range(refidxs.shape[0]):
                        Sigma = covs[i].mean(axis=0)  # (dims, dims)
                        S_oo = Sigma[*self.obsobsmask]
                        S_ho = Sigma[*self.hidobsmask]
                        S_oo = S_oo + (np.eye(self.nobs) * self.reg)
                        S_oo_inv = np.linalg.inv(S_oo)
                        B_map = S_ho @ S_oo_inv  # (nhid, nobs)
                        U[i, self.obsmask, :] = np.eye(self.nobs)
                        U[i, self.hidmask, :] = B_map

                    # Precompute P = U^T U per k over dim
                    # U shape: (k, dim, r) -> P shape: (k, r, r)
                    P = np.einsum('kdr,kds->krs', U, U)
                    # Expand quantities to compute s = -alpha ( y - U z ),
                    # with z = (I + alpha P)^{-1} (alpha U^T y)
                    y = xt - mu_t_aug[:, None]  # (k, b, nt, dims)
                    # alpha(t) = 1 / sigma(t)^2
                    alpha = 1.0 / (np.maximum(sigma_t, 1e-12) ** 2)  # (nt,)
                    # Compute U^T y: (k, r, dims) x (k, b, nt, dims) -> (k, b, nt, r)
                    # UTy: (k, r, dim) x (k, b, nt, dim) -> (k, b, nt, r)
                    UTy = np.einsum('krd,kbnd->kbnr', U.swapaxes(1, 2), y)
                    # Build A = I + alpha P for each (k, nt)
                    I_r = np.eye(r)
                    # Solve z for each (k, b, nt)
                    z = np.zeros_like(UTy)
                    for i in range(refidxs.shape[0]):
                        for t_idx in range(self.nt):
                            A = I_r + (alpha[t_idx]) * P[i]  # (r, r)
                            A_inv = np.linalg.inv(A)
                            z[i, :, t_idx] = (alpha[t_idx]) * (UTy[i, :, t_idx] @ A_inv.T)
                    # Uy: (k, dim, r) x (k, b, nt, r) -> (k, b, nt, dim)
                    Uy = np.einsum('kdr,kbnr->kbnd', U, z)
                    s_target = -alpha[None, None, :, None] * (y - Uy)
                else:
                    s_target = None
            else:
                ## Original W2 + GP augmentation path (unchanged)
                ## Add reg separately for obs and hid in case they are not contiguous
                covs += np.eye(self.dim)[None, None] * self.reg
                L_0, Q_0 = np.linalg.eigh(covs[:, 0])
                L_0_sqrt = np.sqrt(L_0)

                mu_t, C, Sigma_t = self._compute_interpolants(ts, mus, covs)
                ## Add reg separately for obs and hid in case they are not contiguous
                Sigma_t += np.eye(self.dim)[None, None] * self.reg
                mu_t_gpr, sigma_t_gpr = self._compute_mu_t_sigma_t_gpr(refidxs, ts)
                xt = self._sample_xt(
                    refidxs, mu_t, Sigma_t, mu_t_gpr, sigma_t_gpr
                )
                mu_t_aug = self._compute_mu_t_aug(mu_t, mu_t_gpr)
                mu_t_aug_prime = self._compute_mu_t_aug_prime(refidxs, ts, mus)
                A_t_prime = self._compute_A_t_prime(L_0_sqrt, Q_0, C)
                A_t_inv = self._compute_A_t_inv(L_0_sqrt, Q_0, C, ts)
                ut = self._compute_ut(xt, mu_t_aug, mu_t_aug_prime, A_t_prime, A_t_inv)

            ## Flatten and cast into Tensors of shape (k*b*nt, dims)
            ## Also cast to float32 for compatibility with default torch float operations
            ts = np.broadcast_to(ts[None, None, :, None], (*xt.shape[:-1], 1))
            ts_t = torch.from_numpy(ts.reshape((-1, 1)).astype(np.float32))
            xt_t = torch.from_numpy(xt.reshape((-1, xt.shape[-1])).astype(np.float32))
            ut_t = torch.from_numpy(ut.reshape((-1, ut.shape[-1])).astype(np.float32))
            if self.fixgif and self.score_gauss and s_target is not None:
                st_t = torch.from_numpy(s_target.reshape((-1, s_target.shape[-1])).astype(np.float32))
                return ts_t, xt_t, ut_t, st_t
            else:
                return ts_t, xt_t, ut_t
            # return (refidxs, xs, z, mus, covs, ts, xt, gpr_dmu_dt,
            #         mu_t, C, Sigma_t, mu_t_gpr, sigma_t_gpr,
            #         L_0_sqrt, Q_0,
            #         mu_t_aug, mu_t_aug_prime, A_t_prime, A_t_inv, ut)

        else:
            raise StopIteration


class TrajAugCFMSamplerForLoop(IterableDataset):
    def __init__(
        self,
        Xs:          jt.Real[np.ndarray, 'N margidx dims'],
        Xrefs:       jt.Real[np.ndarray, 'Nrefs T refdims'],
        obsmask:     list[bool],
        tidxs:       list[int],
        k:           int,
        n:           int,
        b:           int,
        nt:          int,
        rbfk_scale:  float=1.,
        rbfk_bounds: tuple[int|float, int|float]=(0.05, 5),
        gpr_nt:      int=10,
        rbfd_scale:  float=1.,
        reg:         float=1e-8,
        seed:        int | None=None,
    ) -> None:
        '''Builds sampler for Anisotropic per-sample conditional flow.

        Is a subclass of torch.utils.data.IterableDataset so can be
        passed into a torch.utils.data.DataLoader. This class already batches
        so the DataLoader should be created with the kwarg batch_size=None.

        Batch size is k * b * nt.
        Can be an iterable where a full iteration one cycle through the Xrefs
        Currently only implemented for augmentation via a GP regression with an RBF kernel.

        Args:
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
            gpr_nt:      Number of time points into ref used for GPR
            rbfd_scale:  Scale for RBF distance when resampling batch conditional on ref
            reg:         Regularization to prevent singular matrices
            seed:        NumPy Generator seed for reproducibility
        '''
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

        ## Regularization
        self.reg = reg
        self.obsreg = np.eye(self.nobs)[None, None, ...] * self.reg
        self.hidreg = np.eye(self.nhid)[None, None, ...] * self.reg

        ## RBF params for endpoint sampling
        self.rbfd_scale = rbfd_scale
        self.rbfd_denom = - 2 * (rbfd_scale ** 2)

        ## Gaussian Process Regression params
        self.rbfk_scale = rbfk_scale
        self.rbfk_bounds = rbfk_bounds
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

        ## Reproducability
        self.prng = np.random.default_rng(seed=seed)

    def __len__(self) -> int:
        return self._len

    def _precompute_gprs(self) -> list[GaussianProcessRegressor]:
        '''Pre-compute GPRs on Xrefs using a RBF Kernel'''
        gprs = [
            GaussianProcessRegressor(
                kernel=RBFKernel(
                    length_scale=self.rbfk_scale,
                    length_scale_bounds=self.rbfk_bounds
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
        '''Sample minibatch from each marginal snapshot

        First select indep indices for each marginal i using choice()
        where len(indices) == n.
        Then stack Xs[indices, i] in the time dimension.
        '''
        return np.stack(
            [self.Xs[self.prng.choice(self.Xs.shape[0], size=self.n), i] \
             for i in range(self.Xs.shape[1])],
            axis=1
        )

    def _sample_z_given_refs(
        self,
        xs: jt.Real[np.ndarray, 'n margidx dims'],
        refs: jt.Real[np.ndarray, 'k T refdims']
    ) -> jt.Real[np.ndarray, 'k b margidx dims']:
        r'''Samples z = (x_0, ..., x_M) from prod_i^M pi(x_i | ref)

        Uses the RBF distance from ref as the unnormalized probabilities.
        Sampling is vectorized using a discretized version of inverse transform sampling.
        '''
        xobs = xs[:, :, self.obsmask]
        k = refs.shape[0]
        RBFs = np.zeros((k, self.n, xs.shape[1]))

        ## Get prob tensor based on RBF dist
        for i in range(xs.shape[1]):
            RBFs[:, :, i] = cdist(
                refs[:, self.tidxs[i]], xobs[:, i], metric='sqeuclidean'
            )
        RBFs /= self.rbfd_denom
        RBFs = np.exp(RBFs)
        normconst = np.sum(RBFs, axis=1, keepdims=True)  ## shape (k, 1, xs.shape[1])
        ## each RBF[i, :, j] should be a vector of probs into xs at snapshot j cond on ref i
        RBFs /= normconst

        RBFs_cumsum = np.cumsum(RBFs, axis=1)
        u = self.prng.random((k, self.b, xs.shape[1]))
        z = np.zeros((k, self.b, xs.shape[1], xs.shape[2]))
        for i, j in product(range(k), range(xs.shape[1])):
            idxs = np.searchsorted(RBFs_cumsum[i, :, j], u[i, :, j])
            z[i, :, j] = xs[idxs, j]
        return z

    def _compute_marginal_mu_sigma(
        self,
        z: jt.Real[np.ndarray, 'k b margidx dims'],
    ) -> tuple[jt.Real[np.ndarray, 'k margidx dims'], jt.Real[np.ndarray, 'k margidx dims dims']]:
        r'''Compute \mu and \Sigma based on sampled z

        \Sigma has slight diagonal regularization for numerical stability.
        '''
        mus = z.mean(axis=1)
        covs = np.zeros((self.k, z.shape[2], z.shape[3], z.shape[3]))
        for i, j in product(range(self.k), range(z.shape[2])):
            covs[i, j] = np.cov(z[i, :, j], rowvar=False)
        return mus, covs

    def _compute_mu_t_sigma_t_gpr(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
        ts: jt.Real[np.ndarray, 'nt']
    ) -> tuple[jt.Real[np.ndarray, 'k nt obs'], jt.Real[np.ndarray, 'k nt obs']]:
        r'''Compute \mu_t and \sigma_t from GPRs'''
        mu_t_gpr = np.zeros((refidxs.shape[0], ts.shape[0], self.Xrefs.shape[-1]))
        sigma_t_gpr = np.zeros_like(mu_t_gpr)  ## (k, nt, obs)
        ts = ts.reshape((-1, 1))  ## (nt, 1)
        for i, idx in enumerate(refidxs):
            mu_i, std_i = self.gprs[idx].predict(ts, return_std=True)
            mu_t_gpr[i] = mu_i
            sigma_t_gpr[i] = std_i
        return mu_t_gpr, sigma_t_gpr

    def _compute_interpolants(
        self,
        ts: jt.Real[np.ndarray, 'nt'],
        mus: jt.Real[np.ndarray, 'k margidx dims'],
        covs: jt.Real[np.ndarray, 'k margidx dims dims'],
    ) -> tuple[jt.Real[np.ndarray, 'k nt dims'], jt.Real[np.ndarray, 'k dims dims'], jt.Real[np.ndarray, 'k nt dims dims']]:
        r'''Compute interpolants for W2 geodesic between MVNs

        \mu_t = t \mu_1 + (1 - t) \mu_0
        C = \Sigma_1^{1/2} (\Sigma_1^{1/2} \Sigma_0 \Sigma_1^{1/2})^{-1/2} \Sigma_1^{1/2}
        C_t = tC + (1 - t)I
        \Sigma_t = C_t \Sigma_0 C_t

        Beware numerical errors resulting in non-symmetric matrices!
        '''
        mu_t = np.zeros((self.k, self.nt, self.dim))
        C = np.zeros((self.k, self.dim, self.dim))
        Sigma_t = np.zeros((self.k, self.nt, self.dim, self.dim))
        I = np.eye(self.dim)
        for i in range(self.k):
            L_1, Q_1 = np.linalg.eigh(covs[i, 1])
            Sigma_1_sqrt = Q_1 @ np.diag(np.sqrt(L_1)) @ Q_1.T
            Sigma_101 = Sigma_1_sqrt @ covs[i, 0] @ Sigma_1_sqrt
            Sigma_101 += I * self.reg
            L_101, Q_101 = np.linalg.eigh(Sigma_101)
            L_101_inv_sqrt = 1 / np.sqrt(L_101)
            Sigma_101_inv_sqrt = Q_101 @ np.diag(L_101_inv_sqrt) @ Q_101.T
            C[i] = Sigma_1_sqrt @ Sigma_101_inv_sqrt @ Sigma_1_sqrt
            for j in range(self.nt):
                mu_t[i, j] = (ts[j] * mus[i, 1]) + ((1 - ts[j]) * mus[i, 0])
                C_t = (ts[j] * C[i]) + ((1 - ts[j]) * I)
                Sigma_t[i, j] = C_t @ covs[i, 0] @ C_t
        return mu_t, C, Sigma_t

    def _sample_xt(
        self,
        refidxs: jt.Int[np.ndarray, 'k'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        Sigma_t: jt.Real[np.ndarray, 'k nt dims dims'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
        sigma_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        r'''Sample xt

        First sample xt_obs using \mu_t and \sigma_t from GPR
        Then compute conditional \mu_t_hid|obs and \Sigma_t_hid|obs
        Use conditional params to sample xt_hid
        Return xt = (xt_obs, xt_hid)

        Beware numerical errors resulting in non-symmetric matrices!
        '''
        xt = np.zeros((refidxs.shape[0], self.b, *mu_t.shape[1:]))  ## (k, b, nt, dims)
        eps = self.prng.normal(size=(*xt.shape[:3], mu_t_gpr.shape[-1]))  ## (k, b, nt, obs)
        xt_obs = np.zeros_like(eps)
        for i, j in product(range(self.k), range(self.b)):
            xt_obs[i, j] = (sigma_t_gpr[i] * eps[i, j]) + mu_t_gpr[i]

        Sigma_t_hidobs = Sigma_t[:, :, *self.hidobsmask]                  ## (k, nt, hid, obs)
        Sigma_t_obsobs_inv = Sigma_t[:, :, *self.obsobsmask]              ## (k, nt, obs, obs)
        Sigma_t_obsobs_inv = batch_inv(Sigma_t_obsobs_inv)                ## (k, nt, obs, obs)
        B = np.zeros_like(Sigma_t_hidobs)
        for i, j in product(range(self.k), range(self.nt)):
            B[i, j] = Sigma_t_hidobs[i, j] @ Sigma_t_obsobs_inv[i, j]

        obs_diff = np.zeros_like(xt_obs)
        for i, j in product(range(self.k), range(self.b)):
            obs_diff[i, j] = xt_obs[i, j] - mu_t[i][:, self.obsmask]

        cond_mu_t = np.zeros((self.k, self.b, self.nt, self.nhid))
        for i, j, ell in product(range(self.k), range(self.b), range(self.nt)):
            cond_mu_t[i, j, ell] = B[i, ell] @ obs_diff[i, j, ell]

        cond_Sigma_t = np.zeros((self.k, self.nt, self.nhid, self.nhid))
        for i, j in product(range(self.k), range(self.nt)):
            cond_Sigma_t[i, j] = B[i, j] @ Sigma_t[i, j][*self.obshidmask]
            cond_Sigma_t[i, j] = Sigma_t[i, j][*self.hidhidmask] - cond_Sigma_t[i, j]
            cond_Sigma_t[i, j] += np.eye(cond_Sigma_t.shape[-1]) * self.reg

        cond_A_t = batch_sqrtm(cond_Sigma_t)

        eps = self.prng.normal(size=cond_mu_t.shape)
        xt_hid = np.zeros_like(cond_mu_t)
        for i, j, ell in product(range(self.k), range(self.b), range(self.nt)):
            xt_hid[i, j, ell] = cond_A_t[i, ell] @ eps[i, j, ell]
        xt_hid += cond_mu_t

        xt[:, :, :, self.obsmask] = xt_obs
        xt[:, :, :, self.hidmask] = xt_hid
        return xt

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

        if isinstance(self.gprs[0].kernel_, RBFKernel):
            dmu_dt = np.zeros((self.k, self.nt, self.nobs))
            for i, idx in enumerate(refidxs):
                gpr = self.gprs[idx]
                kernel = gpr.kernel_
                Xtrain = gpr.X_train_
                alpha = gpr.alpha_
                scale = kernel.length_scale
                kstar = kernel(Xtrain, ts)
                chainrule_mult = np.zeros((self.gpr_nt, self.nt))
                for j, ell in product(range(self.gpr_nt), range(self.nt)):
                    chainrule_mult[j, ell] = Xtrain[j, 0] - ts[ell, 0]
                chainrule_mult /= (scale ** 2)
                dkstar = kstar * chainrule_mult
                dmu_dt[i] = dkstar.T @ alpha
            return dmu_dt
        else:
            raise ValueError(f'dmu_dt not implemented for GPR with kernel {self.gprs[0].kernel_}')

    def _compute_mu_t_aug(
        self,
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        mu_t_gpr: jt.Real[np.ndarray, 'k nt obs'],
    ) -> jt.Real[np.ndarray, 'k nt dims']:
        r'''Compute \mu_t augmented with ref data

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

    def _compute_A_t_prime(
        self,
        L_0_sqrt: jt.Real[np.ndarray, 'k dims'],
        Q_0: jt.Real[np.ndarray, 'k dims dims'],
        C: jt.Real[np.ndarray, 'k dims dims']
    ) -> jt.Real[np.ndarray, 'k dims dims']:
        r'''Compute A_t_prime = (C - I) @ Q_0 \Lambda_0^{1/2}

        Derived from Sigma_t = C_t @ Sigma_0 @ C_t
        where Sigma_0 = Q_0 @ \Lambda_0 @ Q_0
        which implies A_t = C_t @ Q_0 \ Lambda_0^{1/2}
        '''
        I = np.eye(self.dim)
        A_t_prime = np.zeros_like(C)
        for i in range(self.k):
            A_t_prime[i] = (C[i] - I) @ Q_0[i] @ np.diag(L_0_sqrt[i])
        # I = np.eye(C.shape[-1])[None, ...]
        # A_t_prime = (C - I) @ Q_0 @ np.apply_along_axis(np.diag, -1, L_0_sqrt)
        return A_t_prime

    def _compute_A_t_inv(
        self,
        L_0_sqrt: jt.Real[np.ndarray, 'k dims'],
        Q_0: jt.Real[np.ndarray, 'k dims dims'],
        C: jt.Real[np.ndarray, 'k dims dims'],
        ts: jt.Real[np.ndarray, 'nt']
    ) -> jt.Real[np.ndarray, 'k nt dims dims']:
        r'''Compute A_t_inv = Q_C (t \Lambda_C + (1 - t)I)^{-1} Q_C^{-1}

        Derived from Sigma_t = C_t @ Sigma_0 @ C_t
        where Sigma_0 = Q_0 @ \Lambda_0 @ Q_0
        which implies A_t = C_t @ Q_0 \ Lambda_0^{1/2}
        '''
        A_t_inv = np.zeros((self.k, self.nt, self.dim, self.dim))
        I = np.ones(self.dim)
        for i in range(self.k):
            L_0_inv_sqrt = 1 / L_0_sqrt[i]
            L_C, Q_C = np.linalg.eigh(C[i])
            for j in range(self.nt):
                L_C_t = (ts[j] * L_C) + ((1 - ts[j]) * I)
                L_C_inv_t = 1 / L_C_t
                C_t_inv = Q_C @ np.diag(L_C_inv_t) @ Q_C.T
                A_t_inv[i, j] = np.diag(L_0_inv_sqrt) @ Q_0[i].T @ C_t_inv
        # L_0_inv_sqrt = 1 / L_0_sqrt
        # Q_0_T = Q_0.swapaxes(1, 2)
        # L_C, Q_C = np.linalg.eigh(C)
        # I = np.ones(L_C.shape[-1]).reshape((1, -1))
        # L_C_t = batch_interp(I, L_C, ts)
        # C_t_inv = Q_C[:, None] @ np.apply_along_axis(np.diag, -1, 1 / L_C_t) @ Q_C[:, None].swapaxes(-1, -2)
        # A_t_inv = np.apply_along_axis(np.diag, -1, L_0_inv_sqrt)[:, None] @ Q_0_T[:, None] @ C_t_inv
        return A_t_inv

    def _compute_ut(
        self,
        xt: jt.Real[np.ndarray, 'k b nt dims'],
        mu_t: jt.Real[np.ndarray, 'k nt dims'],
        mu_t_prime: jt.Real[np.ndarray, 'k nt dims'],
        A_t_prime: jt.Real[np.ndarray, 'k dims dims'],
        A_t_inv: jt.Real[np.ndarray, 'k nt dims dims']
    ) -> jt.Real[np.ndarray, 'k b nt dims']:
        '''Compute ut = A_t_prime @ A_t_inv (xt - mu_t) + mu_t_prime'''
        ut = np.zeros_like(xt)
        for i, ell in product(range(self.k), range(self.nt)):
            A_t_prime_inv = A_t_prime[i] @ A_t_inv[i, ell]
            for j in range(self.b):
                xt_diff = xt[i, j, ell] - mu_t[i, j]
                A_t_xtdiff = A_t_prime_inv @ xt_diff
                ut[i, j, ell] = A_t_xtdiff + mu_t_prime[i, ell]
        # xt_diff = xt - mu_t[:, None]                                     ## (k, b, nt, dims)
        # A_t_prime_inv = A_t_prime[:, None] @ A_t_inv                     ## (k, nt, dims, dims)
        # A_t_xtdiff = A_t_prime_inv[:, None] @ xt_diff[:, :, :, :, None]  ## (k, b, nt, dims, 1)
        # A_t_xtdiff = A_t_xtdiff.squeeze(axis=4)                          ## (k, b, nt, dims)
        # ut = A_t_xtdiff + mu_t_prime[:, None]                            ## (k, b, nt, dims)
        return ut

    def __iter__(self) -> Self:
        '''First resets iteration state, then returns self'''
        self._iteridx = -1  ## -1 instead of 0 because next() increments first
        self.prng.shuffle(self._Xrefidxs)
        return self

    def __next__(self) -> tuple[jt.Float32[Tensor, 'batch 1'], jt.Float32[Tensor, 'batch dims'], jt.Float32[Tensor, 'batch dims']]:
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
            ## Add reg separately for obs and hid in case they are not contiguous
            covs[:, :, *self.obsobsmask] += self.obsreg
            covs[:, :, *self.hidhidmask] += self.hidreg
            L_0, Q_0 = np.linalg.eigh(covs[:, 0])
            L_0_sqrt = np.sqrt(L_0)

            ## Sample t \sim U(0, 1)
            ts = self.prng.random(size=self.nt)

            ## Sample xt
            mu_t, C, Sigma_t = self._compute_interpolants(ts, mus, covs)
            ## Add reg separately for obs and hid in case they are not contiguous
            Sigma_t[:, :, *self.obsobsmask] += self.obsreg
            Sigma_t[:, :, *self.hidhidmask] += self.hidreg
            mu_t_gpr, sigma_t_gpr = self._compute_mu_t_sigma_t_gpr(refidxs, ts)
            xt = self._sample_xt(
                refidxs, mu_t, Sigma_t, mu_t_gpr, sigma_t_gpr
            )

            gpr_dmu_dt = self._compute_gpr_dmudt(refidxs, ts)
            ## Compute ut
            mu_t_aug = self._compute_mu_t_aug(mu_t, mu_t_gpr)
            mu_t_aug_prime = self._compute_mu_t_aug_prime(refidxs, ts, mus)
            A_t_prime = self._compute_A_t_prime(L_0_sqrt, Q_0, C)
            A_t_inv = self._compute_A_t_inv(L_0_sqrt, Q_0, C, ts)
            ut = self._compute_ut(xt, mu_t_aug, mu_t_aug_prime, A_t_prime, A_t_inv)

            ## Flatten and cast into Tensors of shape (k*b*nt, dims)
            ## Also cast to float32 for compatibility with default torch float operations
            ts = np.broadcast_to(ts[None, None, :, None], (*xt.shape[:-1], 1))
            ts = torch.from_numpy(ts.reshape((-1, 1)).astype(np.float32))
            xt = torch.from_numpy(xt.reshape((-1, xt.shape[-1])).astype(np.float32))
            ut = torch.from_numpy(ut.reshape((-1, ut.shape[-1])).astype(np.float32))

            return ts, xt, ut
            # return (refidxs, xs, z, mus, covs, ts, xt, gpr_dmu_dt,
            #         mu_t, C, Sigma_t, mu_t_gpr, sigma_t_gpr,
            #         L_0_sqrt, Q_0,
            #         mu_t_aug, mu_t_aug_prime, A_t_prime, A_t_inv, ut)

        else:
            raise StopIteration


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
    # data = data[:, :, :, len(CONSTOBS):]  ## only keep feats that change
    # dmso = data[0]

    # obsidxs = [0, 1, 2]
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

    k = 4
    n = 128
    b = 8
    nt = 10
    # k = 1
    # b = 1
    # nt = 1
    rbfk_scale = 0.1
    rbfk_bounds = (0.05, 5)
    gpr_nt = 10
    rbfd_scale = 1.
    reg = 1e-8
    seed = 1000
    sampler = TrajAugCFMSampler(
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
        gpr_nt=gpr_nt,
        rbfd_scale=rbfd_scale,
        reg=reg,
        seed=seed,
    )
    for_sampler = TrajAugCFMSamplerForLoop(
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
        gpr_nt=gpr_nt,
        rbfd_scale=rbfd_scale,
        reg=reg,
        seed=seed,
    )

    batch_size = None
    loader = DataLoader(sampler, batch_size=batch_size)
    for_loader = DataLoader(for_sampler, batch_size=batch_size)
    # print(len(loader))
    # print(len(for_loader))
    diffs = np.zeros((len(sampler), 3))
    for i, (a, b) in enumerate(zip(loader, for_loader)):
        for j in range(3):
            diffs[i, j] = (a[j] - b[j]).detach().cpu().numpy().mean()
        # print('compare refidxs', torch.all(a[0] - b[0] == 0).item())
        # print('compare xs', torch.all(a[1] - b[1] == 0).item())
        # print('compare z', torch.all(a[2] - b[2] == 0).item())
        # print('compare mus', torch.all(a[3] - b[3] == 0).item())
        # print('compare covs', torch.all(a[4] - b[4] == 0).item())
        # print('compare covs diff', torch.sum(a[4] - b[4] == 0).item())
        # print('compare covs close', torch.allclose(a[4], b[4]))
        # print('compare ts', torch.all(a[5] - b[5] == 0).item())
        # print('compare xt', torch.all(a[6] - b[6] == 0).item())
        # print('compare xt close', torch.allclose(a[6], b[6]))
        # print('compare gprdmudt', torch.all(a[7] - b[7] == 0).item())
        # print('compare gprdmudt close', torch.allclose(a[7], b[7]))
#
        # print('compare mu_t', torch.all(a[8] - b[8] == 0).item())
        # print('compare C', torch.all(a[9] - b[9] == 0).item())
        # print('compare C close', torch.allclose(a[9], b[9]))
        # print('compare Sigma_t', torch.all(a[10] - b[10] == 0).item())
        # print('compare Sigma_t close', torch.allclose(a[10], b[10]))
        # print('compare mu_t_gpr', torch.all(a[11] - b[11] == 0).item())
        # print('compare sigma_t_gpr', torch.all(a[12] - b[12] == 0).item())
#
        # print('compare L_0_sqrt', torch.all(a[13] - b[13] == 0).item())
        # print('compare L_0_sqrt close', torch.allclose(a[13], b[13]))
        # print('compare Q_0', torch.all(a[14] - b[14] == 0).item())
        # print('compare Q_0 close', torch.allclose(a[14], b[14]))
#
        # print('compare mu_t_aug', torch.all(a[15] - b[15] == 0).item())
        # print('compare mu_t_aug_prime', torch.all(a[16] - b[16] == 0).item())
        # print('compare A_t_prime', torch.all(a[17] - b[17] == 0).item())
        # print('compare A_t_prime close', torch.allclose(a[17], b[17]))
        # print('compare A_t_inv', torch.all(a[18] - b[18] == 0).item())
        # print('compare A_t_inv close', torch.allclose(a[18], b[18]))
        # print('compare ut', torch.all(a[19] - b[19] == 0).item())
        # print('compare ut close', torch.allclose(a[19], b[19]))
        break
    print(diffs)
    # print(torch.allclose(a[14][0] @ a[14][0].T, torch.eye(11, dtype=torch.float64)))
    # print(torch.allclose(b[14][0] @ b[14][0].T, torch.eye(11, dtype=torch.float64)))
    return


if __name__ == "__main__":
    main()
