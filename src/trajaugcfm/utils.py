from collections.abc import Callable, Sequence
from functools import reduce

import jaxtyping as jt
import numpy as np

from trajaugcfm.constants import (
    CONSTOBS,
    DYNOBS,
    OBS
)


def build_indexer(
    allvars: Sequence[str],
    dropvars: Sequence[str] | None=None
) -> Sequence[bool]:
    '''Returns a bool mask hiding dropvars'''
    indexer = np.ones(len(allvars), dtype=bool)
    if dropvars is not None:
        for i, varname in enumerate(allvars):
            for dropvar in dropvars:
                if varname == dropvar:
                    indexer[i] = False
    return indexer


def roundrobin_split_idxs(
    N: int,
    nfenceposts: int,
) -> jt.Int[np.ndarray, 'nfenceposts']:
    '''Return nfenceposts evenly spaced indices from 0 to N-1'''
    fps = [tmp.shape[0] for tmp in np.array_split(np.arange(N), nfenceposts-1)]
    fps[0] -= 1
    fps = [0] + fps
    return np.cumsum(fps)


def batch_interp(
    a: jt.Real[np.ndarray, '#batch *dims'],
    b: jt.Real[np.ndarray, '#batch *dims'],
    t: jt.Real[np.ndarray, 'times']
) -> jt.Real[np.ndarray, 'batch times *dims']:
    '''Interpolates a batch from a to b'''
    t_broadcast = t[None, :, *(None for _ in range(a.ndim-1))]  ## 1 times ...1
    return (t_broadcast * b[:, None]) + ((1 - t_broadcast) * a[:, None])


def batch_eigh_op(
    X: jt.Real[np.ndarray, '... n n'],
    fs: Sequence[Callable[[jt.Real[np.ndarray, '... n']], jt.Real[np.ndarray, '... n']]],
) -> jt.Real[np.ndarray, '... n']:
    '''Computes eigendecomposition of X and returns Q @ fn(fn-1(...f1(L))) @ Q_inv'''
    L, Q = np.linalg.eigh(X)
    fL = reduce(lambda arr, f: f(arr), fs, L)
    return Q @ np.apply_along_axis(np.diag, -1, fL) @ np.swapaxes(Q, -1, -2)


def batch_eigval_replace_neg(
    L: jt.Real[np.ndarray, '... n'],
    atol: float=1e-10
) -> jt.Real[np.ndarray, '... n']:
    '''Replaces negative eigvals with smallest postive eigval'''
    pos_mask = L > atol
    neg_mask = ~pos_mask
    pos_idx = np.argmax(pos_mask, axis=-1, keepdims=True)
    pos_val = np.take_along_axis(L, pos_idx, axis=-1)
    Lpos = np.where(neg_mask, pos_val, L)
    return Lpos


## TODO: make batch_eigval_replace_neg a partial func?
def batch_sqrtm(
    X: jt.Real[np.ndarray, '... n n'],
) -> jt.Real[np.ndarray, '... n n']:
    '''Takes the batch sqrt of symmetric matrices'''
    fs = [batch_eigval_replace_neg, np.sqrt]
    return batch_eigh_op(X, fs)


def batch_inv(
    X: jt.Real[np.ndarray, '... n n']
) -> jt.Real[np.ndarray, '... n n']:
    '''Takes the batch inv of symmetric matrices'''
    fs = [lambda L: 1 / L]
    return batch_eigh_op(X, fs)


## TODO: make batch_eigval_replace_neg a partial func?
def batch_inv_sqrtm(
    X: jt.Real[np.ndarray, '... n n'],
) -> jt.Real[np.ndarray, '... n n']:
    '''Takes the batch inv sqrt of symmetric matrices'''
    fs = [batch_eigval_replace_neg, lambda L: 1 / L, np.sqrt]
    return batch_eigh_op(X, fs)


def main():
    obsidxs = [0, 2, 5]
    obsmask = np.zeros(8, dtype=bool)
    obsmask[obsidxs] = True
    hidmask = ~obsmask

    x = np.arange(8) ** 2
    print(obsidxs)
    print(x)
    print(x[obsmask])
    print(x[hidmask])
    # redundant = [dynvar for dynvar in DYNOBS if '_IF' in dynvar]
    # indexer = build_indexer(OBS, dropvars=CONSTOBS+redundant)  # type: ignore
    # i2 = ~indexer
    # print(indexer)
    # print(i2)


if __name__ == "__main__":
    main()
