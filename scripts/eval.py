import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Literal

import jaxtyping as jt
import numpy as np
import ot
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from tqdm import trange

from trajaugcfm.constants import (
    BASEDIR,
    DATADIR,
    RESDIR,
    CONSTOBS,
    DYNOBS,
    OBS,
)

from trajaugcfm.utils import (
    build_indexer,
)
from script_utils import (
    METRICS_FILENAME,
    MODEL_FILENAME,
    TRAINARGS_FILENAME,
    TRAJGENARGS_FILENAME,
    TRAJGEN_FILENAME,
    EVALARGS_FILENAME,
    EVALS_FILENAME,
    int_or_float,
    load_args,
    load_scalers,
    scale_data_with_scalers
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='eval')
    expgroup = parser.add_argument_group('exp', 'experiment load args')
    expgroup.add_argument(
        '--expname', type=str, required=True,
        help='Load experiment in results/<expname>/.'
    )

    metricgroup = parser.add_argument_group('metric', 'metric computation args')
    metricgroup.add_argument(
        '--sinkhorn-method', type=str, default='sinkhorn',
        choices=['sinkhorn', 'sinkhorn_log', 'sinkhorn_stabilized'],
        help='Sinkhorn solver for computing entropic EMD.'
    )
    metricgroup.add_argument(
        '--reg', type=float, default=0.01,
        help='Regularization for entropic EMD'
    )

    return parser.parse_args()


def chk_fmt_args(args: argparse.Namespace) -> argparse.Namespace:
    ## expgroup check
    exppath = os.path.join(RESDIR, args.expname)
    assert os.path.exists(exppath), f'{exppath} not found'
    args.expname = exppath

    ## metricgroup check
    assert args.reg > 0, f'reg must be positive but got {args.reg}'

    return args


def save_eval_args(args: dict[str, Any], expname: str) -> None:
    '''Save args for eval instance to json file.'''
    evalargs_path = os.path.join(expname, EVALARGS_FILENAME)
    with open(evalargs_path, 'w') as f:
        json.dump(vars(args), f, indent=4)


def compute_pointwise_metrics(
    y: jt.Float64[np.ndarray, 'N T d'],
    yhat: jt.Float64[np.ndarray, 'N T d']
) -> dict[str, jt.Float64[np.ndarray, 'N T']]:
    '''Compute evaluation metrics over y and yhat.

    y is the ground truth trajectory.
    yhat is the inferred trajectory.

    Currently implemented metrics:
        RMSE = || y - yhat || / sqrt(N)
        Cos  =  < y , yhat > / (|| y || * || yhat ||)

    Metrics are computed for each sample and time point
    '''
    N, T, d = y.shape
    pw_metrics = {}
    pw_metrics['RMSE'] = np.linalg.norm(y - yhat, axis=2) / np.sqrt(d)

    ydotyhat = np.linalg.vecdot(y, yhat, axis=2)
    ynorm = np.linalg.norm(y, axis=2)
    yhatnorm = np.linalg.norm(yhat, axis=2)
    pw_metrics['Cosine Similarity'] = ydotyhat / (ynorm * yhatnorm)

    return pw_metrics


def compute_distribution_metrics(
    y: jt.Float64[np.ndarray, 'N T d'],
    yhat: jt.Float64[np.ndarray, 'N T d'],
    reg: float,
    method: Literal['sinkhorn', 'sinkhorn_log', 'sinkhorn_stabilized'],
) -> dict[str, jt.Float64[np.ndarray, 'T']]:
    '''Compute distributional distance metrics over y and yhat.

    y is the ground truth trajectory.
    yhat is the inferred trajectory.

    Currently implemented metrics:
        Earth Movers Distance (OT distance)
        Entropic EMD

    Metrics are computed for each time point
    '''
    N, T, d = y.shape
    dist_metrics = {}
    emds = np.empty(T)
    sinkhorns = np.empty(T)

    a = np.full((N,), 1./N)
    b = a.copy()
    for i in trange(T, desc='Computing distributional distances'):
        M = cdist(y[:, i], yhat[:, i], metric='sqeuclidean')
        emds[i] = ot.emd2(a, b, M)
        sinkhorns[i] = ot.sinkhorn2(a, b, M, reg, method=method)

    assert np.all(~np.isnan(emds))
    assert np.all(~np.isnan(sinkhorns))
    assert np.all(~np.isinf(emds))
    assert np.all(~np.isinf(sinkhorns))

    dist_metrics['EMD'] = emds
    dist_metrics['Entropic EMD'] = sinkhorns
    return dist_metrics


def main() -> None:
    args = parse_args()
    args = chk_fmt_args(args)
    exp_args = load_args(args.expname, TRAINARGS_FILENAME)
    inf_args = load_args(args.expname, TRAJGENARGS_FILENAME)
    save_eval_args(args, args.expname)

    print('Recreating experiment data...\nLoading experiment data...')
    data = np.load(exp_args.data)
    dynmask = build_indexer(OBS, dropvars=CONSTOBS)
    data = data[:, :, :, dynmask]

    dyn_if_vars = [dynvar for dynvar in DYNOBS if '_IF' in dynvar]
    dynifmask = build_indexer(DYNOBS, dropvars=dyn_if_vars)
    data = data[:, :, :, dynifmask]
    data = data[exp_args.drugcombidx]

    obsmask = np.zeros(data.shape[-1], dtype=bool)
    obsmask[exp_args.obsidxs] = True
    hidmask = ~obsmask
    tidxs = [0, -1]
    dobs = obsmask.sum()
    dhid = hidmask.sum()
    d = dobs + dhid

    print('\nSplitting into train-val sets for snapshots and references')
    data_train, data_val = train_test_split(
        data, train_size=exp_args.trainsize,
        random_state=exp_args.seed if exp_args.seed is None else exp_args.seed+2
    )
    print('data train shape', data_train.shape)
    data_train_snapshots, data_train_refs = train_test_split(
        data_train, test_size=exp_args.refsize,
        random_state=exp_args.seed if exp_args.seed is None else exp_args.seed+3
    )
    data_train_snapshots = data_train_snapshots[:, tidxs]
    data_train_refs = data_train_refs[:, :, obsmask]
    print('data train snapshots shape', data_train_snapshots.shape)
    print('data train refs shape', data_train_refs.shape)

    print('data val shape', data_val.shape)
    data_val_snapshots, data_val_refs = train_test_split(
        data_val, test_size=exp_args.refsize,
        random_state=exp_args.seed if exp_args.seed is None else exp_args.seed+4
    )
    data_val_refs_hid = data_val_refs[:, :, hidmask]
    data_val_refs = data_val_refs[:, :, obsmask]
    print('data val snapshots shape', data_val_snapshots.shape)
    print('data val refs shape', data_val_refs.shape)

    print('\nLoading scalers...')
    obs_scaler, hid_scaler = load_scalers(args.expname)
    print('obs mean', obs_scaler.mean_)
    print('obs var', obs_scaler.var_)
    print('hid mean', hid_scaler.mean_)
    print('hid var', hid_scaler.var_)

    print('\nScaling data using train split...')
    (
        data_train_snapshots_scaled,
        data_train_refs_scaled,
        data_val_snapshots_scaled,
        data_val_refs_scaled
    ) = scale_data_with_scalers(
        data_train_snapshots,
        data_train_refs,
        data_val_snapshots,
        data_val_refs,
        obsmask,
        hidmask,
        obs_scaler,
        hid_scaler,
    )

    print('data val snapshots scaled shape', data_val_snapshots_scaled.shape)

    print('Loading saved trajectories...')
    trajs = np.load(os.path.join(args.expname, TRAJGEN_FILENAME))
    print('trajs shape', trajs.shape)

    ## Need to get which traj corresponds to which x0 in data_val_snapshots_scaled
    prng = np.random.default_rng(seed=inf_args.seed)
    idxs = prng.choice(data_val_snapshots_scaled.shape[0], size=trajs.shape[0], replace=False)

    ## Pointwise metrics (RMSE and Cosine Similarity)
    pw_metrics = compute_pointwise_metrics(data_val_snapshots_scaled[idxs], trajs)

    ## Compute absolute error per feature
    abserr = np.abs(data_val_snapshots_scaled[idxs] - trajs)

    ## Distributional metrics (EMD and entropic EMD)
    dist_metrics = compute_distribution_metrics(
        data_val_snapshots_scaled[idxs],
        trajs,
        args.reg,
        args.sinkhorn_method
    )

    ## Save metrics
    np.savez(
        os.path.join(args.expname, EVALS_FILENAME),
        **pw_metrics,
        abserr=abserr,
        **dist_metrics
    )


if __name__ == '__main__':
    main()
