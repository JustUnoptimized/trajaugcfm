import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Literal

import jaxtyping as jt
import matplotlib.pyplot as plt
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

    plotgroup = parser.add_argument_group('plot', 'plot args')
    plotgroup.add_argument(
        '--nplot', type=int, default=20,
        help='Number of individual trajectory losses to plot'
    )
    plotgroup.add_argument(
        '--ncols', type=int, default=2,
        help='Number of columns of subplots showing metrics'
    )
    plotgroup.add_argument(
        '--ax-h', type=int_or_float, default=6,
        help='Height of each metric subplot'
    )
    plotgroup.add_argument(
        '--ax-w', type=int_or_float, default=8,
        help='Width of each metric subplot'
    )
    plotgroup.add_argument(
        '--ncols-indiv', type=int, default=4,
        help='Number of columns of subplots showing metrics on individual variables'
    )
    plotgroup.add_argument(
        '--ax-h-indiv', type=int_or_float, default=3,
        help='Height of each metric subplot for an individual variable'
    )
    plotgroup.add_argument(
        '--ax-w-indiv', type=int_or_float, default=4,
        help='Width of each metric subplot for an individual variable'
    )

    return parser.parse_args()


def chk_fmt_args(args: argparse.Namespace) -> argparse.Namespace:
    ## expgroup check
    exppath = os.path.join(RESDIR, args.expname)
    assert os.path.exists(exppath), f'{exppath} not found'
    args.expname = exppath

    ## metricgroup check
    assert args.reg > 0, f'reg must be positive but got {args.reg}'

    ## plotgroup check
    assert args.nplot > 0 or args.nplot == -1, \
        f'nplot must be positive or -1 but got {args.nplot}'
    assert args.ncols > 0, f'ncols must be positive but got {args.ncols}'
    assert args.ax_w > 0, f'ax-w must be positive but got {args.ax_w}'
    assert args.ax_h > 0, f'ax-h must be positive but got {args.ax_h}'
    assert args.ncols_indiv > 0, f'ncols-indiv must be positive but got {args.ncols_indiv}'
    assert args.ax_w_indiv > 0, f'ax-w-indiv must be positive but got {args.ax_w_indiv}'
    assert args.ax_h_indiv > 0, f'ax-h-indiv must be positive but got {args.ax_h_indiv}'

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

    ## Plotting
    nplotidxs = trajs.shape[0] if args.nplot == -1 else args.nplot
    plotidxs = prng.choice(trajs.shape[0], size=nplotidxs, replace=False)

    nrows, r = divmod(len(pw_metrics.keys()), args.ncols)
    if r > 0:
        nrows += 1

    fig, axs = plt.subplots(
        nrows=nrows, ncols=args.ncols,
        figsize=(args.ncols*args.ax_w, nrows*args.ax_h),
        sharex=True, squeeze=False
    )
    fig.suptitle('Pointwise Metrics Over All Features')

    for i, (name, metric) in enumerate(pw_metrics.items()):
        ax = axs[*divmod(i, args.ncols)]
        ax.set_title(name)

        ax.grid(visible=True, alpha=0.4)
        ax.plot(metric[:args.nplot].T, c='c', alpha=0.3)
        metric_mean = metric.mean(axis=0)
        metric_std = metric.std(axis=0)
        ax.plot(metric_mean, c='b')
        ax.fill_between(
            np.arange(metric.shape[1]),
            metric_mean + metric_std,
            metric_mean - metric_std,
            color='b',
            alpha=0.15
        )

    fig.tight_layout()
    fignamebase = os.path.join(args.expname, 'pw_metric_plots')
    fig.savefig(f'{fignamebase}.pdf')
    fig.savefig(f'{fignamebase}.png')

    nrows, r = divmod(len(dist_metrics.keys()), args.ncols)
    if r > 0:
        nrows += 1

    fig, axs = plt.subplots(
        nrows=nrows, ncols=args.ncols,
        figsize=(args.ncols*args.ax_w, nrows*args.ax_h),
        sharex=True, squeeze=False
    )
    fig.suptitle('Distributional Distance Metrics\nUsing Squared Euclidean Distance')

    for i, (name, metric) in enumerate(dist_metrics.items()):
        ax = axs[*divmod(i, args.ncols)]
        if name == 'Entropic EMD':
            name += f' (reg = {args.reg})'
        ax.set_title(name)

        ax.grid(visible=True, alpha=0.4)
        ax.plot(metric)

    fig.tight_layout()
    fignamebase = os.path.join(args.expname, 'dist_metric_plots')
    fig.savefig(f'{fignamebase}.pdf')
    fig.savefig(f'{fignamebase}.png')

    ## Plot absolute error for each individual variable
    nrows, r = divmod(abserr.shape[2], args.ncols_indiv)
    if r > 0:
        nrows += 1

    fig, axs = plt.subplots(
        nrows=nrows, ncols=args.ncols_indiv,
        figsize=(args.ncols_indiv*args.ax_w_indiv, nrows*args.ax_h_indiv),
        sharex=True, squeeze=False
    )
    fig.suptitle('Absolute Error per Variable')

    varnames = [varname for i, varname in enumerate(DYNOBS) if dynifmask[i]]
    for i in range(nrows*args.ncols_indiv):
        ax = axs[*divmod(i, args.ncols_indiv)]
        if i >= abserr.shape[2]:
            ax.axis('off')
            continue

        ax.set_title(varnames[i])

        ax.grid(visible=True, alpha=0.4)
        abserr_i = abserr[:args.nplot, :, i]
        ax.plot(abserr_i.T, c='c', alpha=0.3)
        mae = abserr_i.mean(axis=0)
        aestd = abserr_i.std(axis=0)
        ax.plot(mae, c='b')
        ax.fill_between(
            np.arange(abserr_i.shape[1]),
            mae - aestd,
            mae + aestd,
            color='b',
            alpha=0.15
        )

    fig.tight_layout()
    fignamebase = os.path.join(args.expname, 'metric_plots_indiv')
    fig.savefig(f'{fignamebase}.pdf')
    fig.savefig(f'{fignamebase}.png')


if __name__ == '__main__':
    main()
