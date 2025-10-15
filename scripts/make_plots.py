import argparse
from collections.abc import Sequence
import json
import os

import jaxtyping as jt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.lib.npyio import NpzFile
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from trajaugcfm.constants import (
    CONSTOBS,
    DYNOBS,
    RESDIR,
    OBS,
)
from trajaugcfm.utils import (
    build_indexer
)

from script_utils import (
    EVALARGS_FILENAME,
    EVALS_FILENAME,
    LOSSES_FILENAME,
    MODEL_FILENAME,
    PLOTARGS_FILENAME,
    TRAINARGS_FILENAME,
    TRAJGEN_FILENAME,
    TRAJGENARGS_FILENAME,
    exitcodewrapper,
    int_or_float,
    load_args,
    load_data,
    load_scalers,
    scale_data_with_scalers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='make plots')
    commongroup = parser.add_argument_group('common', 'common load args')
    commongroup.add_argument(
        '--expname', type=str, required=True,
        help='Load experiment in results/<expname>/.'
    )
    commongroup.add_argument(
        '--ax-h', type=int_or_float, default=4,
        help='Subplot height'
    )
    commongroup.add_argument(
        '--ax-w', type=int_or_float, default=8,
        help='Subplot width'
    )
    commongroup.add_argument(
        '--metric-ncols', type=int, default=2,
        help='Number of subplot cols for figures involving evaluation metrics'
    )
    commongroup.add_argument(
        '--feature-ncols', type=int, default=3,
        help='Number of subplot cols for figures involving individual features (trajs and errors)'
    )
    commongroup.add_argument(
        '--extensions', type=str, nargs='+', default=['pdf', 'png'],
        help='File format for matplotlib. Must be recognized by savefig().'
    )

    trajgroup = parser.add_argument_group('traj', 'trajectory plotting args')
    trajgroup.add_argument(
        '--ax-kde', type=int_or_float, default=2,
        help='KDE subplot width'
    )
    trajgroup.add_argument(
        '--nrefs', type=int, default=100,
        help='Number of reference trajs to plot.' \
            +' Set to -1 to plot all reference trajs.'
    )
    trajgroup.add_argument(
        '--ntrajs', type=int, default=20,
        help='Number of inferred trajs from sdesolve() to plot.' \
            +' Set to -1 to plot all inferred trajs.'
    )

    evalgroup = parser.add_argument_group('eval', 'evaluation plotting args')
    evalgroup.add_argument(
        '--nevals', type=int, default=20,
        help='Number of individual trajectory losses to plot.' \
            +' Set to -1 to plot losses from all trajectories'
    )

    return parser.parse_args()


def chk_fmt_args(args: argparse.Namespace) -> argparse.Namespace:
    ## commongroup check
    exppath = os.path.join(RESDIR, args.expname)
    assert os.path.exists(exppath), f'{exppath} not found'
    args.expname = exppath
    assert args.ax_h > 0, f'ax-h must be positive but got {args.ax_h}'
    assert args.ax_w > 0, f'ax-w must be positive but got {args.ax_w}'
    assert args.metric_ncols > 0, f'metric-ncols must be positive but got {args.metric_ncols}'
    assert args.feature_ncols > 0, f'feature-ncols must be positive but got {args.feature_ncols}'
    assert len(args.extensions) > 0, f'extensions must not be empty but got {args.extensions}'

    ## trajgroup check
    assert args.ax_kde > 0, f'ax-kde must be positive but got {args.ax_kde}'
    assert args.nrefs > 0 or args.nrefs == -1, \
        f'nrefs must be positive or -1 but got {args.nrefs}'
    assert args.ntrajs > 0 or args.ntrajs == -1, \
        f'ntrajs must be positive or -1 but got {args.ntrajs}'

    ## evalgroup check
    assert args.nevals > 0 or args.nevals == -1, \
        f'nevals must be positive or -1 but got {args.nevals}'

    return args


def save_plot_args(args: argparse.Namespace, expname: str) -> None:
    '''Save args for plot instance to json file.'''
    plotargs_path = os.path.join(expname, PLOTARGS_FILENAME)
    with open(plotargs_path, 'w') as f:
        json.dump(vars(args), f, indent=4)


def plot_extensions(
    fig: Figure,
    dirname: str,
    figname: str,
    exts: Sequence[str],
) -> None:
    '''Saves a figure with all supplied extensions'''
    name = os.path.join(dirname, figname)
    for ext in exts:
        fig.savefig(f'{name}.{ext}')


def plot_training(
    epochs: int,
    val_every: int,
    score: bool,
    train_flow_losses: jt.Real[np.ndarray, 'epochs nsteps'],
    train_score_losses: jt.Real[np.ndarray, 'epochs nsteps'] | None,
    val_flow_losses: jt.Real[np.ndarray, 'nvals'],
    val_score_losses: jt.Real[np.ndarray, 'nvals'] | None,
    lrs: jt.Real[np.ndarray, 'epochs'],
    ax_w: int | float,
    ax_h: int | float,
    expname: str,
    extensions: Sequence[str]
) -> None:
    '''Plot training loss and learning rate over epochs'''
    train_step_space = np.arange(epochs)
    val_step_space = np.arange(val_flow_losses.shape[0]) * val_every
    val_step_space[-1] = epochs

    ncols = 2 if score else 1
    fig, axs = plt.subplots(ncols=ncols, figsize=(ax_w*ncols, ax_h), sharex=True)
    fig.supxlabel('Epochs')

    if score:
        ax = axs[0]
    else:
        ax = axs
    ax.set_title('Flow Losses')
    ax.grid(visible=True)
    ax.plot(train_step_space, train_flow_losses.mean(axis=1), label='Train')
    ax.plot(val_step_space, val_flow_losses, label='Val')
    ax.legend(loc='upper right')

    if score:
        ax = axs[1]
        ax.set_title('Score Losses')
        ax.set_ylim(top=5)
        ax.grid(visible=True)
        ax.plot(train_step_space, train_score_losses.mean(axis=1), label='Train')
        ax.plot(val_step_space, val_score_losses, label='Val')
        ax.legend(loc='upper right')

    fig.tight_layout()
    plot_extensions(fig, expname, 'loss', extensions)

    fig, ax = plt.subplots(figsize=(ax_w, ax_h))

    ax.grid(visible=True)
    ax.set_title('Learning Rate')
    ax.plot(train_step_space, lrs)
    fig.tight_layout()
    plot_extensions(fig, expname, 'lrs', extensions)


def plot_true_trajs(
    obsmask: jt.Bool[np.ndarray, 'd'],
    refs: jt.Float64[np.ndarray, 'N T o'] | None,
    refs_hid: jt.Float64[np.ndarray, 'N T d-o'] | None,
    nrefs: int,
    ncols: int,
    axs: jt.Shaped[np.ndarray, 'nrows ncols 3']
) -> None:
    '''Helper method to plot true trajectories.

    Either refs or refs_hid must not be None.
    '''
    d = obsmask.shape[0]
    T = refs.shape[1] if refs is not None else refs_hid.shape[1]
    ts = np.linspace(0, 1, T)
    j = 0
    for i in range(d):
        ## select reference trajectory, if it exists
        if obsmask[i]:
            val_trajs_di = refs[:nrefs, :, i] if refs is not None else None
        elif not obsmask[i]:
            val_trajs_di = refs_hid[:nrefs, :, j] if refs_hid is not None else None
            j += 1

        if val_trajs_di is not None:
            r, c, = divmod(i, ncols)
            ax = axs[r, c, 1]  ## subplot [1] has trajs

            ## Plot trajectories
            if i == 0:
                ax.plot(ts, val_trajs_di[0], c='c', alpha=0.25,
                        label='True')
            ax.plot(ts, val_trajs_di[1:].T, c='c', alpha=0.2)
            ax.set_xlim((0, 1))
            ax.tick_params(axis='x', which='both', direction='in')
            ax.tick_params(axis='y', which='both', left=False)

            ## Plot KDEs
            for ax_idx, t in zip([0, 2], [0, T-1]):
                ax = axs[r, c, ax_idx]
                sns.kdeplot(
                    y=val_trajs_di[:, t],
                    fill=True,
                    ax=ax,
                    color='c',
                    alpha=0.4
                )
                ax.set_xlabel(None)
                ax.set_ylabel(None)
                ax.set_xticks([])

            ## Format left KDE plot
            axs[r, c, 0].xaxis.set_inverted(True)

            ## Format right KDE plot
            axs[r, c, 2].tick_params(axis='y', which='both', left=False)


def plot_trajs(
    trajs: jt.Float64[np.ndarray, 'N T d'],
    obsmask: jt.Bool[np.ndarray, 'd'] | None,
    refs: jt.Float64[np.ndarray, 'N T o'] | None,
    refs_hid: jt.Float64[np.ndarray, 'N T d-o'] | None,
    nrefs: int,
    ntrajs: int,
    varnames: Sequence[str],
    ncols: int,
    ax_w: int | float,
    ax_h: int | float,
    ax_kde: int | float,
    expname: str,
    extensions: Sequence[str]
) -> None:
    '''Plot inferred trajectories

    If obsmask is specified, plot against true trajectories
    '''
    d = trajs.shape[2]
    nrows, r = divmod(d, ncols)
    if r > 0:
        nrows += 1

    ## fig width should account for traj + KDE on both ends per feature
    fig = plt.figure(figsize=(ncols*(ax_w+(ax_kde*2)), nrows*ax_h))

    subfigs = fig.subfigures(
        nrows=nrows, ncols=ncols,
        squeeze=False, wspace=0, hspace=0
    )
    subcols = 3  ## [0, 2] for endpoint marginal kde, [1] for trajs
    wr = [0.2, 0.6, 0.2]  ## width ratios for subcols

    ts =  np.linspace(0, 1, trajs.shape[1])

    ## Make subplots in each subfigure
    axs = np.empty((nrows, ncols, subcols), dtype=object)
    for i in range(d):
        r, c = divmod(i, ncols)
        axs[r, c] = subfigs[r, c].subplots(
            ncols=subcols, sharey=True, width_ratios=wr, gridspec_kw=dict(wspace=0)
        )

    ## Plot ground truth trajectories and KDEs, if any
    if obsmask is not None:
        plot_true_trajs(
            obsmask,
            refs,
            refs_hid,
            nrefs,
            ncols,
            axs
        )

    T = trajs.shape[1]
    ts = np.linspace(0, 1, T)
    for i in range(d):
        r, c = divmod(i, ncols)
        ax = axs[r, c, 1]

        ## Plot Trajectories
        ax.set_title(varnames[i])
        if i == 0:
            ax.plot(ts, trajs[0, :, i], c='tab:orange', alpha=0.5,
                    label='Inferred')
            ax.legend(loc='upper right')
        else:
            ax.plot(ts, trajs[0,:, i], c='tab:orange', alpha=0.5)
        ax.plot(ts, trajs[1:ntrajs, :, i].T, c='tab:orange', alpha=0.5)
        ax.set_xlim((0, 1))
        ax.tick_params(axis='x', which='both', direction='in')
        ax.tick_params(axis='y', which='both', left=False)

        ## Plot KDEs
        for ax_idx, t in zip([0, 2], [0, T-1]):
            ax = axs[r, c, ax_idx]
            sns.kdeplot(
                y=trajs[:ntrajs, t, i],
                fill=True,
                ax=ax,
                color='tab:orange',
                alpha=0.4
            )
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_xticks([])

        ## Format left KDE plot
        axs[r, c, 0].xaxis.set_inverted(True)

        ## Format right KDE plot
        axs[r, c, 2].tick_params(axis='y', which='both', left=False)

    fig.tight_layout()
    ## left for left-most yticks, bottom for xticks, and top for feature names
    fig.subplots_adjust(left=0.03, bottom=0.08, top=0.9)
    plot_extensions(fig, expname, 'trajs', extensions)


def plot_evals(
    eval_metrics: NpzFile,
    nevals: int,
    reg: float,
    varnames: Sequence[str],
    metric_ncols: int,
    feature_ncols: int,
    ax_w: int | float,
    ax_h: int | float,
    expname: str,
    extensions: Sequence[str]
) -> None:
    '''Plot all evaluation metrics'''
    ## sort keys into categories
    pointwise_metrics = []
    feature_metrics = []
    dist_metrics = []
    for name, arr in eval_metrics.items():
        if arr.ndim == 1:
            dist_metrics.append(name)
        elif arr.ndim == 2:
            pointwise_metrics.append(name)
        elif arr.ndim == 3:
            feature_metrics.append(name)
        else:
            raise ValueError(f'Metric array with ndim > 3 is unhandled')

    ## Plot pointwise metrics
    nrows, r = divmod(len(pointwise_metrics), metric_ncols)
    if r > 0:
        nrows += 1

    fig, axs = plt.subplots(
        nrows=nrows, ncols=metric_ncols,
        figsize=(metric_ncols*ax_w, nrows*ax_h),
        sharex=True, squeeze=False
    )

    for i, name in enumerate(pointwise_metrics):
        metric = eval_metrics[name]
        ax = axs[*divmod(i, metric_ncols)]
        ax.set_title(name)

        ax.grid(visible=True, alpha=0.4)
        ax.plot(metric[:nevals].T, c='c', alpha=0.3)
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
    plot_extensions(fig, expname, 'pw_metric_plots', extensions)

    ## Plot distributional distances
    nrows, r = divmod(len(dist_metrics), metric_ncols)
    if r > 0:
        nrows += 1

    fig, axs = plt.subplots(
        nrows=nrows, ncols=metric_ncols,
        figsize=(metric_ncols*ax_w, nrows*ax_h),
        sharex=True, squeeze=False
    )

    for i, name in enumerate(dist_metrics):
        metric = eval_metrics[name]
        ax = axs[*divmod(i, metric_ncols)]
        if name == 'Entropic EMD':
            name += f' (reg = {reg})'
        ax.set_title(name)

        ax.grid(visible=True, alpha=0.4)
        ax.plot(metric)

    fig.tight_layout()
    plot_extensions(fig, expname, 'dist_metric_plots', extensions)

    ## Plot feature metrics
    abserr = eval_metrics[feature_metrics[0]]
    nrows, r = divmod(abserr.shape[2], feature_ncols)
    if r > 0:
        nrows += 1

    fig, axs = plt.subplots(
        nrows=nrows, ncols=feature_ncols,
        figsize=(feature_ncols*ax_w, nrows*ax_h),
        sharex=True, squeeze=False
    )
    fig.suptitle('Absolute Error per Variable')

    for i in range(nrows*feature_ncols):
        ax = axs[*divmod(i, feature_ncols)]
        if i >= abserr.shape[2]:
            ax.axis('off')
            continue

        ax.set_title(varnames[i])

        ax.grid(visible=True, alpha=0.4)
        abserr_i = abserr[:nevals, :, i]
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
    plot_extensions(fig, expname, 'metric_plots_indiv', extensions)


@exitcodewrapper
def main() -> None:
    args = parse_args()
    args = chk_fmt_args(args)
    exp_args = load_args(args.expname, TRAINARGS_FILENAME)
    inf_args = load_args(args.expname, TRAJGENARGS_FILENAME)
    eval_args = load_args(args.expname, EVALARGS_FILENAME)
    save_plot_args(args, args.expname)

    print('Recreating experiment data...\nLoading experiment data...')
    data, varnames = load_data(exp_args.data, exp_args.source, exp_args.drugcombidx)

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
    data_val_snapshots = data_val_snapshots[:, tidxs]
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
    data_val_refs_hid_scaled = hid_scaler.transform(
        data_val_refs_hid.reshape((-1, dhid))
    ).reshape(data_val_refs_hid.shape)

    ## Training Plots
    print('Loading training losses and learning rates...')
    train_metrics = np.load(os.path.join(args.expname, LOSSES_FILENAME))

    print('Plotting losses and learning rates over epochs...')
    plot_training(
        exp_args.epochs,
        exp_args.val_every,
        exp_args.score,
        train_metrics['train_flow_losses'],
        train_metrics['train_score_losses'] if exp_args.score else None,
        train_metrics['val_flow_losses'],
        train_metrics['val_score_losses'] if exp_args.score else None,
        train_metrics['lrs'],
        args.ax_w,
        args.ax_h,
        args.expname,
        args.extensions
    )

    ## Traj Gen Plots
    print('Loading saved trajectories...')
    trajs = np.load(os.path.join(args.expname, TRAJGEN_FILENAME))

    print('Plotting trajectories...')
    plot_trajs(
        trajs,
        obsmask,
        data_val_refs_scaled,
        data_val_refs_hid_scaled,
        args.nrefs,
        args.ntrajs,
        varnames,
        args.feature_ncols,
        args.ax_w,
        args.ax_h,
        args.ax_kde,
        args.expname,
        args.extensions
    )

    ## Traj Eval Plots
    print('Loading evaluation metrics...')
    eval_metrics = np.load(os.path.join(args.expname, EVALS_FILENAME))

    print('Plotting evaluation metrics...')
    plot_evals(
        eval_metrics,
        args.nevals,
        eval_args.reg,
        varnames,
        args.metric_ncols,
        args.feature_ncols,
        args.ax_w,
        args.ax_h,
        args.expname,
        args.extensions,
    )


if __name__ == '__main__':
    main()

