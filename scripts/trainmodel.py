import argparse
import json
import os
import pickle

import jaxtyping as jt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from trajaugcfm.constants import (
    DATADIR,
    RESDIR,
    OBS,
    CONSTOBS,
    DYNOBS,
    IDX2RCMC_SAVENAME
)
from trajaugcfm.models import (
    FlowScoreMLP,
    MLP,
    flowscore_wrapper
)
from trajaugcfm.sampler import build_sampler_class
from trajaugcfm.utils import (
    build_indexer,
)
from script_utils import (
    MODEL_FILENAME,
    TRAINARGS_FILENAME,
    LOSSES_FILENAME,
    exitcodewrapper,
    int_or_float,
    load_data,
    save_scalers,
    scale_data
)

from train import train


def parse_args() -> argparse.Namespace:
    '''Parse all command line argument for training'''
    parser = argparse.ArgumentParser(prog='trainer')
    expgroup = parser.add_argument_group('expgroup', 'experiment setup args')
    expgroup.add_argument(
        '--expname', type=str, required=True,
        help='Results saved to results/<expname>/.' \
            +' If directory exists, contents are overwritten.'
    )
    expgroup.add_argument(
        '--existok', action='store_true',
        help='Flag to allow overwritting results from existing expname'
    )

    datagroup = parser.add_argument_group('datagroup', 'dataset selection args')
    datagroup.add_argument(
        '--data', type=str, required=True,
        help='Directory in data/<data>/ containing data.npy. Data should have shape (N, T, d).'
    )
    datagroup.add_argument(
        '--source', type=str, choices=['synth', 'marm'], required=True,
        help='Data source. Marm requires additional preprocessing.'
    )
    datagroup.add_argument(
        '--drugcombidx', type=int, default=0,
        help='Idx value for idx2rcmc dict'
    )
    datagroup.add_argument(
        '--obsidxs', type=int, nargs='+',
        help='Idxs for features used in trajectory guidance'
    )
    datagroup.add_argument(
        '--trainsize', type=int_or_float, default=0.8,
        help='If float in [0, 1], specifies ratio of train-val split.' \
            +' If int, specifies number of training samples.'
    )
    datagroup.add_argument(
        '--refsize', type=int_or_float, default=0.8,
        help='If float in [0, 1], specifies ratio of ref-snapshot split.' \
            +' If int, specifies number of reference samples.'
    )

    timegroup = parser.add_argument_group('timegroup', 'time sampler args')
    timegroup.add_argument(
        '--time-sampler', type=str, choices=['uniform', 'beta'], required=True,
        help='Sample time from Unif(0, 1) or Beta(a, a).'
    )
    timegroup.add_argument(
        '--beta-a', type=float, default=2.0,
        help='Shape parameter for sampling from Beta(a, a).' \
            +' Ignored if using uniform time sampler.'
    )
    timegroup.add_argument(
        '--use-time-enrich', action='store_true',
        help='Set to enable time embeddings'
    )
    timegroup.add_argument(
        '--time-enrich', type=str, choices=['rff'], default='rff',
        help='Use random fourier features to enrich time.' \
            +' Ignored if use-time-enrich is not set.'
    )
    timegroup.add_argument(
        '--rff-seed', type=int, default=2000,
        help='Seed for consistent rff frequencies across train-test splits.' \
            +' Ignored if not using rff time enrichment.'
    )
    timegroup.add_argument(
        '--rff-scale', type=float, default=1.0,
        help='Used to sample random frequencies from N(0, rff_scale).' \
            +' Ignored if not using rff time enrichment.'
    )
    timegroup.add_argument(
        '--rff-dim', type=int, default=1,
        help='Number of frequency pairs for rff time embedding.' \
            +' Ignored if not using rff time enrichment.'
    )

    flowgroup = parser.add_argument_group('flow', 'flow matcher args')
    flowgroup.add_argument(
        '--flow', type=str, choices=['isotropic', 'anisotropic'], required=True,
        help='Select shape of conditional probability path'
    )
    flowgroup.add_argument(
        '--flow-bridge', type=str, choices=['constant', 'schrodinger'], default='constant',
        help='Select variance schedule for isotropic flow.' \
            +' Ignored if flow is anisotropic.'
    )
    flowgroup.add_argument(
        '--sigma', type=float, default=1.0,
        help='Scale for variance schedule. Ignored if using anisotropic flow.'
    )
    flowgroup.add_argument(
        '--sb-reg', type=float, default=1e-8,
        help='Regularization when computing sigma_t_prime / sigma_t_inv.' \
            +' Only used for isotropic flow, schrodinger bridge.'
    )

    scoregroup = parser.add_argument_group('score', 'score matcher args')
    scoregroup.add_argument(
        '--score', action='store_true',
        help='Set to enable score matching.'
    )
    scoregroup.add_argument(
        '--score-shape', type=str, choices=['anisotropic'], default='anisotropic',
        help='Select shape of conditional probability path for score matching.' \
            +' Ignored if score is not set.'
    )

    samplergroup = parser.add_argument_group('sampler', 'trajectory augmented sampler args')
    samplergroup.add_argument(
        '--k', type=int, default=8,
        help='Number of refs per minibatch'
    )
    samplergroup.add_argument(
        '--n', type=int, default=128,
        help='Number of samples per snapshot for weighted minibatch sampling'
    )
    samplergroup.add_argument(
        '--b', type=int, default=8,
        help='Minibatch size per ref'
    )
    samplergroup.add_argument(
        '--nt', type=int, default=8,
        help='Number of time points sampled per minibatch'
    )
    samplergroup.add_argument(
        '--gprscale', type=int_or_float, default=0.1,
        help='Scale for RBF kernel in Gaussian Process Regressions'
    )
    samplergroup.add_argument(
        '--gprbounds', type=int_or_float, nargs='+', default=None,
        help='Scale bounds for RBF kernel in Gaussian Process Regressions.' \
            +' If specified, must only have 2 numbers in ascending order.' \
            +' If unspecified, keep scale fixed and do not optimize.'
    )
    samplergroup.add_argument(
        '--whitenoise', type=float, default=0.1,
        help='White noise level for White kernel in Gaussian Process Regressions'
    )
    samplergroup.add_argument(
        '--gprnt', type=int, default=8,
        help='Number of training points for the Gaussian Process Regressions'
    )
    samplergroup.add_argument(
        '--rbfdistscale', type=int_or_float, default=1.,
        help='RBF scale for conditional sampling of minibatch given ref'
    )
    samplergroup.add_argument(
        '--reg', type=float, default=1e-8,
        help='Regularization scale for matrices before eigendecomposition'
    )

    modelgroup = parser.add_argument_group('model', 'model args')
    modelgroup.add_argument(
        '--depth', type=int, default=2,
        help='Number of MLP hidden layers'
    )
    modelgroup.add_argument(
        '--width', type=int, default=64,
        help='Width of each MLP hidden layer'
    )

    traingroup = parser.add_argument_group('training', 'training args')
    traingroup.add_argument(
        '--optimizer', type=str, default='AdamW',
        help='Name of optimizer retrieved equivalently to torch.optim.<optimizer>()'
    )
    traingroup.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    traingroup.add_argument(
        '--scheduler', action='store_true',
        help='Set to use CosineAnnealingLR scheduler.'
    )
    traingroup.add_argument(
        '--epochs', type=int, default=1000,
        help='Number of training epochs (defined as a pass through all refs)'
    )
    traingroup.add_argument(
        '--val-every', type=int, default=50,
        help='Interval for computing a validation epoch during training'
    )
    traingroup.add_argument(
        '--gradclip-max-norm', type=float, default=None,
        help='If specified, clip gradient norm.'
    )
    traingroup.add_argument(
        '--progress', action='store_true',
        help='Show training progress bar'
    )

    miscgroup = parser.add_argument_group('misc', 'misc args')
    miscgroup.add_argument(
        '--nogpu', action='store_true',
        help='If set, force training on CPU. If not set, attempt GPU if available'
    )
    miscgroup.add_argument(
        '--seed', type=int, default=None,
        help='Seed for random number generators and reproducability'
    )

    ## Diagnostics
    # diaggroup = parser.add_argument_group('diagnostics', 'logging and instrumentation')
    # diaggroup.add_argument(
        # '--log-metrics', action='store_true',
        # help='Log per-step metrics (loss, norms, grad norm, lr) to results/<exp>/metrics.csv'
    # )

    return parser.parse_args()


def chk_fmt_args(args: argparse.Namespace) -> argparse.Namespace:
    '''Checks and formats command line arguments

    Modifies the internal state of certain args.
    Returns args.
    '''
    ## expgroup check
    exppath = os.path.join(RESDIR, args.expname)
    args.expname = exppath

    ## datagroup check
    datadir = os.path.join(DATADIR, args.data)
    datapath = os.path.join(datadir, 'data.npy')
    assert os.path.exists(datapath), f'{datapath} not found'
    args.data = datadir

    if args.source == 'marm':
        idx2rcmcpath = os.path.join(datadir, IDX2RCMC_SAVENAME)
        assert os.path.exists(idx2rcmcpath), f'{idx2rcmcpath} not found'
        with open(idx2rcmcpath, 'rb') as f:
            idx2rcmc = pickle.load(f)
        assert args.drugcombidx in idx2rcmc.keys(), \
            f'Drug combination {args.drugcombidx} not found'

    assert args.trainsize > 0, f'Trainsize must be positive but got {args.trainsize}'
    assert args.refsize > 0, f'Refsize must be positive but got {args.refsize}'

    ## timegroup check
    assert args.beta_a > 1, f'beta-a must be > 1 but got {args.beta_a}'
    assert args.rff_seed >= 0, f'rff-seed must be non-negative but got {args.rff_seed}'
    assert args.rff_scale >= 0, f'rff-seed must be non-negative but got {args.rff_scale}'
    assert args.rff_dim > 0, f'rff-dim must be positive but got {args.rff_dim}'

    ## flowgroup check
    assert args.sigma >= 0, f'sigma must be non-negative but got {args.sigma}'
    assert args.sb_reg > 0, f'sb_reg must be positive but got {args.sb_reg}'

    ## samplergroup check
    assert args.k > 0, f'k must be positive but got {args.k}'
    assert args.n > 0, f'n must be positive but got {args.n}'
    assert args.b > 0, f'b must be positive but got {args.b}'
    assert args.nt > 0, f'nt must be positive but got {args.nt}'
    assert args.gprscale > 0, f'gprscale must be positive but got {args.gprscale}'
    if args.gprbounds is not None:
        gprbounds = tuple(args.gprbounds)
        assert len(gprbounds) == 2, \
            f'gprbounds must have 2 entries but got {len(gprbounds)}'
        assert gprbounds[0] < gprbounds[1], \
            f'gpr lower bound must be less than upper bound but got {gprbounds}'
        args.gprbounds = gprbounds
    else:
        args.gprbounds = 'fixed'
    assert args.whitenoise >= 0, f'whitenoise must be non-negative but got {args.whitenoise}'
    assert args.gprnt > 0, f'gprnt must be positive but got {args.gprnt}'
    assert args.rbfdistscale > 0, \
        f'rbfdistscale must be positive but got {args.rbfdistscale}'
    assert args.reg > 0, f'reg must be positive but got {args.reg}'

    ## modelgroup check
    assert args.depth >= 0, f'depth must be non-negative but got {args.depth}'
    assert args.width > 0, f'width must be positive but got {args.width}'

    ## traingroup check
    assert args.lr > 0, f'lr must be positive but got {args.lr}'
    assert args.epochs > 0, f'epochs must be positive but got {args.epochs}'
    assert args.val_every > 0, f'valevery must be positive but got {args.valevery}'
    if args.gradclip_max_norm is not None:
        assert args.gradclip_max_norm > 0, \
            f'gradclip-max-norm must be positive but got {args.gradclip_max_norm}'

    if args.seed is not None:
        assert args.seed >= 0, f'seed must be non-negative but got {args.seed}'

    return args


def set_up_exp(args: argparse.Namespace) -> None:
    '''Create expdir if not exist and dump json argfile'''
    os.makedirs(args.expname, exist_ok=args.existok)
    with open(os.path.join(args.expname, TRAINARGS_FILENAME), 'w') as f:
        json.dump(vars(args), f, indent=4)


def save_train_metrics(
    outdir: str,
    score: bool,
    train_flow_losses: jt.Real[np.ndarray, 'epochs nsteps'],
    train_score_losses: jt.Real[np.ndarray, 'epochs nsteps'] | None,
    val_flow_losses: jt.Real[np.ndarray, 'nvals'],
    val_score_losses: jt.Real[np.ndarray, 'nvals'] | None,
    lrs: jt.Real[np.ndarray, 'epochs']
) -> None:
    if score:
        np.savez(
            outdir,
            train_flow_losses=train_flow_losses,
            train_score_losses=train_score_losses,
            val_flow_losses=val_flow_losses,
            val_score_losses=val_score_losses,
            lrs=lrs
        )
    else:
        np.savez(
            outdir,
            train_flow_losses=train_flow_losses,
            val_flow_losses=val_flow_losses,
            lrs=lrs
        )


@exitcodewrapper
def main() -> None:
    args = parse_args()
    args = chk_fmt_args(args)
    set_up_exp(args)

    print('\nLoading data...')
    data, varnames = load_data(args.data, args.source, args.drugcombidx)

    obsmask = np.zeros(data.shape[-1], dtype=bool)
    obsmask[args.obsidxs] = True
    hidmask = ~obsmask
    tidxs = [0, -1]
    dobs = obsmask.sum()
    dhid = hidmask.sum()
    d = dobs + dhid

    print('\nSplitting into train-val sets for snapshots and references')
    data_train, data_val = train_test_split(
        data, train_size=args.trainsize,
        random_state=args.seed if args.seed is None else args.seed+2
    )
    print('data train shape', data_train.shape)
    data_train_snapshots, data_train_refs = train_test_split(
        data_train, test_size=args.refsize,
        random_state=args.seed if args.seed is None else args.seed+3
    )
    data_train_snapshots = data_train_snapshots[:, tidxs]
    data_train_refs = data_train_refs[:, :, obsmask]
    print('data train snapshots shape', data_train_snapshots.shape)
    print('data train refs shape', data_train_refs.shape)

    print('data val shape', data_val.shape)
    data_val_snapshots, data_val_refs = train_test_split(
        data_val, test_size=args.refsize,
        random_state=args.seed if args.seed is None else args.seed+4
    )
    data_val_snapshots = data_val_snapshots[:, tidxs]
    data_val_refs = data_val_refs[:, :, obsmask]
    print('data val snapshots shape', data_val_snapshots.shape)
    print('data val refs shape', data_val_refs.shape)

    print('\nScaling data using train split...')
    (
        data_train_snapshots_scaled,
        data_train_refs_scaled, 
        data_val_snapshots_scaled,
        data_val_refs_scaled,
        obs_scaler,
        hid_scaler
    ) = scale_data(
        data_train_snapshots,
        data_train_refs,
        data_val_snapshots,
        data_val_refs,
        obsmask,
        hidmask
    )

    print('\nSaving scalers...')
    save_scalers(args.expname, obs_scaler, hid_scaler)

    print('\nConstructing Sampler...')
    GCFMSampler = build_sampler_class(
        args.time_sampler,
        args.use_time_enrich,
        args.time_enrich,
        args.flow,
        args.flow_bridge,
        args.score,
        args.score_shape
    )
    print('\nMixins:')
    print(' '.join(GCFMSampler.get_mixin_names())+'\n')

    train_sampler = GCFMSampler(
        np.random.default_rng(seed=args.seed),
        data_train_snapshots_scaled,
        data_train_refs_scaled,
        obsmask,
        tidxs,
        args.k,
        args.n,
        args.b,
        args.nt,
        rbfk_scale=args.gprscale,
        rbfk_bounds=args.gprbounds,
        whitenoise=args.whitenoise,
        gpr_nt=args.gprnt,
        rbfd_scale=args.rbfdistscale,
        reg=args.reg,
        sigma=args.sigma,
        sb_reg=args.sb_reg,
        beta_a=args.beta_a,
        rff_seed=args.rff_seed,
        rff_scale=args.rff_scale,
        rff_dim=args.rff_dim,
    )
    val_sampler = GCFMSampler(
        np.random.default_rng(seed=args.seed if args.seed is None else args.seed+1),
        data_val_snapshots_scaled,
        data_val_refs_scaled,
        obsmask,
        tidxs,
        args.k,
        args.n,
        args.b,
        args.nt,
        rbfk_scale=args.gprscale,
        rbfk_bounds=args.gprbounds,
        whitenoise=args.whitenoise,
        gpr_nt=args.gprnt,
        rbfd_scale=args.rbfdistscale,
        reg=args.reg,
        sigma=args.sigma,
        sb_reg=args.sb_reg,
        beta_a=args.beta_a,
        rff_seed=args.rff_seed,
        rff_scale=args.rff_scale,
        rff_dim=args.rff_dim,
    )
    train_loader = DataLoader(train_sampler, batch_size=None)
    val_loader = DataLoader(val_sampler, batch_size=None)

    print('\nConstructing Model...')
    d_vars = data_train_snapshots.shape[-1]
    d_out = d_vars
    w = args.width
    h = args.depth
    if args.use_time_enrich:
        if args.time_enrich == 'rff':
            d_time = args.rff_dim * 2
    else:
        d_time = 1
    d_in = d_vars + d_time
    if args.score:
        model = FlowScoreMLP(d_in, d_out, w=w, h=h)
    else:
        model = MLP(d_in, d_out, w=w, h=h)
        model = flowscore_wrapper(model)
    print(model)

    device = 'cuda' if (not args.nogpu) and torch.cuda.is_available() else 'cpu'
    print('device:', device)
    model = model.to(device)
    opt = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    print('optimizer:', opt)
    if args.scheduler:
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    else:
        lr_sched = None
    print('lr scheduler:', lr_sched)
    lossfn = torch.nn.MSELoss()

    print('\nTraining model...')
    train_flow_losses, train_score_losses, val_flow_losses, val_score_losses, lrs = train(
        model,
        opt,
        lr_sched,
        train_loader,
        val_loader,
        lossfn,
        args.epochs,
        args.val_every,
        args.gradclip_max_norm,
        args.score,
        args.progress,
        device
    )

    print('\nSaving results...')
    torch.save(model.state_dict(), os.path.join(args.expname, MODEL_FILENAME))
    save_train_metrics(
        os.path.join(args.expname, LOSSES_FILENAME),
        args.score,
        train_flow_losses,
        train_score_losses,
        val_flow_losses,
        val_score_losses,
        lrs
    )


if __name__ == '__main__':
    main()

