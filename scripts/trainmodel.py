import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
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
from trajaugcfm.models import MLP
from trajaugcfm.sampler import TrajAugCFMSampler
from trajaugcfm.utils import (
    build_indexer,
)

from train import train


def int_or_float(x: str) -> int|float:
    '''Convert to int with fallback to float'''
    try:
        return int(x)
    except:
        try:
            return float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f'Could not convert {x} to a int or float'
            )


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
        help='Directory in data/<data>/ containing data.npy'
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
        '--gprbounds', type=int_or_float, nargs='+',
        help='Scale bounds for RBF kernel in Gaussian Process Regressions.' \
            +' If specified, must only have 2 numbers in ascending order.' \
            +' If unspecified, keep scale fixed and do not optimize.'
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
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    traingroup.add_argument(
        '--lossname', type=str, default='MSELoss',
        help='Name of loss fn retrieved equivalently to torch.nn.<lossname>()'
    )
    traingroup.add_argument(
        '--epochs', type=int, default=1000,
        help='Number of training epochs (defined as a pass through all refs)'
    )
    traingroup.add_argument(
        '--valevery', type=int, default=50,
        help='Interval for computing a validation epoch during training'
    )
    traingroup.add_argument(
        '--progress', action='store_true',
        help='Show training progress bar'
    )
    traingroup.add_argument(
        '--val-mean-reduction', action='store_true',
        help='Use mean reduction for validation loss (default keeps legacy sum)'
    )

    timegroup = parser.add_argument_group('time-embed', 'time embedding args')
    timegroup.add_argument(
        '--time-embed', action='store_true',
        help='Enable random Fourier features for time input'
    )
    timegroup.add_argument(
        '--time-embed-dim', type=int, default=16,
        help='Number of frequency pairs for time embedding (phi has 2*dim features)'
    )
    timegroup.add_argument(
        '--time-embed-scale', type=float, default=1.0,
        help='Stddev of RFF frequencies b ~ N(0, scale)'
    )

    scoregroup = parser.add_argument_group('score-head', 'optional score head and loss mixing')
    scoregroup.add_argument(
        '--score-head', action='store_true',
        help='Use dual-head model (flow + score) and add score regression loss'
    )
    scoregroup.add_argument(
        '--score-lambda', type=float, default=0.1,
        help='Weight for score loss term in total loss'
    )

    ## Time sampling options (for stability near endpoints)
    tsgroup = parser.add_argument_group('time-sampling', 'time sampling law for t')
    tsgroup.add_argument(
        '--time-sample', type=str, default='uniform',
        help="Time sampling law: 'uniform' or 'beta' (Beta(a,a))"
    )
    tsgroup.add_argument(
        '--time-beta-a', type=float, default=2.0,
        help='Alpha parameter a for symmetric Beta(a,a) time sampling when enabled'
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

    ## FIX(GIF): Guided Isotropic Flow feature flag and options
    fixgroup = parser.add_argument_group('fixgif', 'guided isotropic flow fix flags')
    fixgroup.add_argument(
        '--fixgif', action='store_true',
        help='Enable Guided Isotropic Flow path: isotropic covariance + bridge means'
    )
    fixgroup.add_argument(
        '--fixgif-sigma', type=str, default='bb',
        help="Sigma schedule for GIF: 'const' or 'bb' (Brownian-bridge)"
    )
    fixgroup.add_argument(
        '--fixgif-sigma-scale', type=float, default=1.0,
        help='Base sigma scale for GIF schedule (stddev units after scaling)'
    )
    fixgroup.add_argument(
        '--fixgif-sigma-eps', type=float, default=0.0,
        help='Additive epsilon inside sigma^2(t)=c^2(eps+t(1-t)) to bound sigma\'/sigma'
    )
    fixgroup.add_argument(
        '--score-gauss', action='store_true',
        help='When used with --fixgif, sampler builds Gaussian score target (low-rank+diag)'
    )

    ## Diagnostics
    diaggroup = parser.add_argument_group('diagnostics', 'logging and instrumentation')
    diaggroup.add_argument(
        '--log-metrics', action='store_true',
        help='Log per-step metrics (loss, norms, grad norm, lr) to results/<exp>/metrics.csv'
    )

    return parser.parse_args()


def chk_fmt_args(args: argparse.Namespace) -> argparse.Namespace:
    '''Checks and formats command line arguments

    May modify internal state of args.
    Returns args.
    '''
    ## expgroup check
    exppath = os.path.join(RESDIR, args.expname)
    args.expname = exppath

    ## datagroup check
    datadir = os.path.join(DATADIR, args.data)
    datapath = os.path.join(datadir, 'data.npy')
    assert os.path.exists(datapath), f'{datapath} not found'
    args.data = datapath

    idx2rcmcpath = os.path.join(datadir, IDX2RCMC_SAVENAME)
    assert os.path.exists(idx2rcmcpath), f'{idx2rcmcpath} not found'
    with open(idx2rcmcpath, 'rb') as f:
        idx2rcmc = pickle.load(f)
    assert args.drugcombidx in idx2rcmc.keys(), \
        f'Drug combination {args.drugcombidx} not found'

    assert args.trainsize > 0, f'Trainsize must be positive but got {args.trainsize}'
    assert args.refsize > 0, f'Refsize must be positive but got {args.refsize}'

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
    assert args.valevery > 0, f'valevery must be positive but got {args.valevery}'

    if args.seed is not None:
        assert args.seed >= 0, f'seed must be non-negative but got {args.seed}'

    ## FIX(GIF): basic validation of sigma schedule
    if hasattr(args, 'fixgif') and args.fixgif:
        assert args.fixgif_sigma in ['const', 'bb'], \
            f"Unsupported --fixgif-sigma '{args.fixgif_sigma}', use 'const' or 'bb'"
        assert args.fixgif_sigma_scale > 0, \
            f'--fixgif-sigma-scale must be positive but got {args.fixgif_sigma_scale}'

    ## Time embedding validation
    if args.time_embed:
        assert args.time_embed_dim > 0, \
            f'--time-embed-dim must be positive when --time-embed is set'
        assert args.time_embed_scale > 0, \
            f'--time-embed-scale must be positive'

    ## Time sampling validation
    assert args.time_sample in ['uniform', 'beta'], \
        f"Unsupported --time-sample '{args.time_sample}', use 'uniform' or 'beta'"
    if args.time_sample == 'beta':
        assert args.time_beta_a > 1.0, \
            f'--time-beta-a should be > 1 to avoid edges; got {args.time_beta_a}'

    return args


def set_up_exp(args: argparse.Namespace) -> None:
    '''Create expdir if not exist and dump json argfile'''
    os.makedirs(args.expname, exist_ok=args.existok)
    with open(os.path.join(args.expname, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)


def main() -> None:
    args = parse_args()
    args = chk_fmt_args(args)
    set_up_exp(args)

    print('\nLoading data...')
    data = np.load(args.data)
    dynmask = build_indexer(OBS, dropvars=CONSTOBS)
    data = data[:, :, :, dynmask]

    dyn_if_vars = [dynvar for dynvar in DYNOBS if '_IF' in dynvar]
    dynifmask = build_indexer(DYNOBS, dropvars=dyn_if_vars)
    data = data[:, :, :, dynifmask]
    data = data[args.drugcombidx]

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
    obs_scaler = StandardScaler()
    hid_scaler = StandardScaler()

    obs_scaler.partial_fit(data_train_snapshots[:, :, obsmask].reshape((-1, dobs)))
    obs_scaler.partial_fit(data_train_refs.reshape((-1, dobs)))
    hid_scaler.fit(data_train_snapshots[:, :, hidmask].reshape((-1, dhid)))

    data_train_snapshots_scaled = np.zeros_like(data_train_snapshots)
    # data_train_refs_scaled = np.zeros_like(data_train_refs)
    data_val_snapshots_scaled = np.zeros_like(data_val_snapshots)
    # data_val_refs_scaled = np.zeros_like(data_val_refs)

    data_train_snapshots_scaled[:, :, obsmask] = obs_scaler.transform(
        data_train_snapshots[:, :, obsmask].reshape((-1, dobs))
    ).reshape(data_train_snapshots[:, :, obsmask].shape)
    data_train_snapshots_scaled[:, :, hidmask] = hid_scaler.transform(
        data_train_snapshots[:, :, hidmask].reshape((-1, dhid))
    ).reshape(data_train_snapshots[:, :, hidmask].shape)
    data_train_refs_scaled = obs_scaler.transform(
        data_train_refs.reshape((-1, dobs))
    ).reshape(data_train_refs.shape)

    data_val_snapshots_scaled[:, :, obsmask] = obs_scaler.transform(
        data_val_snapshots[:, :, obsmask].reshape((-1, dobs))
    ).reshape(data_val_snapshots[:, :, obsmask].shape)
    data_val_snapshots_scaled[:, :, hidmask] = hid_scaler.transform(
        data_val_snapshots[:, :, hidmask].reshape((-1, dhid))
    ).reshape(data_val_snapshots[:, :, hidmask].shape)
    data_val_refs_scaled = obs_scaler.transform(
        data_val_refs.reshape((-1, dobs))
    ).reshape(data_val_refs.shape)

    print('\nConstructing Sampler...')
    train_sampler = TrajAugCFMSampler(
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
        gpr_nt=args.gprnt,
        rbfd_scale=args.rbfdistscale,
        reg=args.reg,
        seed=args.seed,
        fixgif=args.fixgif,
        fixgif_sigma=args.fixgif_sigma,
        fixgif_sigma_scale=args.fixgif_sigma_scale,
        fixgif_sigma_eps=args.fixgif_sigma_eps,
        time_sample=args.time_sample,
        time_beta_a=args.time_beta_a,
        score_gauss=args.score_gauss,
    )
    val_sampler = TrajAugCFMSampler(
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
        gpr_nt=args.gprnt,
        rbfd_scale=args.rbfdistscale,
        reg=args.reg,
        seed=args.seed if args.seed is None else args.seed+1,
        fixgif=args.fixgif,
        fixgif_sigma=args.fixgif_sigma,
        fixgif_sigma_scale=args.fixgif_sigma_scale,
        fixgif_sigma_eps=args.fixgif_sigma_eps,
        time_sample=args.time_sample,
        time_beta_a=args.time_beta_a,
        score_gauss=args.score_gauss,
    )
    trainloader = DataLoader(train_sampler, batch_size=None)
    valloader = DataLoader(val_sampler, batch_size=None)

    print('\nConstructing Model...')
    d_in = data_train_snapshots.shape[-1]
    d_out = d_in
    w = args.width
    h = args.depth
    time_feat_dim = (2 * args.time_embed_dim) if args.time_embed else 1
    if args.score_head:
        from trajaugcfm.models import FlowScoreMLP
        model = FlowScoreMLP(d_in + time_feat_dim, d_out, w=w, h=h)
    else:
        model = MLP(d_in + time_feat_dim, d_out, w=w, h=h)
    print(model)

    device = 'cuda' if (not args.nogpu) and torch.cuda.is_available() else 'cpu'
    print('device:', device)
    model = model.to(device)
    lr = args.lr
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossname = args.lossname
    epochs = args.epochs
    val_every = args.valevery
    progress = args.progress

    print('\nTraining model...')
    metrics_log_path = os.path.join(args.expname, 'metrics.csv') if args.log_metrics else None

    train_losses, val_losses = train(
        model,
        opt,
        trainloader,
        valloader,
        lossname,
        epochs,
        val_every,
        progress,
        device,
        gradclip_max_norm=1.0,
        lr_scheduler='cosine',
        eta_min=1e-5,
        val_mean_reduction=args.val_mean_reduction,
        time_embed=args.time_embed,
        time_embed_dim=args.time_embed_dim,
        time_embed_scale=args.time_embed_scale,
        time_embed_seed=args.seed,
        metrics_log_path=metrics_log_path,
        score_head=args.score_head,
        score_lambda=args.score_lambda,
    )

    print('\nSaving results and plotting losses...')
    torch.save(model.state_dict(), os.path.join(args.expname, 'model.pt'))
    np.savez(
        os.path.join(args.expname, 'losses.npz'),
        train=train_losses,
        val=val_losses
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    train_step_space = np.arange(epochs)
    val_step_space = np.arange(val_losses.shape[0]) * val_every
    val_step_space[-1] = epochs
    ax.grid(visible=True, alpha=0.3)
    ax.plot(train_step_space, train_losses.mean(axis=1), label='train loss')
    # ax.plot(val_step_space, val_losses, label='val loss')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(args.expname, 'loss.png'))
    fig.savefig(os.path.join(args.expname, 'loss.pdf'))


if __name__ == '__main__':
    main()
