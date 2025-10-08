import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Literal

import jaxtyping as jt
import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
import torch
import torch.nn as nn
from torchsde import sdeint

from trajaugcfm.constants import (
    BASEDIR,
    DATADIR,
    RESDIR,
    CONSTOBS,
    DYNOBS,
    OBS,
)
# from trajaugcfm.sampler import TrajAugCFMSampler
from trajaugcfm.models import (
    MLP,
    FlowScoreMLP,
    flowscore_wrapper
)
# from trajaugcfm.sampler import TimeRFFMixin
from trajaugcfm.utils import (
    build_indexer
)
from script_utils import (
    METRICS_FILENAME,
    MODEL_FILENAME,
    TRAINARGS_FILENAME,
    TRAJGENARGS_FILENAME,
    TRAJGEN_FILENAME,
    int_or_float,
    load_args,
    load_scalers,
    scale_data_with_scalers
)


class TorchTimeRFF:
    def __init__(
        self,
        rff_seed: int,
        rff_scale: float,
        rff_dim: int
    ) -> None:
        prng = np.random.default_rng(seed=rff_seed)
        B = prng.normal(loc=0, scale=rff_scale, size=(1, rff_dim)) * 2 * np.pi
        self.B = torch.from_numpy(B.astype(np.float32))  ## (1, rff_dim)

    def __call__(
        self,
        ts: jt.Float32[torch.Tensor, '#batch']
    ):
        Bt = self.B * ts[:, None]  ## (batch, rff_dim)
        if Bt.dim() == 3:
            ## torch broadcasting does not work the same as numpy broadcasting...
            ## if B has shape [1, rff_dim] and ts has shape [batch, 1]
            ## then B * ts has shape [batch, 1, rff_dim]
            Bt = Bt.squeeze(1)
        cosBt = torch.cos(Bt)
        sinBt = torch.sin(Bt)
        return torch.cat((cosBt, sinBt), dim=1)


class SDE(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(
        self,
        model: nn.Module,
        t_enhancer: TorchTimeRFF | None,
        sigma: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.t_enhancer = t_enhancer
        self.sigma = sigma
        self.NFE = 0

    def f(self, t, y):
        if t.dim() != y.dim():
            ## assume t is scalar with dim == 0 or singleton [t] with dim == 1
            t = t.view(-1, 1)  ## (batch, 1)
        if self.t_enhancer is not None:
            t = self.t_enhancer(t)  ## (batch, rff_dim*2)
        if t.shape[0] == 1:
            t = t.expand(y.shape[0], -1)  ## (batch, d_time)
        x = torch.cat((t, y), dim=1)
        vt, st = self.model(x)
        self.NFE += 1
        return vt + st

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='trajgen')
    expgroup = parser.add_argument_group('exp', 'experiment load args')
    expgroup.add_argument(
        '--expname', type=str, required=True,
        help='Load experiment in results/<expname>/.'
    )

    sdegroup = parser.add_argument_group('sde', 'sde solver args')
    sdegroup.add_argument(
        '--sigma', type=float, default=1.0,
        help='SDE diffusion constant'
    )
    sdegroup.add_argument(
        '--method', type=str, choices=['euler', 'milstein', 'srk'],
        default='euler',
        help='SDE solver. Euler-Maruyama, Milstein, or Stochastic Runge-Kutta.'
    )
    sdegroup.add_argument(
        '--n', type=int, default=20,
        help='Number of initial conditions from validation data t=0 for sde solve.' \
            +' Set to -1 to use all validation data.'
    )
    sdegroup.add_argument(
        '--nt', type=int, default=101,
        help='Number of timepoints in tspan for sde solve.' \
            +' Set to -1 to use all validation time points'
    )

    plotgroup = parser.add_argument_group('plot', 'plot args')
    plotgroup.add_argument(
        '--ncols', type=int, default=4,
        help='Number of subfigure cols in trajectory plot'
    )
    plotgroup.add_argument(
        '--ax-w', type=int_or_float, default=4,
        help='Subplot width'
    )
    plotgroup.add_argument(
        '--ax-h', type=int_or_float, default=3,
        help='Subplot height'
    )
    plotgroup.add_argument(
        '--nrefplot', type=int, default=100,
        help='Number of reference trajs to plot. ' \
            +' Set to -1 to plot all reference trajs.'
    )
    plotgroup.add_argument(
        '--ntrajplot', type=int, default=50,
        help='Number of inferred trajs from sdesolve() to plot. ' \
            +' Set to -1 to plot all inferred trajs.'
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

    return parser.parse_args()


def chk_fmt_args(args: argparse.Namespace) -> argparse.Namespace:
    ## expgroup check
    exppath = os.path.join(RESDIR, args.expname)
    assert os.path.exists(exppath), f'{exppath} not found'
    args.expname = exppath

    ## sdegroup check
    assert args.sigma > 0, f'sigma must be positive but got {args.sigma}'
    assert args.n > 0 or args.n == -1, f'n must be positive or -1 but got {args.n}'
    assert args.nt > 0 or args.nt == -1, f'nt must be positive or -1 but got {args.nt}'

    ## plotgroup check
    assert args.ncols > 0, f'ncols must be positive but got {args.ncols}'
    assert args.ax_w > 0, f'ax-w must be positive but got {args.ax_w}'
    assert args.ax_h > 0, f'ax-h must be positive but got {args.ax_h}'
    assert args.nrefplot > 0 or args.nrefplot == -1, \
        f'nrefplot must be positive or -1 but got {args.nrefplot}'
    assert args.ntrajplot > 0 or args.ntrajplot == -1, \
        f'ntrajplot must be positive or -1 but got {args.ntrajplot}'

    ## miscgroup check
    if args.seed is not None:
        assert args.seed >= 0, f'seed must be non-negative but got {args.seed}'

    return args


def save_trajgen_args(args: dict[str, Any], expname: str) -> None:
    '''Save args for trajgen instance to json file.'''
    trajargs_path = os.path.join(expname, TRAJGENARGS_FILENAME)
    with open(trajargs_path, 'w') as f:
        json.dump(vars(args), f, indent=4)


def main() -> None:
    args = parse_args()
    args = chk_fmt_args(args)
    exp_args = load_args(args.expname, TRAINARGS_FILENAME)
    save_trajgen_args(args, exp_args.expname)

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

    print(f'\nLoading model from {os.path.join(args.expname, MODEL_FILENAME)}...')
    d_vars = data_train_snapshots.shape[-1]
    d_out = d_vars
    w = exp_args.width
    h = exp_args.depth
    if exp_args.use_time_enrich:
        if exp_args.time_enrich == 'rff':
            d_time = exp_args.rff_dim * 2
    else:
        d_time = 1
    d_in = d_vars + d_time
    if exp_args.score:
        model = FlowScoreMLP(d_in, d_out, w=w, h=h)
    else:
        model = MLP(d_in, d_out, w=w, h=h)
        model = flowscore_wrapper(model)

    model.load_state_dict(torch.load(os.path.join(args.expname, MODEL_FILENAME), weights_only=True))
    print(model)

    device = 'cuda' if (not args.nogpu) and torch.cuda.is_available() else 'cpu'
    print('device:', device)
    model = model.to(device)

    t_enhancer = None
    if exp_args.use_time_enrich:
        if exp_args.time_enrich == 'rff':
            t_enhancer = TorchTimeRFF(
                exp_args.rff_seed,
                exp_args.rff_scale,
                exp_args.rff_dim
            )

    model_sde = SDE(
        model,
        t_enhancer,
        args.sigma
    )
    print(model_sde)

    model_sde.eval()
    prng = np.random.default_rng(seed=args.seed)
    nx0 = data_val_snapshots_scaled.shape[0] if args.n == -1 else args.n
    idxs = prng.choice(data_val_snapshots_scaled.shape[0], size=nx0, replace=False)
    x0 = torch.from_numpy(data_val_snapshots_scaled[idxs, 0, :].astype(np.float32))
    nts = data_val_refs_scaled.shape[1] if args.nt == -1 else args.nt
    ts = torch.linspace(0, 1, nts)

    with torch.no_grad():
        trajs = sdeint(
            model_sde,
            x0,
            ts,
            method=args.method
        )
    ## convert to numpy w/ common (N, T, d) shape. Additionally upcast from float32 to float64
    trajs = trajs.swapaxes(0, 1).detach().cpu().numpy().astype(np.float64)  ## (N, T, d)

    ## Save (scaled) trajs for future evaluations
    traj_path = os.path.join(exp_args.expname, TRAJGEN_FILENAME)
    np.save(traj_path, trajs)

    ## Save NFE in metrics file and inferred trajs
    metrics_file = os.path.join(exp_args.expname, METRICS_FILENAME)
    if os.path.exists(metrics_file):
        ## Load saved metrics (if exists)
        with open(metrics_file, 'r') as f:
            metrics_dict = json.load(f)
    else:
        ## Otherwise make a new metrics dict
        metrics_dict = {}

    metrics_dict['NFE'] = model_sde.NFE

    ## save metrics file (and overwrite if exists)
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)


if __name__ == '__main__':
    main()

