from typing import Literal

import jaxtyping as jt
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trajaugcfm.utils import torch_bmv


def train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    tsxt: jt.Real[torch.Tensor, 'batch din'],
    ut: jt.Real[torch.Tensor, 'batch dout'],
    eps: jt.Real[torch.Tensor, 'batch dout'],
    lt: jt.Real[torch.Tensor, 'batch dout dout'] | None,
    lossfn: nn.Module,
    gradclip_max_norm: float | None,
    score: bool,
) -> tuple[float, float | None]:
    opt.zero_grad()
    vt, st = model(tsxt)
    loss = lossfn(vt, ut)
    flow_loss = loss.detach().cpu().item()
    if score:
        lambda_st = torch_bmv(lt, st)
        ## negative eps in loss because loss = || lambda_st + eps ||^2
        loss2 = lossfn(lambda_st, -eps)
        score_loss = loss2.detach().cpu().item()
        loss += loss2
    else:
        score_loss = None

    loss.backward()
    if gradclip_max_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradclip_max_norm)
    opt.step()

    return flow_loss, score_loss


def train_epoch(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    dataloader: DataLoader,
    lossfn: nn.Module,
    gradclip_max_norm: float | None,
    score: bool,
    progress: bool,
    device: Literal['cuda', 'cpu']
) -> tuple[jt.Real[np.ndarray, 'nsteps'], jt.Real[np.ndarray, 'nsteps'] | None]:
    epoch_flow_losses = np.zeros(len(dataloader))
    epoch_score_losses = np.zeros(len(dataloader)) if score else None

    if progress:
        pbar = tqdm.tqdm(
            total=len(dataloader),
            desc='Train Epoch Steps',
            leave=None
        )

    model.train()
    for i, batch in enumerate(dataloader):
        ts, xt, ut, eps, lt = batch
        tsxt = torch.concat((ts, xt), dim=-1)
        tsxt = tsxt.to(device)
        ut = ut.to(device)
        if score:
            eps = eps.to(device)
            lt = lt.to(device)
        flow_loss, score_loss = train_step(
            model,
            opt,
            tsxt,
            ut,
            eps,
            lt,
            lossfn,
            gradclip_max_norm,
            score,
        )
        epoch_flow_losses[i] = flow_loss
        if score:
            epoch_score_losses[i] = score_loss

        if progress:
            pbar.update(1)
    if progress:
        pbar.close()

    return epoch_flow_losses, epoch_score_losses


def val_step(
    model: nn.Module,
    dataloader: DataLoader,
    lossfn: nn.Module,
    score: bool,
    progress: bool,
    device: Literal['cuda', 'cpu']
) -> tuple[float, float | None]:
    flow_loss = 0.
    score_loss = 0. if score else None

    if progress:
        pbar = tqdm.tqdm(
            total=len(dataloader),
            desc='Val Epoch Steps',
            leave=None
        )

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            ts, xt, ut, eps, lt = batch
            tsxt = torch.concat((ts, xt), dim=-1)
            tsxt = tsxt.to(device)
            ut = ut.to(device)
            vt, st = model(tsxt)
            flow_loss += lossfn(vt, ut).detach().cpu().item()
            if score:
                eps = eps.to(device)
                lt = lt.to(device)
                lambda_st = torch_bmv(lt, st)
                ## negative eps in loss because loss = || lambda_st + eps ||^2
                score_loss += lossfn(lambda_st, -eps).detach().cpu().item()

            if progress:
                pbar.update(1)

    if progress:
        pbar.close()

    flow_loss /= len(dataloader)
    if score:
        score_loss /= len(dataloader)
    return flow_loss, score_loss


## TODO: maybe add typevar and change return signature?
## TODO: return learning rate over epochs for plotting?
def train(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler.LRScheduler | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lossfn: nn.Module,
    epochs: int,
    val_every: int,
    gradclip_max_norm: float | None,
    score: bool,
    progress: bool,
    device: Literal['cuda', 'cpu']
) -> tuple[jt.Real[np.ndarray, 'epochs nsteps'], jt.Real[np.ndarray, 'epochs nsteps'] | None, jt.Real[np.ndarray, 'nvals'], jt.Real[np.ndarray, 'nvals'] | None]:
    if val_every > 0:
        nvals, r = divmod(epochs, val_every)
        nvals += 1 if r > 0 else 0  ## val_every does not evenly divide epochs
    else:
        nvals = 0
    nvals += 1  ## for final val step after training

    train_flow_losses = np.zeros((epochs, len(train_loader)))
    train_score_losses = np.zeros((epochs, len(train_loader))) if score else None
    val_flow_losses = np.zeros(nvals)
    val_score_losses = np.zeros(nvals) if score else None

    if progress:
        pbar = tqdm.tqdm(total=epochs, desc='Training Epochs')
    else:
        pbar = None

    j = 0  ## val counter
    for i in range(epochs):
        if i % val_every == 0:
            flow_loss, score_loss = val_step(
                model,
                val_loader,
                lossfn,
                score,
                progress,
                device
            )
            val_flow_losses[j] = flow_loss
            if score:
                val_score_losses[j] = score_loss
            j += 1

        epoch_flow_losses, epoch_score_losses = train_epoch(
            model,
            opt,
            train_loader,
            lossfn,
            gradclip_max_norm,
            score,
            progress,
            device
        )
        train_flow_losses[i] = epoch_flow_losses
        if score:
            train_score_losses[i] = epoch_score_losses

        if lr_sched is not None:
            lr_sched.step()

        if progress:
            pbar.update(1)

    if progress:
        pbar.close()

    ## final validation step
    flow_loss, score_loss = val_step(
        model,
        val_loader,
        lossfn,
        score,
        progress,
        device
    )
    val_flow_losses[-1] = flow_loss
    if score:
        val_score_losses[-1] = score_loss

    return train_flow_losses, train_score_losses, val_flow_losses, val_score_losses


def main() -> None:
    import os

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from trajaugcfm.constants import (
        BASEDIR,
        DATADIR,
        RESDIR,
        CONSTOBS,
        DYNOBS,
        OBS,
    )
    from trajaugcfm.models import (
        MLP,
        FlowScoreMLP,
        flowscore_wrapper
    )
    from trajaugcfm.sampler import (
        build_sampler_class,
        GCFMSamplerBase
    )
    from trajaugcfm.utils import build_indexer

    experiment = 'mix_ics'
    data = np.load(os.path.join(DATADIR, experiment, 'data.npy'))  ## (drugcombs, N, T, *dims)
    dynmask = build_indexer(OBS, dropvars=CONSTOBS)
    data = data[:, :, :, dynmask]

    dyn_if_vars = [dynvar for dynvar in DYNOBS if '_IF' in dynvar]
    dynifmask = build_indexer(DYNOBS, dropvars=dyn_if_vars)
    data = data[:, :, :, dynifmask]
    dmso = data[0]
    print('dmso shape', dmso.shape)

    obsmask = np.zeros(data.shape[-1], dtype=bool)
    obsidxs = [0, 1, 2]
    obsmask[obsidxs] = True
    hidmask = ~obsmask
    tidxs = [0, 400]
    dobs = obsmask.sum()
    dhid = hidmask.sum()
    d = dobs + dhid

    dmso_train, dmso_val = train_test_split(
        dmso, train_size=0.8, random_state=42
    )
    print('dmso train shape', dmso_train.shape)
    dmso_train_snapshots, dmso_train_refs = train_test_split(
        dmso_train, train_size=0.8, random_state=42
    )
    dmso_train_snapshots = dmso_train_snapshots[:, tidxs]
    dmso_train_refs = dmso_train_refs[:, :, obsmask]
    print('dmso train snapshots shape', dmso_train_snapshots.shape)
    print('dmso train refs shape', dmso_train_refs.shape)

    print('dmso val shape', dmso_val.shape)
    dmso_val_snapshots, dmso_val_refs = train_test_split(
        dmso_val, train_size=0.8, random_state=42
    )
    dmso_val_snapshots = dmso_val_snapshots[:, tidxs]
    dmso_val_refs = dmso_val_refs[:, :, obsmask]
    print('dmso val snapshots shape', dmso_val_snapshots.shape)
    print('dmso val refs shape', dmso_val_refs.shape)


    obs_scaler = StandardScaler()
    hid_scaler = StandardScaler()

    obs_scaler.partial_fit(dmso_train_snapshots[:, :, obsmask].reshape((-1, dobs)))
    obs_scaler.partial_fit(dmso_train_refs.reshape((-1, dobs)))
    hid_scaler.fit(dmso_train_snapshots[:, :, hidmask].reshape((-1, dhid)))

    dmso_train_snapshots_scaled = np.zeros_like(dmso_train_snapshots)
    dmso_train_refs_scaled = np.zeros_like(dmso_train_refs)
    dmso_val_snapshots_scaled = np.zeros_like(dmso_val_snapshots)
    dmso_val_refs_scaled = np.zeros_like(dmso_val_refs)

    dmso_train_snapshots_scaled[:, :, obsmask] = obs_scaler.transform(
        dmso_train_snapshots[:, :, obsmask].reshape((-1, dobs))
    ).reshape(dmso_train_snapshots[:, :, obsmask].shape)
    dmso_train_snapshots_scaled[:, :, hidmask] = hid_scaler.transform(
        dmso_train_snapshots[:, :, hidmask].reshape((-1, dhid))
    ).reshape(dmso_train_snapshots[:, :, hidmask].shape)
    dmso_train_refs_scaled[:] = obs_scaler.transform(
        dmso_train_refs.reshape((-1, dobs))
    ).reshape(dmso_train_refs.shape)

    dmso_val_snapshots_scaled[:, :, obsmask] = obs_scaler.transform(
        dmso_val_snapshots[:, :, obsmask].reshape((-1, dobs))
    ).reshape(dmso_val_snapshots[:, :, obsmask].shape)
    dmso_val_snapshots_scaled[:, :, hidmask] = hid_scaler.transform(
        dmso_val_snapshots[:, :, hidmask].reshape((-1, dhid))
    ).reshape(dmso_val_snapshots[:, :, hidmask].shape)
    dmso_val_refs_scaled[:] = obs_scaler.transform(
        dmso_val_refs.reshape((-1, dobs))
    ).reshape(dmso_val_refs.shape)


    time_sampler = 'uniform'
    use_time_enrich = True
    time_enrich = 'rff'
    flow = 'anisotropic'
    flow_bridge = 'schrodinger'
    score = False
    score_shape = 'anisotropic'

    GCFMSampler = build_sampler_class(
        time_sampler=time_sampler,
        use_time_enrich=use_time_enrich,
        time_enrich=time_enrich,
        flow=flow,
        flow_bridge=flow_bridge,
        score=score,
        score_shape=score_shape
    )
    print('Mixins:', GCFMSampler.get_mixin_names())

    seed = 1000
    train_prng = np.random.default_rng(seed=seed)
    val_prng = np.random.default_rng(seed=seed+1)
    k = 2
    n = 16
    b = 4
    nt = 8
    rbfk_scale = 0.1
    # rbfk_bounds = (0.05, 5)
    rbfk_bounds = 'fixed'
    gpr_nt = 10
    rbfd_scale = 1.
    reg = 1e-6
    # sigma = 1.0
    sigma = 0.15
    sb_reg = 1e-8
    # sb_reg = 0.05
    # sb_reg = 1e-2
    beta_a = 2.0
    rff_seed = 2000
    rff_scale = 1.0
    rff_dim = 3

    train_sampler = GCFMSampler(
        train_prng,
        dmso_train_snapshots_scaled,
        dmso_train_refs_scaled,
        obsmask,
        tidxs,
        k,
        n,
        b,
        nt,
        rbfk_scale=rbfk_scale,
        rbfk_bounds=rbfk_bounds,
        gpr_nt=gpr_nt,
        reg=reg,
        sigma=sigma,
        sb_reg=sb_reg,
        beta_a=beta_a,
        rff_seed=rff_seed,
        rff_scale=rff_scale,
        rff_dim=rff_dim
    )
    val_sampler = GCFMSampler(
        val_prng,
        dmso_val_snapshots_scaled,
        dmso_val_refs_scaled,
        obsmask,
        tidxs,
        k,
        n,
        b,
        nt,
        rbfk_scale=rbfk_scale,
        rbfk_bounds=rbfk_bounds,
        gpr_nt=gpr_nt,
        reg=reg,
        sigma=sigma,
        sb_reg=sb_reg,
        beta_a=beta_a,
        rff_seed=rff_seed,
        rff_scale=rff_scale,
        rff_dim=rff_dim
    )

    train_loader = DataLoader(train_sampler, batch_size=None)
    val_loader = DataLoader(val_sampler, batch_size=None)

    # train_sampler = TrajAugCFMSampler(
        # dmso_train_snapshots_scaled,
        # dmso_train_refs_scaled,
        # obsmask,
        # tidxs,
        # k,
        # b,
        # nt,
        # rbfk_scale=rbfk_scale,
        # rbfk_bounds=rbfk_bounds,
        # gpr_nt=gpr_nt,
        # rbfd_scale=rbfd_scale,
        # reg=reg,
        # seed=seed,
    # )
    # val_sampler = TrajAugCFMSampler(
        # dmso_val_snapshots_scaled,
        # dmso_val_refs_scaled,
        # obsmask,
        # tidxs,
        # k,
        # b,
        # nt,
        # rbfk_scale=rbfk_scale,
        # rbfk_bounds=rbfk_bounds,
        # gpr_nt=gpr_nt,
        # rbfd_scale=rbfd_scale,
        # reg=reg,
        # seed=seed+1,
    # )
    # trainloader = DataLoader(train_sampler, batch_size=None)
    # valloader = DataLoader(val_sampler, batch_size=None)

    # print('enumerating trainloader...')
    # for ts, xt, ut in trainloader:
        # pass
    # print('enumerating valloader...')
    # for ts, xt, ut in valloader:
        # pass

    d_vars = dmso_train_snapshots.shape[-1]
    d_out = d_vars
    w = 64
    h = 2
    if use_time_enrich:
        if time_enrich == 'rff':
            d_time = rff_dim * 2
    else:
        d_time = 1

    d_in = d_vars + d_time
    if score:
        model = FlowScoreMLP(d_in, d_out, w=w, h=h)
    else:
        model = MLP(d_in, d_out, w=w, h=h)
        model = flowscore_wrapper(model)
    print(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    model = model.to(device)
    lr = 5e-5
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_sched = None
    lossname = 'MSELoss'
    lossfn = getattr(nn, lossname)()
    epochs = 50
    val_every = 10
    # gradclip_max_norm = None
    gradclip_max_norm = 1.0
    progress = True
    # progress = False

    # train_losses, val_losses = train(
        # model,
        # opt,
        # trainloader,
        # valloader,
        # lossname,
        # epochs,
        # val_every,
        # progress,
        # device
    # )
    train_flow_losses, train_score_losses, val_flow_losses, val_score_losses = train2(
        model,
        opt,
        lr_sched,
        train_loader,
        val_loader,
        lossfn,
        epochs,
        val_every,
        gradclip_max_norm,
        score,
        progress,
        device
    )

    # print(train_losses.shape)
    # print(val_losses.shape)
    # print(val_losses[[0, 1, 2]], val_losses[[-3, -2, -1]])

    fig, axs = plt.subplots(nrows=2, figsize=(8, 8), sharex=True)
    train_step_space = np.arange(epochs) + 1
    val_step_space = np.arange(val_flow_losses.shape[0]) * val_every
    val_step_space[-1] = epochs
    axs[0].grid(visible=True)
    axs[0].plot(train_step_space, train_flow_losses.mean(axis=1), label='train flow loss')
    axs[0].plot(val_step_space, val_flow_losses, label='val loss')
    axs[0].legend(loc='upper right')
    if score:
        axs[1].grid(visible=True)
        axs[1].plot(train_step_space, train_score_losses.mean(axis=1), label='train score loss')
        axs[1].plot(val_step_space, val_score_losses, label='val loss')
        axs[1].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('loss.png')
    fig.savefig('loss.pdf')


if __name__ == '__main__':
    main()
