import os
import csv
from typing import Final, Literal

import jaxtyping as jt
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trajaugcfm.constants import (
    BASEDIR,
    DATADIR,
    RESDIR,
    CONSTOBS,
    DYNOBS,
    OBS,
)
from trajaugcfm.sampler import TrajAugCFMSampler
from trajaugcfm.utils import (
    build_indexer
)


def train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    X: jt.Real[torch.Tensor, '*batch din'],
    y: jt.Real[torch.Tensor, '*batch dout'],
    lossfn: nn.Module,
    gradclip_max_norm: float | None = None,
) -> float:
    opt.zero_grad()
    yhat = model(X)
    loss = lossfn(yhat, y)
    loss.backward()
    if gradclip_max_norm is not None and gradclip_max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradclip_max_norm)
    opt.step()
    return loss.detach().cpu().item()


def train_epoch(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    dataloader: DataLoader,
    lossfn: nn.Module,
    progress: bool,
    device: Literal['cuda', 'cpu'],
    gradclip_max_norm: float | None,
    time_embed: bool=False,
    time_embed_dim: int=0,
    time_embed_scale: float=1.0,
    time_embed_B: torch.Tensor | None=None,
) -> jt.Real[np.ndarray, 'epoch']:
    epoch_train_losses = np.zeros(len(dataloader))

    if progress:
        pbar = tqdm.tqdm(
            total=len(dataloader),
            desc='Train Epoch Steps',
            leave=None
        )
    else:
        pbar = None

    model.train()  ## in case previous step was eval
    for i, (ts, xt, ut) in enumerate(dataloader):
        if time_embed:
            Bt = 2 * torch.pi * (ts @ time_embed_B.view(1, -1))
            phi = torch.concat((torch.cos(Bt), torch.sin(Bt)), dim=-1)
            tsxt = torch.concat((phi, xt), dim=-1)
        else:
            tsxt = torch.concat((ts, xt), dim=-1)
        tsxt = tsxt.to(device)
        ut = ut.to(device)
        epoch_train_losses[i] = train_step(model, opt, tsxt, ut, lossfn, gradclip_max_norm)

        if progress:
            pbar.update(1)
    if progress:
        pbar.close()

    return epoch_train_losses


def val_step(
    model: nn.Module,
    dataloader: DataLoader,
    lossfn: nn.Module,
    progress: bool,
    device: Literal['cuda', 'cpu'],
    time_embed: bool=False,
    time_embed_dim: int=0,
    time_embed_scale: float=1.0,
    time_embed_B: torch.Tensor | None=None,
) -> float:
    '''Evals current model using validation split

    Returns mean loss value of full validation data.
    '''
    loss_sum = 0.0
    nsteps = 0

    if progress:
        pbar = tqdm.tqdm(
            total=len(dataloader),
            desc='Val Epoch Steps',
            leave=None
        )
    else:
        pbar = None

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                ts, xt, ut, _ = batch
            else:
                ts, xt, ut = batch
            nsteps += 1
            if time_embed:
                # Random Fourier Features for t with fixed frequencies B
                # phi(t) = [cos(2pi B t), sin(2pi B t)]
                # B shape: (time_embed_dim,), ts shape: (batch, 1)
                Bt = 2 * torch.pi * (ts @ time_embed_B.view(1, -1))
                phi = torch.concat((torch.cos(Bt), torch.sin(Bt)), dim=-1)
                tsxt = torch.concat((phi, xt), dim=-1)
            else:
                tsxt = torch.concat((ts, xt), dim=-1)
            tsxt = tsxt.to(device)
            ut = ut.to(device)
            vt = model(tsxt)
            loss_sum += lossfn(vt, ut).detach().cpu().item()

            if progress:
                pbar.update(1)
    # return average of per-step mean losses
    return loss_sum / max(nsteps, 1)


def train(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    trainloader: DataLoader,
    valloader: DataLoader,
    lossname: str,
    epochs: int,
    val_every: int,
    progress: bool,
    device = Literal['cuda', 'cpu'],
    gradclip_max_norm: float | None = None,
    lr_scheduler: str = 'none',
    eta_min: float = 0.0,
    val_mean_reduction: bool=False,
    time_embed: bool=False,
    time_embed_dim: int=0,
    time_embed_scale: float=1.0,
    time_embed_seed: int | None=None,
    metrics_log_path: str | None=None,
    score_head: bool=False,
    score_lambda: float=0.1,
) -> tuple[jt.Real[np.ndarray, 'nepochs nsteps'], jt.Real[np.ndarray, 'nvals']]:
    if val_every > 0:
        nvals, r = divmod(epochs, val_every)
        nvals += 1 if r > 0 else 0  ## val_every does not evenly divide epochs
    else:
        nvals = 0
    nvals += 1  ## for final val step after training
    train_losses = np.zeros((epochs, len(trainloader)))
    val_losses = np.zeros(nvals)

    trainlossfn = getattr(nn, lossname)(reduction='mean')
    vallossfn = getattr(nn, lossname)(reduction='mean' if val_mean_reduction else 'sum')
    # Optional LR scheduler
    scheduler = None
    if lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=eta_min
        )

    # Optional time embedding: pre-sample fixed frequencies B ~ N(0, scale)
    time_embed_B = None
    if time_embed:
        gen = torch.Generator()
        if time_embed_seed is not None:
            gen.manual_seed(int(time_embed_seed))
        time_embed_B = torch.normal(
            mean=0.0,
            std=float(time_embed_scale),
            size=(int(time_embed_dim),),
            generator=gen,
        )

    # Optional metrics logging
    metrics_file = None
    metrics_writer = None
    if metrics_log_path is not None:
        os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
        metrics_file = open(metrics_log_path, 'w', newline='')
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(['epoch', 'split', 'loss_mean', 'lr'])

    if progress:
        pbar = tqdm.tqdm(total=epochs, desc='Training Epochs')
    else:
        pbar = None

    j = 0  ## val counter
    for i in range(epochs):
        ## do val first!
        if i % val_every == 0:
            # For score-head models, evaluate using the flow head only
            if score_head:
                def _wrapped_model(tsxt):
                    v, _ = model(tsxt)
                    return v
                class _EvalModule(nn.Module):
                    def __init__(self, f):
                        super().__init__()
                        self.f = f
                    def forward(self, x):
                        return self.f(x)
                eval_model = _EvalModule(_wrapped_model).to(device)
                val_losses[j] = val_step(
                    eval_model,
                    valloader,
                    vallossfn,
                    progress,
                    device,
                    time_embed=time_embed,
                    time_embed_dim=time_embed_dim,
                    time_embed_scale=time_embed_scale,
                    time_embed_B=time_embed_B,
                )
            else:
                val_losses[j] = val_step(
                    model,
                    valloader,
                    vallossfn,
                    progress,
                    device,
                    time_embed=time_embed,
                    time_embed_dim=time_embed_dim,
                    time_embed_scale=time_embed_scale,
                    time_embed_B=time_embed_B,
                )
            if metrics_writer is not None:
                lr = opt.param_groups[0]['lr']
                metrics_writer.writerow([i, 'val', float(val_losses[j]), lr])
            j += 1
        if score_head:
            # Manual loop to compute combined loss per step
            epoch_train_losses = np.zeros(len(trainloader))
            if progress:
                pbar_epoch = tqdm.tqdm(total=len(trainloader), desc='Train Epoch Steps', leave=None)
            else:
                pbar_epoch = None
            model.train()
            for k, batch in enumerate(trainloader):
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    ts, xt, ut, st = batch
                else:
                    ts, xt, ut = batch
                    st = None
                if time_embed:
                    Bt = 2 * torch.pi * (ts @ time_embed_B.view(1, -1))
                    phi = torch.concat((torch.cos(Bt), torch.sin(Bt)), dim=-1)
                    tsxt = torch.concat((phi, xt), dim=-1)
                else:
                    tsxt = torch.concat((ts, xt), dim=-1)
                tsxt = tsxt.to(device)
                ut = ut.to(device)
                st = st.to(device) if st is not None else None
                opt.zero_grad()
                v_pred, s_pred = model(tsxt)
                loss_flow = trainlossfn(v_pred, ut)
                if st is not None:
                    loss_score = trainlossfn(s_pred, st)
                else:
                    # Fallback: encourage score to match residual
                    loss_score = trainlossfn(s_pred, ut - v_pred)
                loss = loss_flow + (float(score_lambda) * loss_score)
                loss.backward()
                if gradclip_max_norm is not None and gradclip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradclip_max_norm)
                opt.step()
                epoch_train_losses[k] = float(loss_flow.detach().cpu().item())
                if metrics_writer is not None:
                    lr = opt.param_groups[0]['lr']
                    metrics_writer.writerow([i, 'train_score', float(loss_score.detach().cpu().item()), lr])
                if pbar_epoch is not None:
                    pbar_epoch.update(1)
            if pbar_epoch is not None:
                pbar_epoch.close()
            train_losses[i] = epoch_train_losses
        else:
            train_losses[i] = train_epoch(
                model,
                opt,
                trainloader,
                trainlossfn,
                progress,
                device,
                gradclip_max_norm,
                time_embed=time_embed,
                time_embed_dim=time_embed_dim,
                time_embed_scale=time_embed_scale,
                time_embed_B=time_embed_B,
            )
        if progress:
            pbar.update(1)
        if scheduler is not None:
            scheduler.step()
        if metrics_writer is not None:
            lr = opt.param_groups[0]['lr']
            metrics_writer.writerow([i, 'train', float(train_losses[i].mean()), lr])
    if progress:
        pbar.close()

    ## final val step (use flow head only when score_head)
    if score_head:
        def _wrapped_model(tsxt):
            v, _ = model(tsxt)
            return v
        class _EvalModule(nn.Module):
            def __init__(self, f):
                super().__init__()
                self.f = f
            def forward(self, x):
                return self.f(x)
        eval_model = _EvalModule(_wrapped_model).to(device)
        val_losses[-1] = val_step(
            eval_model,
            valloader,
            vallossfn,
            progress,
            device,
            time_embed=time_embed,
            time_embed_dim=time_embed_dim,
            time_embed_scale=time_embed_scale,
            time_embed_B=time_embed_B,
        )
    else:
        val_losses[-1] = val_step(
            model,
            valloader,
            vallossfn,
            progress,
            device,
            time_embed=time_embed,
            time_embed_dim=time_embed_dim,
            time_embed_scale=time_embed_scale,
            time_embed_B=time_embed_B,
        )
    if metrics_writer is not None:
        lr = opt.param_groups[0]['lr']
        metrics_writer.writerow([epochs, 'val_final', float(val_losses[-1]), lr])
        metrics_file.close()

    return train_losses, val_losses


def plot_grad_flow(named_params, ax):
    '''From discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7'''
    ave_grads = []
    layers = []
    for n, p in named_params:
        if p.requires_grad and ('bias' not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    ax.plot(ave_grads, alpha=0.3, color='b')
    ax.hlines(0, 0, len(ave_grads)+1, linewidth=1, color='k')
    ax.set_xticks(range(0, len(ave_grads), 1), layers, rotation='vertical')
    ax.set_xlim(xmin=0, xmax=len(ave_grads))
    ax.set_xlabel('Layers')
    ax.set_ylabel('Avg Grad')
    ax.set_title('Gradient Flow')
    ax.grid(True)


def main() -> None:
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from trajaugcfm.models import MLP

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


    k = 4
    b = 8
    nt = 10
    rbfk_scale = 1.
    rbfk_bounds = (0.1, 5)
    gpr_nt = 10
    rbfd_scale = 1.
    reg = 1e-8
    seed = 1000

    train_sampler = TrajAugCFMSampler(
        dmso_train_snapshots_scaled,
        dmso_train_refs_scaled,
        obsmask,
        tidxs,
        k,
        b,
        nt,
        rbfk_scale=rbfk_scale,
        rbfk_bounds=rbfk_bounds,
        gpr_nt=gpr_nt,
        rbfd_scale=rbfd_scale,
        reg=reg,
        seed=seed,
    )
    val_sampler = TrajAugCFMSampler(
        dmso_val_snapshots_scaled,
        dmso_val_refs_scaled,
        obsmask,
        tidxs,
        k,
        b,
        nt,
        rbfk_scale=rbfk_scale,
        rbfk_bounds=rbfk_bounds,
        gpr_nt=gpr_nt,
        rbfd_scale=rbfd_scale,
        reg=reg,
        seed=seed+1,
    )
    trainloader = DataLoader(train_sampler, batch_size=None)
    valloader = DataLoader(val_sampler, batch_size=None)

    # print('enumerating trainloader...')
    # for ts, xt, ut in trainloader:
        # pass
    # print('enumerating valloader...')
    # for ts, xt, ut in valloader:
        # pass

    d_in = dmso_train_snapshots.shape[-1]
    d_out = d_in
    w = 64
    h = 2
    model = MLP(d_in+1, d_out, w=w, h=h)
    print(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    model = model.to(device)
    lr = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossname = 'MSELoss'
    epochs = 50
    val_every = 5
    progress = True
    # progress = False

    train_losses, val_losses = train(
        model,
        opt,
        trainloader,
        valloader,
        lossname,
        epochs,
        val_every,
        progress,
        device
    )

    # print(train_losses.shape)
    # print(val_losses.shape)
    # print(val_losses[[0, 1, 2]], val_losses[[-3, -2, -1]])

    fig, ax = plt.subplots(figsize=(8, 4))
    train_step_space = np.arange(epochs)
    val_step_space = np.arange(val_losses.shape[0]) * val_every
    val_step_space[-1] = epochs
    ax.grid(visible=True)
    ax.plot(train_step_space, train_losses.mean(axis=1), label='train loss')
    # ax.plot(val_step_space, val_losses, label='val loss')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('loss.png')
    fig.savefig('loss.pdf')


def modtest() -> None:
    epochs = 18
    val_every = 3
    num_vals, r = divmod(epochs, val_every)
    num_vals += 1 if r > 0 else 0
    num_vals += 1  ## for final evaluation
    print('num vals', num_vals)
    for i in range(epochs):
        if val_every > 0 and i % val_every == 0:
            print(i, 'do eval')

if __name__ == '__main__':
    main()
    # modtest()
