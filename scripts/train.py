from typing import Literal

import jaxtyping as jt
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trajaugcfm.utils import torch_bmv
type Result = tuple[
    jt.Real[np.ndarray, 'epochs nsteps'],         ## train flow losses
    jt.Real[np.ndarray, 'epochs nsteps'] | None,  ## train score losses
    jt.Real[np.ndarray, ' nvals'],                 ## val flow losses
    jt.Real[np.ndarray, ' nvals'] | None,          ## val score losses
    jt.Real[np.ndarray, ' epochs']                 ## lrs
]

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
) -> tuple[jt.Real[np.ndarray, ' nsteps'], jt.Real[np.ndarray, ' nsteps'] | None]:
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
) -> Result:
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
    ## TODO: if later modify opt to have multiple param groups
    ##       update this array accordingly
    lrs = np.full(epochs, opt.param_groups[0]['lr'])

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
            ## TODO: if later modify opt to have multiple param groups
            ##       update this array accordingly
            lrs[i] = lr_sched.get_last_lr()[0]
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

    return train_flow_losses, train_score_losses, val_flow_losses, val_score_losses, lrs

