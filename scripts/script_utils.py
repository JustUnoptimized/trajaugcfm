from collections.abc import Iterator

import jaxtyping as jt
import numpy as np
from matplotlib.axes import Axes
from torch.nn.parameter import Parameter


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


def plot_grad_flow(
    named_params: Iterator[tuple[str, Parameter]],
    ax: Axes
) -> None:
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

