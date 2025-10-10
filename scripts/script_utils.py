from collections.abc import Iterator
import json
import os
from types import SimpleNamespace
from typing import Literal, Unpack

import jaxtyping as jt
import joblib
import numpy as np
from matplotlib.axes import Axes
from sklearn.preprocessing import StandardScaler
from torch.nn.parameter import Parameter

from trajaugcfm.constants import (
    CONSTOBS,
    DYNOBS,
    OBS,
    RESDIR
)
from trajaugcfm.utils import (
    build_indexer
)

type ScaledData = tuple[
    jt.Real[np.ndarray, 'Ntrain margidx dim'],
    jt.Real[np.ndarray, 'Ntrain T dim'],
    jt.Real[np.ndarray, 'Nval margidx dim'],
    jt.Real[np.ndarray, 'Nval T dim'],
]

OBS_SCALER_FILENAME = 'obs_scaler.z'
HID_SCALER_FILENAME = 'hid_scaler.z'
MODEL_FILENAME = 'model.pt'
METRICS_FILENAME = 'metrics.json'
TRAINARGS_FILENAME = 'args.json'
LOSSES_FILENAME = 'losses.npz'
TRAJGENARGS_FILENAME = 'trajgen_args.json'
TRAJGEN_FILENAME = 'trajs_scaled.npy'
EVALARGS_FILENAME = 'eval_args.json'
EVALS_FILENAME = 'evals.npz'


def int_or_float(x: str) -> int | float:
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


def load_args(expname: str, filename) -> SimpleNamespace:
    exppath = os.path.join(RESDIR, expname)
    with open(os.path.join(exppath, filename), 'r') as f:
        exp_args = json.load(f)
    return SimpleNamespace(**exp_args)


def load_data(datapath: str, source: Literal['synth', 'marm'], drugcombidx: int) -> np.ndarray:
    data = np.load(datapath)
    if source == 'marm':
        dynmask = build_indexer(OBS, dropvars=CONSTOBS)
        data = data[:, :, :, dynmask]

        dyn_if_vars = [dynvar for dynvar in DYNOBS if '_IF' in dynvar]
        dynifmask = build_indexer(DYNOBS, dropvars=dyn_if_vars)
        data = data[:, :, :, dynifmask]
        data = data[drugcombidx]

    return data


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


def scale_data(
    data_train_snapshots: jt.Real[np.ndarray, 'Ntrain margidx dim'],
    data_train_refs: jt.Real[np.ndarray, 'Ntrain T dim'],
    data_val_snapshots: jt.Real[np.ndarray, 'Nval margidx dim'],
    data_val_refs: jt.Real[np.ndarray, 'Nval T dim'],
    obsmask: jt.Bool[np.ndarray, 'dim'],
    hidmask: jt.Bool[np.ndarray, 'dim'],
) -> tuple[Unpack[ScaledData], StandardScaler, StandardScaler]:
    dobs = obsmask.sum()
    dhid = hidmask.sum()
    obs_scaler = StandardScaler()
    hid_scaler = StandardScaler()

    obs_scaler.partial_fit(data_train_snapshots[:, :, obsmask].reshape((-1, dobs)))
    obs_scaler.partial_fit(data_train_refs.reshape((-1, dobs)))
    hid_scaler.fit(data_train_snapshots[:, :, hidmask].reshape((-1, dhid)))

    data_train_snapshots_scaled = np.zeros_like(data_train_snapshots)
    data_val_snapshots_scaled = np.zeros_like(data_val_snapshots)

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


    ## only here to use packing/unpacking to keep line lengths short
    scaled_data = (
        data_train_snapshots_scaled,
        data_train_refs_scaled, 
        data_val_snapshots_scaled,
        data_val_refs_scaled
    )

    return *scaled_data, obs_scaler, hid_scaler


def scale_data_with_scalers(
    data_train_snapshots: jt.Real[np.ndarray, 'Ntrain margidx dim'],
    data_train_refs: jt.Real[np.ndarray, 'Ntrain T dim'],
    data_val_snapshots: jt.Real[np.ndarray, 'Nval margidx dim'],
    data_val_refs: jt.Real[np.ndarray, 'Nval T dim'],
    obsmask: jt.Bool[np.ndarray, 'dim'],
    hidmask: jt.Bool[np.ndarray, 'dim'],
    obs_scaler: StandardScaler,
    hid_scaler: StandardScaler
) -> ScaledData:
    dobs = obsmask.sum()
    dhid = hidmask.sum()

    data_train_snapshots_scaled = np.zeros_like(data_train_snapshots)
    data_val_snapshots_scaled = np.zeros_like(data_val_snapshots)

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


    ## only here to use packing/unpacking to keep line lengths short
    scaled_data = (
        data_train_snapshots_scaled,
        data_train_refs_scaled, 
        data_val_snapshots_scaled,
        data_val_refs_scaled
    )

    return scaled_data


def save_scalers(
    outdir: str,
    obs_scaler: StandardScaler,
    hid_scaler: StandardScaler
) -> None:
    joblib.dump(obs_scaler, os.path.join(outdir, OBS_SCALER_FILENAME))
    joblib.dump(hid_scaler, os.path.join(outdir, HID_SCALER_FILENAME))


def load_scalers(outdir: str) -> tuple[StandardScaler, StandardScaler]:
    obs_scaler = joblib.load(os.path.join(outdir, OBS_SCALER_FILENAME))
    hid_scaler = joblib.load(os.path.join(outdir, HID_SCALER_FILENAME))
    return obs_scaler, hid_scaler

