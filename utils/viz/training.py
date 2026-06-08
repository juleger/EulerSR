from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.tri as mtri
from utils.viz._style import VAR_NAMES, _DPI
from utils.refs import to_mach
if TYPE_CHECKING:
    from models.base import SRModel

_PROC_RE = re.compile(r'^aoa([+-]?\d+\.\d+)_m(\d+\.\d+)$')


def plot_training_curves(train_losses, val_losses, val_l2s, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=_DPI, constrained_layout=True)
    ep = np.arange(1, len(train_losses) + 1)
    epv = np.linspace(1, len(train_losses), len(val_losses))
    axes[0].semilogy(ep,  train_losses, label='train', lw=1.5)
    axes[0].semilogy(epv, val_losses,   label='val',   lw=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    val_l2s = np.array(val_l2s)
    for vi, vn in enumerate(VAR_NAMES):
        axes[1].semilogy(epv, val_l2s[:, vi], label=vn, lw=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Relative L2')
    axes[1].set_title('Validation L2')
    axes[1].legend()
    plt.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def plot_prediction(model: SRModel, knn: dict, path: str | Path,
        stats: np.ndarray, mesh_hr, out_dir: Path, tag: str = '', mesh_lr=None):
    import jax.numpy as jnp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = Path(path)
    m = _PROC_RE.match(path.stem)
    aoa_in, mach_in = float(m.group(1)), float(m.group(2))

    d = np.load(path)
    hr_prim = d['hr_primitives'].astype(np.float32)
    lr_prim = d['lr_primitives'].astype(np.float32)
    hr_pos = d['hr_node_pos'].astype(np.float32)
    lr_pos = d['lr_node_pos'].astype(np.float32)
    mu, sig = stats['mu'].astype(np.float32), stats['sig'].astype(np.float32)

    from preprocessing.dataset import _MACH_MID, _MACH_SCALE, _AOA_SCALE
    pts = np.concatenate([hr_pos, lr_pos], axis=0).astype(np.float64)
    pos_center = ((pts.max(0) + pts.min(0)) / 2).astype(np.float32)
    pos_scale = float((pts.max(0) - pts.min(0)).max() / 2)

    N = hr_pos.shape[0]
    hr_pos_n = (hr_pos - pos_center) / pos_scale
    lr_pos_n = (lr_pos - pos_center) / pos_scale
    hr_feat = jnp.array(np.stack([
        hr_pos_n[:, 0], hr_pos_n[:, 1],
        np.full(N, (mach_in - _MACH_MID) / _MACH_SCALE, np.float32),
        np.full(N, aoa_in / _AOA_SCALE, np.float32),
    ], axis=1))

    lr_prim_n = (lr_prim - mu) / sig
    lr_parts  = [lr_prim_n, lr_pos_n[:, 0:1], lr_pos_n[:, 1:2]]
    if 'lr_primitives_grad' in d:
        lr_grad = d['lr_primitives_grad'].astype(np.float32)
        grad_p = lr_grad[:, 3, :]
        div_u = lr_grad[:, 1, 0] + lr_grad[:, 2, 1]
        lr_parts += [np.arcsinh(grad_p / (sig[3] + 1e-8)),
                    np.arcsinh(div_u  / (sig[1] + 1e-8))[:, None]]
    lr_feat = jnp.array(np.concatenate(lr_parts, axis=1))

    pred = np.array(model.predict(hr_feat, lr_feat, knn)) * sig + mu

    mach_lr = to_mach(lr_prim)
    mach_hr = to_mach(hr_prim)
    mach_pred = to_mach(pred)
    mach_err = np.abs(mach_pred - mach_hr)
    l2 = float(np.sqrt(((mach_pred - mach_hr) ** 2).sum() / ((mach_hr ** 2).sum() + 1e-8)))

    def _tri(m):
        return mtri.Triangulation(np.asarray(m.points[:, 0]), np.asarray(m.points[:, 1]), np.asarray(m.tris))
    
    triang_hr = _tri(mesh_hr)
    triang_lr = _tri(mesh_lr) if mesh_lr is not None else triang_hr
    vmin, vmax = float(mach_hr.min()), float(mach_hr.max())

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), dpi=_DPI,
                              constrained_layout=True)

    panels = [
        (axes[0], mach_lr,   triang_lr, 'Basse Résolution',  vmin, vmax, 'viridis'),
        (axes[1], mach_hr,   triang_hr, 'Haute Résolution',  vmin, vmax, 'viridis'),
        (axes[2], mach_pred, triang_hr, 'Prédiction SR',     vmin, vmax, 'viridis'),
        (axes[3], mach_err,  triang_hr, 'Erreur |ΔMach|',   0,    None, 'hot'),
    ]
    tpcs = []
    for ax, field, triang, label, vm, vM, cmap in panels:
        tpc = ax.tripcolor(triang, facecolors=field, cmap=cmap, vmin=vm, vmax=vM)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(label, fontsize=10, fontweight='bold', pad=4)
        tpcs.append(tpc)

    cb1 = fig.colorbar(tpcs[1], ax=axes[:3], orientation='horizontal',
                       location='bottom', shrink=0.82, pad=0.02, aspect=45)
    cb1.set_label('Mach', fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    cb2 = fig.colorbar(tpcs[3], ax=axes[3], orientation='horizontal',
                       location='bottom', shrink=0.82, pad=0.02, aspect=18)
    cb2.set_label('|ΔMach|', fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    model_name = model.__class__.__name__
    fig.suptitle(f'{model_name}  —  M = {mach_in:.2f}   AoA = {aoa_in:.1f}°   L₂ = {l2:.4f}',
                 fontsize=14, fontweight='bold', y=1.05)

    plt.savefig(out_dir / f'pred{tag}_mach.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)


