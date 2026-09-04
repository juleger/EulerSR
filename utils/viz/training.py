from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.tri as mtri
from utils.viz._style import VAR_NAMES, _DPI
from utils.refs import to_mach, _PROC_RE
from utils.layout import load_sample
if TYPE_CHECKING:
    from models.base import SRModel


def plot_training_curves(train_losses, val_losses, val_l2s, out_path,
                         lr_track=None, idw_l2_ref=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=_DPI, constrained_layout=True)

    ep = np.arange(1, len(train_losses) + 1)
    epv = np.linspace(1, max(len(train_losses), 1), max(len(val_losses), 1))

    # — Loss (train + val) —
    axes[0].semilogy(ep, train_losses, label='train', lw=1.5)
    if val_losses:
        axes[0].semilogy(epv, val_losses, label='val', lw=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    # — Val L2 par variable + baselines IDW —
    val_l2s_np = np.array(val_l2s) if val_l2s else np.empty((0, 5))
    labels = VAR_NAMES + ['Mach']
    for vi in range(val_l2s_np.shape[1]):
        axes[1].semilogy(epv, val_l2s_np[:, vi], label=labels[vi], lw=1.5)
    if idw_l2_ref is not None:
        idw_vals = list(idw_l2_ref['l2']) + [idw_l2_ref['mach']]
        for vi, idw_v in enumerate(idw_vals[:val_l2s_np.shape[1]]):
            axes[1].axhline(idw_v, ls='--', lw=0.9, alpha=0.45, color=f'C{vi}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('L2 relative (physique)')
    axes[1].set_title('Validation L2  (-- = IDW baseline)' if idw_l2_ref else 'Validation L2')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)


def _snapshot_feats(d, mach_in: float, aoa_in: float, stats,
                    coord_norm: str = 'object', mesh_meta: dict | None = None,
                    mach_norm: tuple[float, float] | None = None):
    """Construit (hr_feat, lr_feat) jnp pour predict() + primitives physiques du snapshot.

    mach_norm = (mid, scale) résolu pour le run d'entraînement en cours (cf.
    train.py) -- repli sur l'historique (0.7, 3.0) si non fourni."""
    import jax.numpy as jnp

    hr_prim = d['hr_primitives'].astype(np.float32)
    lr_prim = d['lr_primitives'].astype(np.float32)
    hr_pos = d['hr_node_pos'].astype(np.float32)
    lr_pos = d['lr_node_pos'].astype(np.float32)
    mu, sig = stats['mu'].astype(np.float32), stats['sig'].astype(np.float32)

    from preprocessing.dataset import _MACH_MID, _MACH_SCALE, _AOA_SCALE
    from utils.coords import center_scale
    ctr, scl = center_scale(hr_pos, lr_pos, coord_norm, mesh_meta)
    pos_center = ctr.astype(np.float32)
    pos_scale = scl
    mach_mid, mach_scale = mach_norm if mach_norm is not None else (_MACH_MID, _MACH_SCALE)

    N = hr_pos.shape[0]
    hr_pos_n = (hr_pos - pos_center) / pos_scale
    lr_pos_n = (lr_pos - pos_center) / pos_scale
    hr_feat = jnp.array(np.stack([
        hr_pos_n[:, 0], hr_pos_n[:, 1],
        np.full(N, (mach_in - mach_mid) / mach_scale, np.float32),
        np.full(N, aoa_in / _AOA_SCALE, np.float32),
    ], axis=1))

    lr_prim_n = (lr_prim - mu) / sig
    lr_parts = [lr_prim_n, lr_pos_n[:, 0:1], lr_pos_n[:, 1:2]]
    if 'lr_primitives_grad' in d:
        lr_grad = d['lr_primitives_grad'].astype(np.float32)
        grad_p = lr_grad[:, 3, :]
        div_u = lr_grad[:, 1, 0] + lr_grad[:, 2, 1]
        lr_parts += [np.arcsinh(grad_p / (sig[3] + 1e-8)),
                     np.arcsinh(div_u / (sig[1] + 1e-8))[:, None]]
    else:
        # Fichiers sans gradients LR : zéros pour maintenir la dimension 9
        # (colonnes 6-8 utilisées par les modèles avec use_lr_grad=True)
        N_lr = lr_prim.shape[0]
        lr_parts += [np.zeros((N_lr, 2), np.float32),
                     np.zeros((N_lr, 1), np.float32)]
    lr_feat = jnp.array(np.concatenate(lr_parts, axis=1))
    return hr_feat, lr_feat, hr_prim, lr_prim


def _tri(mesh):
    return mtri.Triangulation(np.asarray(mesh.points[:, 0]),
                              np.asarray(mesh.points[:, 1]),
                              np.asarray(mesh.tris))


def plot_val_panels(model: SRModel, knn: dict, cases: list, stats,
                    mesh_hr, mesh_lr, out_path: str | Path,
                    title: str | None = None, dpi: int = _DPI,
                    coord_norm: str = 'object',
                    mach_norm: tuple[float, float] | None = None):
    # Panneau de suivi : une ligne par cas, colonnes LR / IDW / SR / HR (champ Mach).
    if not cases:
        return
    mu, sig = stats['mu'].astype(np.float32), stats['sig'].astype(np.float32)

    k_use = min(6, np.asarray(knn['idx']).shape[1])
    idx_np = np.asarray(knn['idx'])[:, :k_use].astype(np.int32)
    dist_np = np.asarray(knn['dist'])[:, :k_use].astype(np.float64)
    w = 1.0 / (dist_np ** 2 + 1e-12)
    w = (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

    triang_hr = _tri(mesh_hr)
    triang_lr = _tri(mesh_lr) if mesh_lr is not None else triang_hr

    n = len(cases)
    col_titles = ['LR', 'IDW', 'SR', 'HR']
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * n), dpi=dpi,
                              constrained_layout=True)
    axes = np.atleast_2d(axes)

    for r, (path, label) in enumerate(cases):
        path = Path(path)
        m = _PROC_RE.match(path.stem)
        aoa_in, mach_in = float(m.group(1)), float(m.group(2))
        d = load_sample(path)

        hr_feat, lr_feat, hr_prim, lr_prim = _snapshot_feats(
            d, mach_in, aoa_in, stats, coord_norm, mesh_hr.metadata, mach_norm=mach_norm)

        pred = np.array(model.predict(hr_feat, lr_feat, knn)) * sig + mu

        interp = (w[:, :, None] * lr_prim[idx_np]).sum(axis=1)

        mach_hr = to_mach(hr_prim)
        fields = [(to_mach(lr_prim), triang_lr), (to_mach(interp), triang_hr),
                   (to_mach(pred), triang_hr), (mach_hr, triang_hr)]
        vmin, vmax = float(mach_hr.min()), float(mach_hr.max())

        tpc_last = None
        for c, (field, triang) in enumerate(fields):
            tpc = axes[r, c].tripcolor(triang, facecolors=field,
                                       cmap='viridis', vmin=vmin, vmax=vmax)
            axes[r, c].set_aspect('equal')
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11,
                                     fontweight='bold', pad=4)
            tpc_last = tpc

        # Colorbar unique sur le bord droit de la ligne
        cb = fig.colorbar(tpc_last, ax=axes[r, :], location='right',
                          shrink=0.82, pad=0.02, aspect=22)
        cb.set_label(f'{label}  —  M={mach_in:.2f}  AoA={aoa_in:.1f}°  (Mach)',
                     fontsize=9)
        cb.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

