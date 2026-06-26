from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.tri as mtri
from utils.viz._style import VAR_NAMES, _DPI
from utils.refs import to_mach, _PROC_RE
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


def _snapshot_feats(d, mach_in: float, aoa_in: float, stats):
    """Construit (hr_feat, lr_feat) jnp pour predict() + primitives physiques du snapshot."""
    import jax.numpy as jnp

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


def plot_huber_regime(model: 'SRModel', knn: dict, cases: list, stats,
                      mesh_hr, delta: float, out_path: str | Path,
                      epoch: int | None = None):
    """Carte spatiale du régime Huber (L2/L1) sur les cas de référence.

    Chaque cellule = une variable sur un cas : tripcolor de |pred-tg| / delta,
    vert=L2, jaune=seuil, rouge=L1. Titre = variable + % L1.
    """
    if not cases:
        return
    mu = stats['mu'].astype(np.float32)
    sig = stats['sig'].astype(np.float32)

    triang = _tri(mesh_hr)
    n_cases = len(cases)
    n_vars = 4
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'huber', ['#2ecc71', '#f1c40f', '#e74c3c'], N=256)
    norm = mcolors.Normalize(vmin=0, vmax=2.0)  # delta → jaune, 2×delta → rouge saturé

    fig, axes = plt.subplots(n_cases, n_vars,
                              figsize=(3.5 * n_vars, 3.0 * n_cases),
                              dpi=_DPI, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for r, (path, label) in enumerate(cases):
        path = Path(path)
        m = _PROC_RE.match(path.stem)
        aoa_in, mach_in = float(m.group(1)), float(m.group(2))
        d = np.load(path)

        hr_feat, lr_feat, hr_prim, _ = _snapshot_feats(d, mach_in, aoa_in, stats)
        pred_norm = np.array(model.predict(hr_feat, lr_feat, knn))
        tg_norm = (hr_prim - mu) / sig
        tg_std = np.sqrt((tg_norm ** 2).mean(0) + 1e-8)  # (4,) std par variable
        resid = np.abs(pred_norm - tg_norm) / tg_std  # résidu relatif (N_cells, 4)

        for c, vname in enumerate(VAR_NAMES):
            r_v = resid[:, c]
            l1_pct = float((r_v >= delta).mean()) * 100
            axes[r, c].tripcolor(triang, facecolors=r_v / delta,
                                 cmap=cmap, norm=norm)
            axes[r, c].set_aspect('equal')
            axes[r, c].set_axis_off()
            axes[r, c].set_title(f'{vname}  L1: {l1_pct:.1f}%', fontsize=8, pad=2)

        axes[r, 0].set_ylabel(label, fontsize=8)

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes, location='right', shrink=0.6, pad=0.02, aspect=30)
    cb.set_label(f'|résidu norm.| / δ   (δ={delta})', fontsize=8)
    cb.ax.axhline(1.0, color='k', lw=1.0, ls='--')
    cb.ax.tick_params(labelsize=7)

    ep_str = f'  —  epoch {epoch}' if epoch is not None else ''
    fig.suptitle(f'Régime Huber{ep_str}', fontsize=11, fontweight='bold')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_val_panels(model: SRModel, knn: dict, cases: list, stats,
                    mesh_hr, mesh_lr, out_path: str | Path,
                    title: str | None = None, dpi: int = _DPI):
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
        d = np.load(path)

        hr_feat, lr_feat, hr_prim, lr_prim = _snapshot_feats(d, mach_in, aoa_in, stats)

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

