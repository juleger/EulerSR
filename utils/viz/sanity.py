import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

import utils.viz._style  # noqa: F401
from utils.refs import to_mach, REFERENCE_CASES, find_ref_file

_PROC_RE = re.compile(r'^aoa([+-]?\d+\.\d+)_m(\d+\.\d+)$')
_DPI = 500

def _load_pair(path):
    d = np.load(path)
    hr = {'pos': d['hr_node_pos'], 'prim': d['hr_primitives']}
    lr = {'pos': d['lr_node_pos'], 'prim': d['lr_primitives']}
    m = _PROC_RE.match(Path(path).stem)
    return hr, lr, float(m.group(1)), float(m.group(2))   # hr, lr, aoa, mach_in


def _interp_lr_to_hr(lr_pos, hr_pos, lr_field, k: int = 6):
    dists, idxs = cKDTree(lr_pos).query(hr_pos, k=k)
    w = 1.0 / (dists + 1e-10)
    w /= w.sum(axis=1, keepdims=True)
    return (w * lr_field[idxs]).sum(axis=1)


def _triang(mesh):
    pts = np.asarray(mesh.points)
    return mtri.Triangulation(pts[:, 0], pts[:, 1], np.asarray(mesh.tris))

def plot_dataset_distribution(data_dir, out_dir):
    data_dir = Path(data_dir)
    splits_cfg = [
        ('Train', 'steelblue', 8,  0.35),
        ('Val',   'tomato',    30, 0.5),
        ('Test',  'seagreen',  40, 0.6),
    ]

    points = {}
    for split, *_ in splits_cfg:
        machs, aoas = [], []
        split_dir = data_dir / 'processed' / split.lower()
        if split_dir.exists():
            for f in split_dir.glob('aoa*.npz'):
                mm = _PROC_RE.match(f.stem)
                if mm:
                    aoas.append(float(mm.group(1)))
                    machs.append(float(mm.group(2)))
        points[split] = (np.array(machs), np.array(aoas))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=_DPI)

    for split, color, size, alpha in splits_cfg:
        machs, aoas = points[split]
        if len(machs):
            ax.scatter(machs, aoas, s=size, c=color, alpha=alpha,
                       label=f'{split}  (n={len(machs)})', linewidths=0)

    ax.set_xlabel('Mach inlet', fontsize=11)
    ax.set_ylabel('AoA (°)', fontsize=11)
    ax.set_title('Distribution du dataset — (Mach, AoA)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, markerscale=1.8, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / '1_dataset_distribution.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > 1_dataset_distribution.png")


def plot_reference_cases(ref_paths, mesh_hr, mesh_lr, out_dir):

    n = len(ref_paths)

    cases = []
    for path in ref_paths:
        hr, lr, aoa, mach_in = _load_pair(path)
        mach_hr_f = to_mach(hr['prim'])
        mach_lr_f = to_mach(lr['prim'])
        mach_idw  = _interp_lr_to_hr(lr['pos'], hr['pos'], mach_lr_f)
        cases.append(dict(aoa=aoa, mach_in=mach_in,
                          mach_hr=mach_hr_f, mach_lr=mach_lr_f,
                          mach_idw=mach_idw))

    triang_hr = _triang(mesh_hr)
    triang_lr = _triang(mesh_lr)

    fig, axes = plt.subplots(3, n,
                              figsize=(3.2 * n, 9.5),
                              dpi=_DPI, constrained_layout=True)
    if n == 1:
        axes = axes[:, None]

    for col, c in enumerate(cases):
        vmin = min(c['mach_hr'].min(), c['mach_lr'].min(), c['mach_idw'].min())
        vmax = max(c['mach_hr'].max(), c['mach_lr'].max(), c['mach_idw'].max())
        col_title = f"M={c['mach_in']:.2f}  AoA={c['aoa']:+.0f}°"

        axes[0, col].tripcolor(triang_hr, facecolors=c['mach_hr'],
                               cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0, col].set_aspect('equal'); axes[0, col].axis('off')
        axes[0, col].set_title(col_title, fontsize=7, fontweight='bold', pad=2)

        axes[1, col].tripcolor(triang_lr, facecolors=c['mach_lr'],
                               cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, col].set_aspect('equal'); axes[1, col].axis('off')

        sc = axes[2, col].tripcolor(triang_hr, facecolors=c['mach_idw'],
                                    cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2, col].set_aspect('equal'); axes[2, col].axis('off')

        fig.colorbar(sc,
                     ax=axes[:, col].tolist(),
                     orientation='horizontal', location='bottom',
                     label='Mach', shrink=0.9, pad=0.02, aspect=25)

    for row_idx, lbl in enumerate(['HR', 'LR', 'IDW\nLR->HR']):
        axes[row_idx, 0].text(-0.08, 0.5, lbl,
                               transform=axes[row_idx, 0].transAxes,
                               va='center', ha='right', fontsize=8,
                               fontweight='bold', rotation=0)

    fig.suptitle(f'Sanity check — {n} cas de référence',
                 fontsize=13, fontweight='bold')
    plt.savefig(Path(out_dir) / '2_reference_cases.png',
                dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > 2_reference_cases.png  ({n} cas)")


def plot_interp_errors(ref_paths, out_dir):
    labels, l2_rels = [], []
    for path in ref_paths:
        hr, lr, aoa, mach_in = _load_pair(path)
        mach_hr_f = to_mach(hr['prim'])
        mach_idw  = _interp_lr_to_hr(lr['pos'], hr['pos'], to_mach(lr['prim']))
        l2_rel = float(
            np.linalg.norm(mach_hr_f - mach_idw) / (np.linalg.norm(mach_hr_f) + 1e-8)
        )
        labels.append(f"M={mach_in:.2f}\nAoA={aoa:+.0f}°")
        l2_rels.append(l2_rel * 100)

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels)), 4.5), dpi=_DPI)
    bars = ax.bar(labels, l2_rels, color='steelblue', alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    for bar, v in zip(bars, l2_rels):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05 * max(l2_rels),
                f'{v:.2f} %', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    ax.set_ylabel('L2 relatif (%)', fontsize=10)
    ax.set_ylim(0, max(l2_rels) * 1.25)
    ax.set_title("Erreur d'interpolation IDW (LR->HR) — L2 relative Mach",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / '4_interp_l2.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > 4_interp_l2.png")


def plot_interp_error_panels(ref_paths, mesh_hr, out_dir):

    n = len(ref_paths)

    cases = []
    for path in ref_paths:
        hr, lr, aoa, mach_in = _load_pair(path)
        mach_hr_f = to_mach(hr['prim'])
        mach_idw  = _interp_lr_to_hr(lr['pos'], hr['pos'], to_mach(lr['prim']))
        error     = np.abs(mach_hr_f - mach_idw)
        l2_rel    = float(np.linalg.norm(error) / (np.linalg.norm(mach_hr_f) + 1e-8))
        cases.append(dict(aoa=aoa, mach_in=mach_in,
                          mach_hr=mach_hr_f, mach_idw=mach_idw,
                          error=error, l2_rel=l2_rel))

    vmax_e = max(c['error'].max() for c in cases)
    triang  = _triang(mesh_hr)

    fig, axes = plt.subplots(3, n,
                              figsize=(3.2 * n, 9.5),
                              dpi=_DPI, constrained_layout=True)
    if n == 1:
        axes = axes[:, None]

    im_err = None
    for col, c in enumerate(cases):
        vmin_m = min(c['mach_hr'].min(), c['mach_idw'].min())
        vmax_m = max(c['mach_hr'].max(), c['mach_idw'].max())
        col_title = f"M={c['mach_in']:.2f}  AoA={c['aoa']:+.0f}°"

        axes[0, col].tripcolor(triang, facecolors=c['mach_hr'],
                               cmap='RdBu_r', vmin=vmin_m, vmax=vmax_m)
        axes[0, col].set_aspect('equal'); axes[0, col].axis('off')
        axes[0, col].set_title(col_title, fontsize=7, fontweight='bold', pad=2)

        sc_m = axes[1, col].tripcolor(triang, facecolors=c['mach_idw'],
                                      cmap='RdBu_r', vmin=vmin_m, vmax=vmax_m)
        axes[1, col].set_aspect('equal'); axes[1, col].axis('off')

        fig.colorbar(sc_m,
                     ax=[axes[0, col], axes[1, col]],
                     orientation='horizontal', location='bottom',
                     label='Mach', shrink=0.9, pad=0.02, aspect=25)

        sc_e = axes[2, col].tripcolor(triang, facecolors=c['error'],
                                      cmap='hot', vmin=0, vmax=vmax_e)
        axes[2, col].set_aspect('equal'); axes[2, col].axis('off')
        axes[2, col].set_title(f"L2 rel = {c['l2_rel']*100:.2f} %",
                               fontsize=7, pad=2)
        if im_err is None:
            im_err = sc_e

    fig.colorbar(im_err,
                 ax=axes[2, :].tolist(),
                 orientation='horizontal', location='bottom',
                 label='|ΔMach|', shrink=0.9, pad=0.02, aspect=25)

    for row_idx, lbl in enumerate(['HR', 'IDW\nLR->HR', '|ΔMach|']):
        axes[row_idx, 0].text(-0.08, 0.5, lbl,
                               transform=axes[row_idx, 0].transAxes,
                               va='center', ha='right', fontsize=8,
                               fontweight='bold')

    fig.suptitle("Erreur d'interpolation IDW (LR->HR) — champs Mach",
                 fontsize=13, fontweight='bold')
    plt.savefig(Path(out_dir) / '3_interp_error_panels.png',
                dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > 3_interp_error_panels.png")
