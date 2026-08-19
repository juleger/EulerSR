from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

import utils.viz._style  # noqa: F401
from utils.viz._style import _DPI, CMAP_FIELD, CMAP_ERR, CMAP_GRAD
from utils.refs import to_mach, REFERENCE_CASES, _PROC_RE
from utils.metrics import w2
from utils.layout import DataLayout
from utils.layout import load_sample

_SPLIT_COLORS = {'train': 'steelblue', 'val': 'tomato', 'test': 'seagreen'}


def _snap_mach(m: float, tol: float = 0.05) -> float:
    for m_ref, _, _ in REFERENCE_CASES:
        if abs(m - m_ref) < tol:
            return m_ref
    return m


def _load_pair(path):
    d = load_sample(path)
    hr_gp = (d['hr_grad_p'] if 'hr_grad_p' in d
             else d['hr_primitives_grad'][:, 3, :] if 'hr_primitives_grad' in d else None)
    hr = {'pos': d['hr_node_pos'], 'prim': d['hr_primitives'], 'grad_p': hr_gp}
    lr = {'pos': d['lr_node_pos'], 'prim': d['lr_primitives'],
           'grad': d['lr_primitives_grad'] if 'lr_primitives_grad' in d else None}
    m = _PROC_RE.match(Path(path).stem)
    return hr, lr, float(m.group(1)), float(m.group(2))  # aoa, mach_in


def _idw(src_pos, tgt_pos, src_field, k=6):
    dists, idx = cKDTree(src_pos).query(tgt_pos, k=k)
    w = 1.0 / (dists ** 2 + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    if src_field.ndim == 1:
        return (w * src_field[idx]).sum(axis=1)
    return (w[:, :, None] * src_field[idx]).sum(axis=1)


def _triang(mesh):
    pts = np.asarray(mesh.points)
    return mtri.Triangulation(pts[:, 0], pts[:, 1], np.asarray(mesh.tris))


def _kde(ax, vals, color, label, lw=2.0, ls='-'):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    v = np.clip(vals, lo, hi)
    try:
        kde = gaussian_kde(v, bw_method='scott')
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, kde(xs), ls, lw=lw, color=color, label=label)
        ax.fill_between(xs, kde(xs), alpha=0.12, color=color)
    except Exception:
        ax.hist(v, bins=30, density=True, histtype='step',
                lw=lw, color=color, linestyle=ls, label=label)


def _scan_dataset(layout: DataLayout, splits=('train', 'val', 'test'), max_per_split=0):
    for split in splits:
        sd = layout.proc_dir() / split
        if not sd.exists():
            continue
        files = sorted(sd.glob('aoa*.npz'))
        if max_per_split > 0:
            files = files[:max_per_split]
        for f in files:
            m = _PROC_RE.match(f.stem)
            if m:
                yield split, float(m.group(1)), float(m.group(2)), f


def _wall_profile(vals_at_cells, wc, n_pts=300):
    """
    Profil scalaire lisse sur faces supérieure/inférieure de la paroi.
    Split par rapport au centre-y des cellules paroi (robuste quelle que
    soit la position de l'obstacle dans le domaine)"""
    bw = wc.bary[wc.unique_wall_cell_ids]
    v = np.asarray(vals_at_cells)[wc.unique_wall_cell_ids]
    x, y = bw[:, 0], bw[:, 1]
    y_mid = 0.5 * (float(y.min()) + float(y.max()))

    def _surface(mask):
        xi, vi = x[mask], v[mask]
        if len(xi) < 3:
            return xi, vi
        ord_ = np.argsort(xi)
        xi, vi = xi[ord_], vi[ord_]
        _, uniq = np.unique(np.round(xi * 1e5).astype(int), return_index=True)
        xi, vi = xi[uniq], vi[uniq]
        if len(xi) < 2:
            return xi, vi
        f = interp1d(xi, vi, kind='linear', bounds_error=False,
                       fill_value=(vi[0], vi[-1]))
        x_r = np.linspace(xi.min(), xi.max(), n_pts // 2)
        return x_r, f(x_r)

    return _surface(y >= y_mid), _surface(y < y_mid)


def _wall_mach_profiles(prim, wc, n_pts=300):
    """Mach local en paroi — faces supérieure et inférieure."""
    return _wall_profile(to_mach(prim), wc, n_pts)

def plot_dataset_distribution(layout: DataLayout, out_dir):
    splits_cfg = [('Train', 'steelblue', 8, 0.4),
                  ('Val', 'tomato', 15, 0.8),
                  ('Test', 'seagreen', 20, 0.9)]
    points = {}
    for split, *_ in splits_cfg:
        machs, aoas = [], []
        sd = layout.proc_dir() / split.lower()
        if sd.exists():
            for f in sd.glob('aoa*.npz'):
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
    for mach_t, aoa_t, label in REFERENCE_CASES:
        ax.scatter(mach_t, aoa_t, s=120, marker='*', c='gold',
                   edgecolors='k', linewidths=0.8, zorder=5)
        ax.annotate(label, (mach_t, aoa_t), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Mach inlet'); ax.set_ylabel('AoA (°)')
    ax.set_title('Distribution dataset — (Mach, AoA)', fontweight='bold')
    ax.legend(fontsize=10, markerscale=1.8, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / '1_dataset_distribution.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print('  > 1_dataset_distribution.png')

def plot_reference_fields(ref_paths, mesh_hr, mesh_lr, out_dir):
    """
    N lignes × 3 colonnes : LR · IDW · HR
    Colorbar champ partagée par ligne (plage HR).
    """
    n = len(ref_paths)
    triang_hr = _triang(mesh_hr)
    triang_lr = _triang(mesh_lr)

    cases = []
    for path in ref_paths:
        hr, lr, aoa, mach_in = _load_pair(path)
        mach_hr = to_mach(hr['prim'])
        mach_lr = to_mach(lr['prim'])
        mach_idw = _idw(lr['pos'], hr['pos'], mach_lr)
        cases.append(dict(aoa=aoa, mach_in=mach_in, mach_hr=mach_hr,
                          mach_lr=mach_lr, mach_idw=mach_idw))

    ar = max(float(triang_hr.y.max() - triang_hr.y.min()) /
             max(float(triang_hr.x.max() - triang_hr.x.min()), 1e-6), 0.1)
    panel_w, cb_w = 2.8, 0.22
    panel_h = panel_w * ar
    fig = plt.figure(figsize=(3 * panel_w + cb_w, panel_h * n), dpi=_DPI)
    gs = gridspec.GridSpec(n, 4, figure=fig,
                            width_ratios=[panel_w, panel_w, panel_w, cb_w],
                            hspace=0.01, wspace=0.01,
                            left=0.04, right=0.99, top=0.95, bottom=0.02)
    axes = [[fig.add_subplot(gs[ri, ci]) for ci in range(3)] for ri in range(n)]
    cb_ax = [fig.add_subplot(gs[ri, 3]) for ri in range(n)]

    for ci, title in enumerate(['LR', 'IDW', 'HR']):
        axes[0][ci].set_title(title, fontsize=10, fontweight='bold', pad=3)

    for ri, c in enumerate(cases):
        vmin, vmax = float(c['mach_hr'].min()), float(c['mach_hr'].max())
        kw = dict(cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
        axes[ri][0].tripcolor(triang_lr, facecolors=np.clip(c['mach_lr'], vmin, vmax), **kw)
        axes[ri][1].tripcolor(triang_hr, facecolors=np.clip(c['mach_idw'], vmin, vmax), **kw)
        axes[ri][2].tripcolor(triang_hr, facecolors=c['mach_hr'], **kw)
        for ax in axes[ri]:
            ax.set_aspect('equal'); ax.axis('off')
        axes[ri][0].text(-0.01, 0.5,
                         f"M={_snap_mach(c['mach_in']):.2f}  AoA={c['aoa']:+.0f}°",
                         transform=axes[ri][0].transAxes,
                         va='center', ha='center', fontsize=8, fontweight='bold',
                         rotation=90)
        sm = plt.cm.ScalarMappable(cmap=CMAP_FIELD,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        _p = cb_ax[ri].get_position()
        _h = _p.height * 0.80
        cb_ax[ri].set_position([_p.x0, _p.y0 + (_p.height - _h) / 2, _p.width, _h])
        fig.colorbar(sm, cax=cb_ax[ri], label='Mach')
        cb_ax[ri].tick_params(labelsize=6)

    fig.suptitle('Champs test - Mach local - LR/IDW/HR',
                 fontsize=11, fontweight='bold', y=0.99)
    plt.savefig(Path(out_dir) / '2a_reference_fields.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  > 2a_reference_fields.png  ({n} cas)')

def plot_primitive_fields(ref_paths, mesh_hr, out_dir):
    """
    N lignes (cas) × 4 colonnes (rho, u, v, p) — champs HR uniquement.
    Colorbar individuelle par sous-figure, plage globale par variable.
    Colormaps adaptées : viridis (rho), plasma (u), RdBu_r (v), inferno (p).
    """
    import matplotlib.colors as mcolors
    from utils.viz._style import VAR_NAMES, CMAP_PRIMS

    _DIVERGING = {'RdBu', 'RdBu_r', 'bwr', 'seismic', 'coolwarm'}

    n = len(ref_paths)
    triang_hr = _triang(mesh_hr)

    cases = []
    for path in ref_paths:
        hr, lr, aoa, mach_in = _load_pair(path)
        cases.append({'aoa': aoa, 'mach_in': mach_in, 'prim': hr['prim'].astype(np.float32)})

    def _norm(fi, prim):
        lo, hi = float(prim[:, fi].min()), float(prim[:, fi].max())
        if CMAP_PRIMS[fi] in _DIVERGING and lo < 0 < hi:
            return mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
        return mcolors.Normalize(vmin=lo, vmax=hi)

    ar = max(float(triang_hr.y.max() - triang_hr.y.min()) /
                   max(float(triang_hr.x.max() - triang_hr.x.min()), 1e-6), 0.1)
    panel_w = 2.8
    panel_h = panel_w * ar

    fig, axes = plt.subplots(n, 4,
                             figsize=(4 * panel_w + 0.3, n * panel_h + 0.4),
                             dpi=_DPI, constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    for ri, c in enumerate(cases):
        for fi in range(4):
            ax = axes[ri, fi]
            m = ax.tripcolor(triang_hr, facecolors=c['prim'][:, fi],
                              cmap=CMAP_PRIMS[fi], norm=_norm(fi, c['prim']))
            ax.set_aspect('equal'); ax.axis('off')
            if ri == 0:
                ax.set_title(VAR_NAMES[fi], fontsize=11, fontweight='bold', pad=3)
            cb = fig.colorbar(m, ax=ax, shrink=0.75, pad=0.02, aspect=18)
            cb.ax.tick_params(labelsize=5)

        axes[ri, 0].text(-0.06, 0.5,
                         f"M={_snap_mach(c['mach_in']):.2f}  AoA={c['aoa']:+.0f}°",
                         transform=axes[ri, 0].transAxes,
                         va='center', ha='right', fontsize=8, fontweight='bold',
                         rotation=90)

    fig.suptitle('Champs primitifs HR — cas de référence (ρ, u, v, p)',
                 fontsize=11, fontweight='bold')
    plt.savefig(Path(out_dir) / '2c_primitive_fields.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  > 2c_primitive_fields.png  ({n} cas)')

def plot_reference_errors(ref_paths, mesh_hr, mesh_lr, out_dir):
    n = len(ref_paths)
    triang_hr = _triang(mesh_hr)

    cases = []
    for path in ref_paths:
        hr, lr, aoa, mach_in = _load_pair(path)
        mach_hr = to_mach(hr['prim'])
        mach_idw = _idw(lr['pos'], hr['pos'], to_mach(lr['prim']))
        err_rel = np.abs(mach_idw - mach_hr) / (np.abs(mach_hr) + 1e-6)
        w2_val = w2(mach_idw, mach_hr) / (np.mean(np.abs(mach_hr)) + 1e-8)
        if hr['grad_p'] is not None:
            w = np.linalg.norm(hr['grad_p'].astype(np.float32), axis=-1)
            # Annule le poids des cellules de frontière (artefact LSQ aux BC)
            pos = hr['pos']; px, py = pos[:, 0], pos[:, 1]; m = 0.1
            bnd = ((px < px.min()+m)|(px > px.max()-m)|(py < py.min()+m)|(py > py.max()-m))
            w[bnd] = 0.0
            w = w + 1e-6; w /= w.sum()
            num = np.sum(w * (mach_idw - mach_hr) ** 2)
            den = np.sum(w * mach_hr ** 2) + 1e-8
            l2w = float(np.sqrt(num / den))
        else:
            l2w = None
        cases.append(dict(aoa=aoa, mach_in=mach_in, err_rel=err_rel, l2w=l2w, w2=w2_val))

    err_vmax = float(np.percentile(
        np.concatenate([c['err_rel'] for c in cases]), 95))

    ar = max(float(triang_hr.y.max() - triang_hr.y.min()) /
             max(float(triang_hr.x.max() - triang_hr.x.min()), 1e-6), 0.1)
    panel_w = 3.2
    panel_h = panel_w * ar
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n, panel_h), dpi=_DPI)
    fig.subplots_adjust(left=0.01, right=0.97, top=0.88, bottom=0.01, wspace=0.01)
    if n == 1:
        axes = [axes]

    sc_ref = None
    for ci, c in enumerate(cases):
        sc = axes[ci].tripcolor(triang_hr, facecolors=c['err_rel'],
                                cmap=CMAP_ERR, vmin=0, vmax=err_vmax)
        axes[ci].set_aspect('equal'); axes[ci].axis('off')
        axes[ci].text(0.5, 1.01, f"M={_snap_mach(c['mach_in']):.2f}  AoA={c['aoa']:+.0f}°",
                      transform=axes[ci].transAxes, ha='center', va='bottom',
                      fontsize=9, fontweight='bold')
        parts = []
        if c['l2w'] is not None:
            parts.append(f"L₂w={c['l2w']*100:.3f}%")
        parts.append(f"W₂={c['w2']*100:.3f}%")
        axes[ci].text(0.5, 0.03, '  ·  '.join(parts),
                      transform=axes[ci].transAxes,
                      ha='center', va='bottom', fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        if sc_ref is None:
            sc_ref = sc

    fig.colorbar(sc_ref, ax=axes, location='right', shrink=0.80, label='Erreur rel. Mach (IDW)')
    fig.suptitle('Erreur relative Mach — IDW/HR',
                 fontsize=11, fontweight='bold')
    plt.savefig(Path(out_dir) / '2b_reference_errors.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  > 2b_reference_errors.png  ({n} cas)')


def plot_pressure_gradient(ref_paths, mesh_hr, out_dir):
    n = len(ref_paths)
    triang_hr = _triang(mesh_hr)

    cases = []
    for path in ref_paths:
        hr, lr, aoa, mach_in = _load_pair(path)
        if hr['grad_p'] is not None:
            gp = np.linalg.norm(hr['grad_p'].astype(np.float32), axis=-1)
        else:
            gp_lr = (np.linalg.norm(lr['grad'].astype(np.float32)[:, 3, :], axis=-1)
                     if lr['grad'] is not None else np.zeros(len(lr['pos'])))
            gp = _idw(lr['pos'], hr['pos'], gp_lr)
        # Masque les cellules de frontière du domaine (artefact LSQ aux BC)
        pos = hr['pos']
        px, py = pos[:, 0], pos[:, 1]
        margin = 0.1
        bnd = ((px < px.min() + margin) | (px > px.max() - margin) |
               (py < py.min() + margin) | (py > py.max() - margin))
        gp_vis = gp.copy()
        gp_vis[bnd] = 0.0
        cases.append(dict(aoa=aoa, mach_in=mach_in, gp=gp_vis))

    vmax = float(np.percentile(np.concatenate([c['gp'] for c in cases]), 99))

    ar = max(float(triang_hr.y.max() - triang_hr.y.min()) /
             max(float(triang_hr.x.max() - triang_hr.x.min()), 1e-6), 0.1)
    panel_w = 3.2
    panel_h = panel_w * ar
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n, panel_h), dpi=_DPI)
    fig.subplots_adjust(left=0.01, right=0.97, top=0.91, bottom=0.01, wspace=0.01)
    if n == 1:
        axes = [axes]
    fig.suptitle('Gradient de pression HR  |∇p|',
                 fontsize=11, fontweight='bold', y=0.99)

    sc_ref = None
    for ci, c in enumerate(cases):
        sc = axes[ci].tripcolor(triang_hr, facecolors=c['gp'],
                                cmap=CMAP_GRAD, vmin=0, vmax=vmax)
        axes[ci].set_aspect('equal'); axes[ci].axis('off')
        axes[ci].text(0.5, 1.01, f"M={_snap_mach(c['mach_in']):.2f}  AoA={c['aoa']:+.0f}°",
                      transform=axes[ci].transAxes, ha='center', va='bottom',
                      fontsize=9, fontweight='bold')
        if sc_ref is None:
            sc_ref = sc

    fig.colorbar(sc_ref, ax=axes, location='right', shrink=0.80, label='|∇p|')
    plt.savefig(Path(out_dir) / '3_pressure_gradient.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print('  > 3_pressure_gradient.png')

def plot_cp_profiles(ref_paths, mesh_hr, mesh_lr, out_dir):
    from utils.aero import build_wall_cache

    wc_hr = build_wall_cache(mesh_hr)
    wc_lr = build_wall_cache(mesh_lr)

    n = len(ref_paths)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 2.8 * n),
                             dpi=_DPI, constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02, hspace=0.04, wspace=0.02)
    if n == 1:
        axes = [axes]

    for ri, path in enumerate(ref_paths):
        hr, lr, aoa, mach_in = _load_pair(path)
        prim_idw = _idw(lr['pos'], hr['pos'], lr['prim'])

        xu_hr, mm_hr = _wall_mach_profiles(hr['prim'], wc_hr)[0]
        xu_idw, mm_idw = _wall_mach_profiles(prim_idw, wc_hr)[0]
        xu_lr, mm_lr = _wall_mach_profiles(lr['prim'], wc_lr)[0]

        ax = axes[ri]
        ax.plot(xu_hr, mm_hr, color='#111111', lw=2.2, label='HR (GT)')
        ax.plot(xu_idw, mm_idw, color='#d63030', lw=1.8, label='IDW')
        ax.plot(xu_lr, mm_lr, color='#1a5fb4', lw=1.5, label='LR')

        # Ligne M=1 seulement si dans la plage des données
        ylim = ax.get_ylim()
        if ylim[0] <= 1.0 <= ylim[1]:
            ax.axhline(1.0, color='gray', lw=0.8, ls=':', alpha=0.6)
            ax.set_ylim(ylim)

        ax.set_ylabel('Mach'); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.set_title(f"M={_snap_mach(mach_in):.2f}  AoA={aoa:+.0f}°",
                     fontsize=10, fontweight='bold')

    axes[-1].set_xlabel('x')
    fig.suptitle('Mach local paroi sup. (LR/IDW/HR)',
                 fontsize=13, fontweight='bold')
    plt.savefig(Path(out_dir) / '4_cp_profiles.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print('  > 4_cp_profiles.png')

def plot_param_distributions(layout: DataLayout, out_dir):
    """KDE 1D de la distribution de Mach inlet (gauche) et AoA (droite) par split."""
    records = {s: {'mach': [], 'aoa': []} for s in ('train', 'val', 'test')}
    for split, aoa, mach_in, _ in _scan_dataset(layout):
        records[split]['mach'].append(mach_in)
        records[split]['aoa'].append(aoa)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=_DPI, constrained_layout=True)
    fig.suptitle('Distributions Mach et AoA par split', fontsize=13, fontweight='bold')

    for ax, key, xlabel, title in zip(
        axes,
        ['mach', 'aoa'],
        ['Mach inlet', 'AoA (°)'],
        ['Mach inlet', "Angle d'attaque AoA"],
    ):
        for split, color in _SPLIT_COLORS.items():
            vals = np.array(records[split][key], dtype=float)
            if len(vals) > 3:
                _kde(ax, vals, color, f'{split}  (n={len(vals)})', lw=2.0)
        ax.set_xlabel(xlabel); ax.set_ylabel('Densité')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    plt.savefig(Path(out_dir) / '5_param_distributions.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print('  > 5_param_distributions.png')

def plot_aero_distributions(layout: DataLayout, mesh_hr, mesh_lr, out_dir):
    """KDE de CL et CD : HR (trait plein) vs LR (tirets pastel même couleur)."""
    from utils.aero import build_wall_cache, aero_coeffs

    wc_hr = build_wall_cache(mesh_hr)
    wc_lr = build_wall_cache(mesh_lr)

    CL_hr, CD_hr = [], []
    CL_lr, CD_lr = [], []
    for split, aoa, mach_in, path in _scan_dataset(layout):
        d = load_sample(path)
        if 'hr_primitives' in d:
            ac = aero_coeffs(d['hr_primitives'].astype(np.float32), wc_hr, mach_in)
            CL_hr.append(ac['CL']); CD_hr.append(ac['CD'])
        ac_lr = aero_coeffs(d['lr_primitives'].astype(np.float32), wc_lr, mach_in)
        CL_lr.append(ac_lr['CL']); CD_lr.append(ac_lr['CD'])

    # Couleurs : plein = HR, pastel tirets = LR
    _CL_HR = '#1a5fb4'
    _CL_LR = '#85b4e0'
    _CD_HR = '#c01c28'
    _CD_LR = '#e08080'

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=_DPI, constrained_layout=True)
    fig.suptitle('Distributions CL et CD (HR/LR)',
                 fontsize=13, fontweight='bold')

    for ax, (hr_vals, lr_vals, c_hr, c_lr, xlabel, title) in zip(
        axes,
        [
            (CL_hr, CL_lr, _CL_HR, _CL_LR, r'$C_L$', 'Portance $C_L$'),
            (CD_hr, CD_lr, _CD_HR, _CD_LR, r'$C_D$', 'Traînée $C_D$'),
        ]
    ):
        if hr_vals:
            _kde(ax, np.array(hr_vals, dtype=float), c_hr,
                 f'HR  (n={len(hr_vals)})', lw=2.2, ls='-')
        if lr_vals:
            _kde(ax, np.array(lr_vals, dtype=float), c_lr,
                 f'LR  (n={len(lr_vals)})', lw=1.8, ls='--')
        ax.set_xlabel(xlabel); ax.set_ylabel('Densité')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    plt.savefig(Path(out_dir) / '6_aero_distributions.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print('  > 6_aero_distributions.png')

def plot_dataset_balance(layout: DataLayout, mesh_lr, out_dir, max_per_split=0):
    """KDE de trois scalaires physiques (Mach_max, |grad p|_max, CD) par split."""
    from utils.aero import build_wall_cache, aero_coeffs

    wc_lr = build_wall_cache(mesh_lr)

    records = {s: {'mach_max': [], 'gp_max': [], 'CD': []}
               for s in ('train', 'val', 'test')}

    for split, aoa, mach_in, path in _scan_dataset(layout, max_per_split=max_per_split):
        d = load_sample(path)
        lr_prim = d['lr_primitives'].astype(np.float32)
        records[split]['mach_max'].append(float(to_mach(lr_prim).max()))
        if 'lr_primitives_grad' in d:
            gp = np.linalg.norm(d['lr_primitives_grad'].astype(np.float32)[:, 3, :], axis=-1)
            records[split]['gp_max'].append(float(gp.max()))
        try:
            records[split]['CD'].append(aero_coeffs(lr_prim, wc_lr, mach_in)['CD'])
        except Exception:
            pass

    panels = [
        ('mach_max', r'$M_{max}$ (LR)', 'Mach maximum local'),
        ('gp_max', r'$|\nabla p|_{max}$', 'Gradient de pression max'),
        ('CD', r'$C_D$ (LR)', 'Coefficient de traînée $C_D$'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=_DPI, constrained_layout=True)
    fig.suptitle('Équilibre des splits — scalaires physiques LR',
                 fontsize=13, fontweight='bold')

    for ax, (key, xlabel, title) in zip(axes, panels):
        for split, color in _SPLIT_COLORS.items():
            vals = np.array(records[split][key], dtype=float)
            if len(vals) > 5:
                _kde(ax, vals, color, f'{split}  (n={len(vals)})', lw=1.8)
        ax.set_xlabel(xlabel); ax.set_ylabel('Densité')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    plt.savefig(Path(out_dir) / '7_dataset_balance.png', dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print('  > 7_dataset_balance.png')
