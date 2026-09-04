"""
Figures d'intro du rapport (report/figures/) : maillages, champs de référence
et erreurs IDW comparés sur plusieurs géométries au même régime (Mach, AoA).

Usage :
    python report_figures.py --data data/ --geometries diamond naca0012 rae2822 \
        --mach 0.9 --aoa 2.0 --out_dir report/figures/
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import euler.jax_fvm.src.mesh  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utils.viz._style  # noqa: F401
from utils.viz._style import _DPI, CMAP_FIELD, CMAP_ERR
from utils.viz.sanity import _load_pair, _triang, _idw, _snap_mach
from utils.refs import to_mach, _PROC_RE
from utils.metrics import w2
from utils.layout import DataLayout


def _find_closest_case(layout: DataLayout, mach_t: float, aoa_t: float) -> Path:
    """Fichier processé (HR) le plus proche de (mach_t, aoa_t) sur train/val/test."""
    best, best_d = None, None
    for split in ('test', 'val', 'train'):
        d = layout.proc_dir() / split
        if not d.exists():
            continue
        for f in d.glob('aoa*.npz'):
            m = _PROC_RE.match(f.stem)
            if not m:
                continue
            aoa, mach = float(m.group(1)), float(m.group(2))
            dist = (mach - mach_t) ** 2 + ((aoa - aoa_t) / 5.0) ** 2
            if best_d is None or dist < best_d:
                best, best_d = f, dist
    return best


def build_mesh_figure(geometries, meshes, out_path, zoom=None):
    """geometries : noms. meshes : dict geom -> (triang_lr, triang_hr).
    zoom : (xmin, xmax, ymin, ymax) optionnel, domaine complet sinon
    illisible (obstacle minuscule face au domaine de champ lointain)."""
    n = len(geometries)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.2 * n), dpi=_DPI)
    if n == 1:
        axes = axes[np.newaxis, :]
    for ri, geom in enumerate(geometries):
        triang_lr, triang_hr = meshes[geom]
        for ci, (triang, label) in enumerate([(triang_lr, 'LR'), (triang_hr, 'HR')]):
            ax = axes[ri][ci]
            ax.triplot(triang, lw=0.25, color='0.25')
            ax.set_aspect('equal')
            if zoom is not None:
                xmin, xmax, ymin, ymax = zoom
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            ax.axis('off')
            if ri == 0:
                ax.set_title(label, fontsize=12, fontweight='bold')
        axes[ri][0].text(-0.03, 0.5, geom, transform=axes[ri][0].transAxes,
                          va='center', ha='right', fontsize=11, fontweight='bold', rotation=90)
    fig.suptitle('Maillages LR / HR par géométrie', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  > {out_path.name}')


def build_reference_fields_figure(geometries, cases, out_path, zoom=None):
    """N lignes (géométries) x 3 colonnes (LR, IDW, HR), 1 cas par géométrie.
    constrained_layout : seule option fiable pour des panneaux à aspect fixe
    ('equal') : un calcul manuel de marges/figsize laisse presque toujours
    du blanc résiduel (l'aspect ratio réel du panneau rendu ne colle jamais
    exactement aux fractions gridspec supposées)."""
    n = len(geometries)
    if zoom is not None:
        xmin, xmax, ymin, ymax = zoom
        ar = (ymax - ymin) / (xmax - xmin)
    else:
        t = cases[geometries[0]]['triang_hr']
        ar = (t.y.max() - t.y.min()) / max(t.x.max() - t.x.min(), 1e-6)

    panel_w = 2.4
    panel_h = panel_w * ar
    fig, axes = plt.subplots(n, 3, figsize=(3 * panel_w, n * panel_h), dpi=_DPI,
                              constrained_layout=True, squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)

    for ri, geom in enumerate(geometries):
        c = cases[geom]
        triang_lr, triang_hr = c['triang_lr'], c['triang_hr']
        vmin, vmax = float(c['mach_hr'].min()), float(c['mach_hr'].max())
        kw = dict(cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
        ax_lr, ax_idw, ax_hr = axes[ri]
        ax_lr.tripcolor(triang_lr, facecolors=np.clip(c['mach_lr'], vmin, vmax), **kw)
        ax_idw.tripcolor(triang_hr, facecolors=np.clip(c['mach_idw'], vmin, vmax), **kw)
        sc_hr = ax_hr.tripcolor(triang_hr, facecolors=c['mach_hr'], **kw)
        for ax in (ax_lr, ax_idw, ax_hr):
            ax.set_aspect('equal')
            if zoom is not None:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            ax.axis('off')
        if ri == 0:
            ax_lr.set_title('LR', fontsize=11, fontweight='bold', pad=3)
            ax_idw.set_title('IDW', fontsize=11, fontweight='bold', pad=3)
            ax_hr.set_title('HR', fontsize=11, fontweight='bold', pad=3)
        # nom de géométrie seul, à la verticale (le régime M/AoA est commun
        # à toutes les lignes, inutile de le répéter : il part dans le titre)
        ax_lr.text(-0.06, 0.5, geom, transform=ax_lr.transAxes, va='center', ha='center',
                   rotation=90, fontsize=9.5, fontweight='bold')
        cb = fig.colorbar(sc_hr, ax=list(axes[ri]), shrink=0.75, aspect=12, pad=0.015, label='Mach')
        cb.ax.tick_params(labelsize=6)
        cb.set_label('Mach', fontsize=7)

    ref = cases[geometries[0]]
    fig.suptitle(f"Champs Mach local — LR / IDW / HR (M={ref['mach']:.2f}, AoA={ref['aoa']:+.0f}°)",
                 fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    print(f'  > {out_path.name}')


def build_reference_errors_figure(geometries, cases, out_path, zoom=None):
    """1 ligne, N colonnes : erreur relative Mach (IDW vs HR), métrique L2 (non pondérée) + W2."""
    n = len(geometries)
    if zoom is not None:
        xmin0, xmax0, ymin0, ymax0 = zoom
        ar = (ymax0 - ymin0) / (xmax0 - xmin0)
    else:
        t = cases[geometries[0]]['triang_hr']
        ar = (t.y.max() - t.y.min()) / max(t.x.max() - t.x.min(), 1e-6)
    panel_w = 2.4
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n, panel_w * ar + 0.35), dpi=_DPI,
                              constrained_layout=True, squeeze=False)
    # h_pad plus large que w_pad : espace vertical requis entre suptitle et
    # titres de colonne (sinon collision : w_pad=0.01 seul est trop serré
    # verticalement aussi, constrained_layout ne les distingue pas assez).
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.08, wspace=0.01)
    axes = list(axes[0])

    err_vmax = float(np.percentile(
        np.concatenate([cases[g]['err_rel'] for g in geometries]), 95))

    sc_ref = None
    for ci, geom in enumerate(geometries):
        c = cases[geom]
        sc = axes[ci].tripcolor(c['triang_hr'], facecolors=c['err_rel'],
                                 cmap=CMAP_ERR, vmin=0, vmax=err_vmax)
        axes[ci].set_aspect('equal')
        if zoom is not None:
            xmin, xmax, ymin, ymax = zoom
            axes[ci].set_xlim(xmin, xmax)
            axes[ci].set_ylim(ymin, ymax)
        axes[ci].axis('off')
        axes[ci].set_title(geom, fontsize=9.5, fontweight='bold', pad=4)
        axes[ci].text(0.5, 0.02, f"L₂={c['l2'] * 100:.3f}%  ·  W₂={c['w2'] * 100:.3f}%",
                      transform=axes[ci].transAxes, ha='center', va='bottom',
                      fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))
        if sc_ref is None:
            sc_ref = sc

    cb = fig.colorbar(sc_ref, ax=axes, location='right', shrink=0.75, aspect=15, pad=0.015,
                       label='Erreur rel. Mach (IDW)')
    cb.ax.tick_params(labelsize=7)
    ref = cases[geometries[0]]
    fig.suptitle(f"Erreur IDW — Mach local, IDW vs HR (M={ref['mach']:.2f}, AoA={ref['aoa']:+.0f}°)",
                 fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    print(f'  > {out_path.name}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/')
    p.add_argument('--geometries', nargs='+', default=['diamond', 'naca0012', 'rae2822'])
    p.add_argument('--mach', type=float, default=0.9)
    p.add_argument('--aoa', type=float, default=2.0)
    p.add_argument('--lr_res', type=float, default=0.1)
    p.add_argument('--hr_res', type=float, default=0.025)
    p.add_argument('--out_dir', default='report/figures/')
    p.add_argument('--zoom', type=float, nargs=4, default=[1.2, 2.8, 1.5, 2.5],
                   metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                   help="Fenêtre de zoom pour la figure de maillage (domaine complet "
                        "[0,4]x[0,4] sinon illisible). Passe --zoom 0 4 0 4 pour le domaine entier.")
    p.add_argument('--zoom_fields', type=float, nargs=4, default=[0.8, 3.6, 0.4, 3.6],
                   metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                   help="Fenêtre de zoom pour les figures champs/erreurs (plus large que "
                        "--zoom pour garder le cône de choc visible).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data).resolve()

    meshes = {}
    cases = {}
    for geom in args.geometries:
        layout = DataLayout.from_root(data_root, geom, args.lr_res, args.hr_res)
        mesh_hr = np.load(layout.mesh_path(args.hr_res), allow_pickle=True).item()
        mesh_lr = np.load(layout.mesh_path(args.lr_res), allow_pickle=True).item()
        triang_hr, triang_lr = _triang(mesh_hr), _triang(mesh_lr)
        meshes[geom] = (triang_lr, triang_hr)

        path = _find_closest_case(layout, args.mach, args.aoa)
        if path is None:
            print(f"  [WARN] aucun cas trouvé pour {geom} près de M={args.mach}, AoA={args.aoa}")
            continue
        hr, lr, aoa, mach_in = _load_pair(path)
        mach_hr = to_mach(hr['prim'])
        mach_lr = to_mach(lr['prim'])
        mach_idw = _idw(lr['pos'], hr['pos'], mach_lr)
        err_rel = np.abs(mach_idw - mach_hr) / (np.abs(mach_hr) + 1e-6)
        w2_val = w2(mach_idw, mach_hr) / (np.mean(np.abs(mach_hr)) + 1e-8)
        # L2 relative NON pondérée (contrairement à L2w de utils.viz.sanity, qui
        # pondère par le gradient de pression), demandé explicitement pour
        # cette figure de rapport.
        l2_val = float(np.sqrt(np.mean((mach_idw - mach_hr) ** 2) / (np.mean(mach_hr ** 2) + 1e-8)))
        print(f"  [{geom}] cas retenu M={mach_in:.3f} AoA={aoa:+.2f}° "
              f"(cible M={args.mach} AoA={args.aoa})  L2={l2_val*100:.3f}%  W2={w2_val*100:.3f}%")
        cases[geom] = dict(
            triang_hr=triang_hr, triang_lr=triang_lr,
            mach_hr=mach_hr, mach_lr=mach_lr, mach_idw=mach_idw,
            mach=_snap_mach(mach_in), aoa=aoa,
            err_rel=err_rel, w2=w2_val, l2=l2_val,
        )

    # Maillages : cf. report_mesh_figures.py (style marqueurs CL colorés de
    # euler/jax_fvm/src/plot.plot_mesh), plus propre que le triplot brut
    # d'origine ici, ne pas régénérer meshes_lr_hr.png.
    zoom_fields = tuple(args.zoom_fields) if args.zoom_fields != [0, 4, 0, 4] else None
    if len(cases) == len(args.geometries):
        # Deux versions gardées à chaque fois : zoomée (lisible, obstacle/choc au
        # premier plan) et domaine complet (contexte champ lointain) : les deux
        # ont leur usage selon l'endroit du rapport.
        build_reference_fields_figure(args.geometries, cases, out_dir / 'reference_fields_multi_geo.png', zoom=zoom_fields)
        build_reference_fields_figure(args.geometries, cases, out_dir / 'reference_fields_multi_geo_full.png', zoom=None)
        build_reference_errors_figure(args.geometries, cases, out_dir / 'reference_errors_multi_geo.png', zoom=zoom_fields)
        build_reference_errors_figure(args.geometries, cases, out_dir / 'reference_errors_multi_geo_full.png', zoom=None)
    else:
        print("  [WARN] cas manquants pour au moins une géométrie, figures fields/errors sautées.")

    print(f"\nDone — figures dans {out_dir}/")


if __name__ == '__main__':
    main()
