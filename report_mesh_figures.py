"""
Figures de maillage pour le rapport (report/figures/) :
  1. diamond seul, domaine complet, LR vs HR (le cas le plus étudié).
  2. toutes les géométries, format ligne compact (HR en haut, LR en bas),
     zoomé sur l'obstacle.

Reprend le style "marqueurs de CL colorés" de euler/jax_fvm/src/plot.plot_mesh
(paroi distinguée du reste du bord) plutôt qu'un triplot brut.

Usage :
    python report_mesh_figures.py --data data/ --out_dir report/figures/
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
from matplotlib import tri as mtri
from matplotlib.pyplot import cm

from utils.layout import DataLayout

_DPI = 200
_BC_COLORS = cm.rainbow(np.linspace(0, 1, 10))


def _load_mesh(data_root: Path, geometry: str, res: float):
    layout = DataLayout.from_root(data_root, geometry, 0.1, 0.025)
    return np.load(layout.mesh_path(res), allow_pickle=True).item()


def _draw_mesh(ax, mesh, zoom=None, lw=0.25, wall_lw=1.1):
    """Triplot + bord coloré par marqueur de CL (paroi distinguée du champ
    lointain), même logique que euler/jax_fvm/src/plot.plot_mesh."""
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.tris)
    ax.triplot(triang, color='0.35', lw=lw)
    markers = np.unique(np.asarray(mesh.face_markers))
    for bc_marker in markers[markers > 0]:
        ids = np.where(np.asarray(mesh.face_markers) == bc_marker)[0]
        pts = np.asarray(mesh.points)[np.asarray(mesh.faces)[ids]]
        color = _BC_COLORS[int(bc_marker) % len(_BC_COLORS)]
        for seg in pts:
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=wall_lw, zorder=5)
    ax.set_aspect('equal')
    if zoom is not None:
        xmin, xmax, ymin, ymax = zoom
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.axis('off')


def build_diamond_full_domain(data_root, out_path):
    mesh_lr = _load_mesh(data_root, 'diamond', 0.1)
    mesh_hr = _load_mesh(data_root, 'diamond', 0.025)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), dpi=_DPI)
    _draw_mesh(axes[0], mesh_lr, zoom=None)
    axes[0].set_title(r'LR ($h=0.1$)', fontsize=13, fontweight='bold')
    _draw_mesh(axes[1], mesh_hr, zoom=None, lw=0.12)
    axes[1].set_title(r'HR ($h=0.025$)', fontsize=13, fontweight='bold')
    fig.suptitle('Maillage diamant — domaine complet', fontsize=14, fontweight='bold')
    # rect : sans reserve en haut, tight_layout ignore le suptitle et les titres
    # de colonne viennent se superposer a lui.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  > {out_path.name}')


def build_all_geometries_compact(data_root, geometries, out_path, zoom):
    n = len(geometries)
    # Hauteur deduite de l'aspect reel de la fenetre de zoom : un figsize fixe
    # laissait une bande blanche enorme entre les deux rangees (obstacle tres
    # aplati, aspect 'equal' -> les axes ne remplissent pas leur boite).
    xmin, xmax, ymin, ymax = zoom
    panel_w = 2.2
    panel_h = panel_w * (ymax - ymin) / (xmax - xmin)
    fig, axes = plt.subplots(2, n, figsize=(panel_w * n, 2 * panel_h + 0.55), dpi=_DPI)
    for ci, geom in enumerate(geometries):
        mesh_hr = _load_mesh(data_root, geom, 0.025)
        mesh_lr = _load_mesh(data_root, geom, 0.1)
        _draw_mesh(axes[0][ci], mesh_hr, zoom=zoom, lw=0.15, wall_lw=1.0)
        _draw_mesh(axes[1][ci], mesh_lr, zoom=zoom, lw=0.35, wall_lw=1.0)
        axes[0][ci].set_title(geom, fontsize=10, fontweight='bold')
    axes[0][0].text(-0.15, 0.5, 'HR', transform=axes[0][0].transAxes,
                     va='center', ha='center', fontsize=12, fontweight='bold', rotation=90)
    axes[1][0].text(-0.15, 0.5, 'LR', transform=axes[1][0].transAxes,
                     va='center', ha='center', fontsize=12, fontweight='bold', rotation=90)
    fig.suptitle('Maillages par géométrie (zoom obstacle)', fontsize=13, fontweight='bold')
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.03, right=0.99, top=0.82, bottom=0.02)
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  > {out_path.name}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/')
    p.add_argument('--geometries', nargs='+',
                    default=['diamond', 'naca0012', 'naca2412', 'rae2822', 'oneraD', 'oa209', 'naca23012'])
    # Obstacle réellement centré sur x=1.5 (bbox x in [1.0, 2.0]), pas x=2.0
    # (centre du domaine), vérifié sur les 7 géométries via face_markers.
    p.add_argument('--zoom', type=float, nargs=4, default=[0.6, 2.6, 1.6, 2.4],
                    metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    p.add_argument('--out_dir', default='report/figures/')
    args = p.parse_args()

    data_root = Path(args.data).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_diamond_full_domain(data_root, out_dir / 'mesh_diamond_full_domain.png')
    build_all_geometries_compact(data_root, args.geometries, out_dir / 'mesh_all_geometries_compact.png',
                                  zoom=tuple(args.zoom))
    print(f"\nDone — figures dans {out_dir}/")


if __name__ == '__main__':
    main()
