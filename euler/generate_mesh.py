"""Génération centralisée des maillages Euler avec JAX-FVM.
Usage :
    uv run python generate_mesh.py                       # tous les cas, résolutions par défaut
    uv run python generate_mesh.py --geom oa209          # un cas, résolutions par défaut
    uv run python generate_mesh.py --geom naca2412 -h 0.1 0.05
"""
import argparse
import sys
from math import radians, tan
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))              # jax_fvm
sys.path.insert(0, str(_HERE / "meshing"))  # modules de maillage

import airfoils, diamond, bump

_AIRFOIL_PARAMS = dict(Lx=4.0, Ly=4.0, chord=1.0, cx=1.5, cy=2.0)
_DEFAULT_H = [0.3, 0.2, 0.15, 0.1, 0.05, 0.025]
_FIG_DIR = _HERE.parent / "data" / "figures" / "meshes"


def _airfoil_builder(name):
    return lambda h, out_dir: airfoils.build_mesh(name, h=h, out_dir=out_dir, **_AIRFOIL_PARAMS)


GEOMS = {name: (_airfoil_builder(name), _DEFAULT_H) for name in airfoils.AIRFOILS}
GEOMS["diamond"] = (
    lambda h, out_dir: diamond.build_mesh(Lx=4.0, Ly=4.0, h=h, chord=1.0,
                                          height=tan(radians(5)), cx=1.5, cy=2.0, out_dir=out_dir),
    _DEFAULT_H,
)
GEOMS["bump"] = (
    lambda h, out_dir: bump.build_mesh(Lx=3.0, Ly=1.0, h=h, thickness=0.04,
                                       chord=1.0, center=1.5, out_dir=out_dir),
    [0.05, 0.02],
)
# "cylinder" (cercle, cas test classique, hors distribution des formes en
# train) est déjà couvert par la boucle GEOMS ci-dessus via AIRFOILS.

# Variantes OOD du diamond (translation / demi-angle) et du rae2822
# (translation), pour évaluer la robustesse géométrique des modèles SR.
GEOMS["diamond_tx"] = (
    lambda h, out_dir: diamond.build_mesh(Lx=4.0, Ly=4.0, h=h, chord=1.0,
                                          height=tan(radians(5)), cx=2.0, cy=2.0,
                                          case="diamond_tx", out_dir=out_dir),
    _DEFAULT_H,
)
GEOMS["diamond_txy"] = (
    lambda h, out_dir: diamond.build_mesh(Lx=4.0, Ly=4.0, h=h, chord=1.0,
                                          height=tan(radians(5)), cx=2.0, cy=2.5,
                                          case="diamond_txy", out_dir=out_dir),
    _DEFAULT_H,
)
GEOMS["diamond_alpha10"] = (
    lambda h, out_dir: diamond.build_mesh(Lx=4.0, Ly=4.0, h=h, chord=1.0,
                                          height=tan(radians(10)), cx=1.5, cy=2.0,
                                          case="diamond_alpha10", out_dir=out_dir),
    _DEFAULT_H,
)
GEOMS["diamond_sym"] = (
    lambda h, out_dir: diamond.build_mesh(Lx=4.0, Ly=4.0, h=h, chord=1.0,
                                          height=tan(radians(5)), cx=1.5, cy=2.0,
                                          case="diamond_sym", symmetric=True, out_dir=out_dir),
    _DEFAULT_H,
)
# Domaine agrandi (6x4 -> 6x6) a position du diamant inchangee (cx=1.5, cy=2.0) :
# isole l'effet du farfield sans le melanger a une translation.
GEOMS["diamond_big"] = (
    lambda h, out_dir: diamond.build_mesh(Lx=6.0, Ly=6.0, h=h, chord=1.0,
                                          height=tan(radians(5)), cx=1.5, cy=2.0,
                                          case="diamond_big", out_dir=out_dir),
    _DEFAULT_H,
)
GEOMS["rae2822_txy"] = (
    lambda h, out_dir: airfoils.build_mesh("rae2822_txy", h=h, out_dir=out_dir,
                                           Lx=4.0, Ly=4.0, chord=1.0, cx=2.0, cy=2.5),
    _DEFAULT_H,
)


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--geom", nargs="*", choices=list(GEOMS), default=list(GEOMS))
    parser.add_argument("-h", "--mesh-size", dest="h", type=float, nargs="+", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fig-dir", default=str(_FIG_DIR))
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for geom in args.geom:
        builder, default_h = GEOMS[geom]
        for h in (args.h if args.h is not None else default_h):
            mesh, path = builder(h, args.out_dir)
            mesh.plot_mesh(filename=str(fig_dir / f"{path.stem}.png"), dpi=500)


if __name__ == "__main__":
    main()
