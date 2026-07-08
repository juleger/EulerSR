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


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--geom", nargs="*", choices=list(GEOMS), default=list(GEOMS))
    parser.add_argument("-h", "--mesh-size", dest="h", type=float, nargs="+", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    for geom in args.geom:
        builder, default_h = GEOMS[geom]
        for h in (args.h if args.h is not None else default_h):
            mesh, path = builder(h, args.out_dir)
            mesh.plot_mesh(filename=str(path.with_suffix(".png")), dpi=500)


if __name__ == "__main__":
    main()
