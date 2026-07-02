"""Génération centralisée des maillages Euler avec JAX-FVM.
Usage :
    uv run python generate_mesh.py --geom naca0012 -h 0.1
    uv run python generate_mesh.py --geom bump -h 0.1 0.05 0.025
"""
import argparse
import sys
from math import radians, tan
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))              # jax_fvm
sys.path.insert(0, str(_HERE / "meshing"))  # modules de maillage

import bump, diamond, naca0012, rae2822, oneraD

# paramètres géométriques
GEOMS = {
    "naca0012": (naca0012, dict(Lx=4.0, Ly=4.0, chord=1.0, cx=1.5, cy=2.0, m=0.0, p=0.0, t=0.12)),
    "rae2822":  (rae2822, dict(Lx=4.0, Ly=4.0, chord=1.0, cx=1.5, cy=2.0)),
    "oneraD":   (oneraD, dict(Lx=4.0, Ly=4.0, chord=1.0, cx=1.5, cy=2.0)),
    "bump":     (bump, dict(Lx=3.0, Ly=1.0, thickness=0.04, chord=1.0, center=1.5)),
    "diamond":  (diamond, dict(Lx=4.0, Ly=4.0, chord=1.0, cx=1.5, cy=2.0, height=tan(radians(5)))),
}


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--geom", required=True, choices=list(GEOMS))
    parser.add_argument("-h", "--mesh-size", dest="h", type=float, nargs="+", required=True)
    args = parser.parse_args()

    module, params = GEOMS[args.geom]
    for h in args.h:
        mesh, path = module.build_mesh(h=h, export_vtk=False, **params)
        mesh.plot_mesh(filename=path.with_suffix(".png"), dpi=500)


if __name__ == "__main__":
    main()
