from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from mesh_utils import (WALL, MeshSizeParams, _menger_curvature,
                         sample_airfoil_boundary, local_size_kdtree,
                         make_airfoil_refinement, build_outer_boundary,
                         triangulate_with_hole)
from jax_fvm.src.mesh import Mesh

mesh_dir = repo_root / "meshes" / "naca"
mesh_dir.mkdir(parents=True, exist_ok=True)

DEFAULT_SIZE_PARAMS = MeshSizeParams()

def _thickness(xc: np.ndarray, t: float) -> np.ndarray:
    xc = np.maximum(xc, 0.0)
    return 5*t*(0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2
                + 0.2843*xc**3 - 0.1036*xc**4)


def _camber(xc: np.ndarray, m: float, p: float):
    if m == 0.0 or p == 0.0:
        return np.zeros_like(xc), np.zeros_like(xc)
    yc  = np.where(xc < p,
                   m/p**2*(2*p*xc - xc**2),
                   m/(1-p)**2*((1-2*p) + 2*p*xc - xc**2))
    dyc = np.where(xc < p,
                   2*m/p**2*(p - xc),
                   2*m/(1-p)**2*(p - xc))
    return yc, dyc


def naca_contour(m: float, p: float, t: float, n_pts: int = 200,
                 chord: float = 1.0, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Contour NACA 4 chiffres, sens trigonométrique, sans point dupliqué."""
    beta = np.linspace(0.0, np.pi, n_pts)
    xc   = 0.5 * (1.0 - np.cos(beta))
    yt   = _thickness(xc, t)
    yc, dyc = _camber(xc, m, p)
    theta   = np.arctan(dyc)

    upper = np.column_stack([(xc - yt*np.sin(theta))*chord + x0,
                              (yc + yt*np.cos(theta))*chord + y0])
    lower = np.column_stack([(xc + yt*np.sin(theta))*chord + x0,
                              (yc - yt*np.cos(theta))*chord + y0])[::-1]
    return np.concatenate([upper, lower[1:-1]])

def build_mesh(Lx=4.0, Ly=4.0, h=0.05, m=0.0, p=0.0, t=0.12, chord=1.0,
               cx=None, cy=None, export_vtk=False, size_params=DEFAULT_SIZE_PARAMS):

    cx = Lx / 2 if cx is None else cx
    cy = Ly / 2 if cy is None else cy
    x_le = cx - chord / 2.0
    x_te = cx + chord / 2.0

    if x_le < 0 or x_te > Lx:
        raise ValueError(f"Profil (LE={x_le:.2f}, TE={x_te:.2f}) hors du domaine [0, {Lx}].")

    max_thickness = t * chord
    dense         = naca_contour(m, p, t, n_pts=2000, chord=chord, x0=x_le, y0=cy)
    tree          = cKDTree(dense)
    kappa         = _menger_curvature(dense)

    size_func  = lambda pt: local_size_kdtree(pt, tree, cx, chord, max_thickness, h, size_params, kappa)
    outer_pts, outer_ms     = build_outer_boundary(Lx, Ly, size_func, h)
    airfoil_pts, airfoil_ms = sample_airfoil_boundary(dense, 2000, chord, x_le, cy, h)
    refinement              = make_airfoil_refinement(tree, cx, cy, chord, max_thickness, h, size_params, kappa)

    yc_mid  = float(_camber(np.array([0.5]), m, p)[0][0])
    hole_pt = (cx, cy + yc_mid * chord)

    mesh    = triangulate_with_hole(outer_pts, outer_ms, airfoil_pts, airfoil_ms, hole_pt, refinement)
    naca_id = f"naca{int(m*100):01d}{int(p*10):01d}{int(t*100):02d}"
    mesh.set_metadata(
        case=naca_id, h=h, domain={"Lx": Lx, "Ly": Ly},
        obstacle_length=chord, chord=chord,
        thickness=t, camber=m, camber_pos=p,
        height=max_thickness, center={"cx": cx, "cy": cy},
    )
    mesh.print_statistics()

    path = mesh_dir / f"{naca_id}_h{h}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path


if __name__ == "__main__":
    Lx = 4.0; Ly = 4.0
    chord = 1.0; cx = 1.5; cy = Ly / 2
    m, p, t = 0.0, 0.0, 0.12

    for h in [0.1, 0.05, 0.025]:
        mesh, path = build_mesh(Lx=Lx, Ly=Ly, h=h, m=m, p=p, t=t,
                                chord=chord, cx=cx, cy=cy, export_vtk=False)
        mesh.plot_mesh(filename=mesh_dir / f"naca0012_h{h}.png", dpi=500)
