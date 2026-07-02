from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # euler/ (jax_fvm)
sys.path.insert(0, str(Path(__file__).parent))                # meshing/ (mesh_utils)

from mesh_utils import (WALL, MeshSizeParams, _menger_curvature, sample_airfoil_boundary, local_size_kdtree,
                         make_airfoil_refinement, build_outer_boundary, triangulate_with_hole)
from jax_fvm.src.mesh import Mesh

mesh_dir = repo_root / "meshes" / "oneraD"
mesh_dir.mkdir(parents=True, exist_ok=True)

DEFAULT_SIZE_PARAMS = MeshSizeParams()


def _strictly_increasing_x(arr: np.ndarray) -> np.ndarray:
    keep = [0]
    for i in range(1, len(arr)):
        if arr[i, 0] > arr[keep[-1], 0] + 1e-12:
            keep.append(i)
    return arr[keep]


def _load_dat(path: Path):
    """Parse un fichier .dat ONERA D au format Selig"""
    pts = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < 10:
        raise ValueError(f"Trop peu de points ({len(pts)}) dans {path}")

    i_le = int(np.argmin(pts[:, 0]))          # bord d'attaque
    upper = pts[:i_le + 1][::-1]              # BA -> BF (extrados)
    lower = pts[i_le:]                        # BA -> BF (intrados)
    upper = _strictly_increasing_x(upper)
    lower = _strictly_increasing_x(lower)
    return upper, lower


_DAT_PATH = Path(__file__).parent / "oneraD.dat"
_ONERAD_UPPER, _ONERAD_LOWER = _load_dat(_DAT_PATH)


def oneraD_contour(n_pts: int = 300, chord: float = 1.0, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    cs_upper = PchipInterpolator(_ONERAD_UPPER[:, 0], _ONERAD_UPPER[:, 1])
    cs_lower = PchipInterpolator(_ONERAD_LOWER[:, 0], _ONERAD_LOWER[:, 1])

    beta  = np.linspace(0.0, np.pi, n_pts)
    xc    = 0.5 * (1.0 - np.cos(beta))
    upper = np.column_stack([xc * chord + x0, cs_upper(xc) * chord + y0])
    lower = np.column_stack([xc * chord + x0, cs_lower(xc) * chord + y0])[::-1]
    return np.concatenate([upper, lower[1:-1]])


def build_mesh(Lx=4.0, Ly=4.0, h=0.05, chord=1.0, cx=None, cy=None,
               export_vtk=False, size_params=DEFAULT_SIZE_PARAMS):

    cx = Lx / 2 if cx is None else cx
    cy = Ly / 2 if cy is None else cy
    x_le = cx - chord / 2.0
    x_te = cx + chord / 2.0

    if x_le < 0 or x_te > Lx:
        raise ValueError(f"Profil (LE={x_le:.2f}, TE={x_te:.2f}) hors du domaine [0, {Lx}].")

    cs_u = PchipInterpolator(_ONERAD_UPPER[:, 0], _ONERAD_UPPER[:, 1])
    cs_l = PchipInterpolator(_ONERAD_LOWER[:, 0], _ONERAD_LOWER[:, 1])
    xc_scan = np.linspace(0.0, 1.0, 1000)
    t_max = float(np.max(cs_u(xc_scan) - cs_l(xc_scan))) * chord

    dense = oneraD_contour(n_pts=2000, chord=chord, x0=x_le, y0=cy)
    tree = cKDTree(dense)
    kappa = _menger_curvature(dense)

    size_func = lambda pt: local_size_kdtree(pt, tree, cx, chord, t_max, h, size_params, kappa)
    outer_pts, outer_ms = build_outer_boundary(Lx, Ly, size_func, h)
    airfoil_pts, airfoil_ms = sample_airfoil_boundary(dense, 2000, chord, x_le, cy, h)
    refinement = make_airfoil_refinement(tree, cx, cy, chord, t_max, h, size_params, kappa)

    y_camber_mid = 0.5 * (float(cs_u(0.5)) + float(cs_l(0.5)))
    hole_pt = (cx, cy + y_camber_mid * chord)

    mesh = triangulate_with_hole(outer_pts, outer_ms, airfoil_pts, airfoil_ms, hole_pt, refinement)
    mesh.set_metadata(
        case="oneraD", h=h, domain={"Lx": Lx, "Ly": Ly},
        obstacle_length=chord, chord=chord,
        thickness=t_max / chord, height=t_max,
        center={"cx": cx, "cy": cy},
    )
    mesh.print_statistics()

    path = mesh_dir / f"oneraD_h{h}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path


if __name__ == "__main__":
    Lx = 4.0; Ly = 4.0
    chord = 1.0; cx = 1.5; cy = Ly / 2

    for h in [0.2, 0.1, 0.05, 0.025]:
        mesh, path = build_mesh(Lx=Lx, Ly=Ly, h=h, chord=chord, cx=cx, cy=cy, export_vtk=False)
        mesh.plot_mesh(filename=mesh_dir / f"oneraD_h{h}.png", dpi=500)
