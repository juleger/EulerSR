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

mesh_dir = repo_root / "meshes" / "rae2822"
mesh_dir.mkdir(parents=True, exist_ok=True)

DEFAULT_SIZE_PARAMS = MeshSizeParams()

def _load_dat(path: Path):
    """Parse un fichier .dat RAE 2822 au format AGARD/UIUC (65+65 points)."""
    pts = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
            if 0.0 <= x <= 1.0:
                pts.append([x, y])
        except ValueError:
            continue
    pts = np.array(pts)
    if len(pts) != 130:
        raise ValueError(f"Attendu 130 points (65+65), obtenu {len(pts)} dans {path}")
    return pts[:65], pts[65:]


_DAT_PATH = Path(__file__).parent / "rae2822.dat"
_RAE2822_UPPER, _RAE2822_LOWER = _load_dat(_DAT_PATH)


def rae2822_contour(n_pts: int = 300, chord: float = 1.0, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    cs_upper = PchipInterpolator(_RAE2822_UPPER[:, 0], _RAE2822_UPPER[:, 1])
    cs_lower = PchipInterpolator(_RAE2822_LOWER[:, 0], _RAE2822_LOWER[:, 1])

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

    cs_u = PchipInterpolator(_RAE2822_UPPER[:, 0], _RAE2822_UPPER[:, 1])
    cs_l = PchipInterpolator(_RAE2822_LOWER[:, 0], _RAE2822_LOWER[:, 1])
    xc_scan = np.linspace(0.0, 1.0, 1000)
    t_max = float(np.max(cs_u(xc_scan) - cs_l(xc_scan))) * chord

    dense = rae2822_contour(n_pts=2000, chord=chord, x0=x_le, y0=cy)
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
        case="rae2822", h=h, domain={"Lx": Lx, "Ly": Ly},
        obstacle_length=chord, chord=chord,
        thickness=t_max / chord, height=t_max,
        center={"cx": cx, "cy": cy},
    )
    mesh.print_statistics()

    path = mesh_dir / f"rae2822_h{h}.npy"
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
        mesh.plot_mesh(filename=mesh_dir / f"rae2822_h{h}.png", dpi=500)
