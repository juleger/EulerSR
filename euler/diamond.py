from pathlib import Path
import sys
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from mesh_utils import (WALL, INLET, OUTLET, MeshSizeParams,
                         triangle_area_from_h, smoothstep,
                         build_outer_boundary, triangulate_with_hole)
from jax_fvm.src.mesh import Mesh

mesh_dir = repo_root / "meshes" / "diamond"
mesh_dir.mkdir(parents=True, exist_ok=True)

DEFAULT_SIZE_PARAMS = MeshSizeParams(growth_rate=0.07, left_growth_rate=0.18)


def sample_loop(verts, h, markers):
    """Points uniformes le long d'une boucle fermée de segments (contour diamant)."""
    pts, ms = [], []
    for i, m in enumerate(markers):
        p0, p1 = np.array(verts[i]), np.array(verts[(i + 1) % len(verts)])
        n   = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / h)))
        seg = np.linspace(p0, p1, n, endpoint=False)
        pts.append(seg)
        ms.extend([m] * len(seg))
    return np.concatenate(pts), ms


def _distance_diamond(point, cx, cy, chord, height):
    px = (point[0] - cx) / (chord / 2.0)
    py = (point[1] - cy) / (height / 2.0)
    if abs(px) + abs(py) <= 1.0:
        return 0.0
    s = abs(px) + abs(py)
    nearest = np.array([cx + np.sign(px) * (abs(px) / s) * (chord / 2.0),
                        cy + np.sign(py) * (abs(py) / s) * (height / 2.0)])
    return float(np.linalg.norm(np.array(point) - nearest))


def _local_size(point, cx, cy, chord, height, h, size_params):
    x_le  = cx - chord / 2.0
    px    = float(point[0])
    dist  = _distance_diamond(point, cx, cy, chord, height)

    if dist <= size_params.obstacle_factor * height:
        return float(h)

    growth_length   = max(height, h)
    left_gap        = x_le - px
    left_transition = max(size_params.left_margin_factor * height, 4.0 * h)
    left_blend      = smoothstep(left_gap / left_transition)
    growth_rate     = size_params.growth_rate + left_blend * (
                      size_params.left_growth_rate - size_params.growth_rate)
    max_size_factor = size_params.max_size_factor * (1.0 + 3.0 * left_blend)

    target = h * (1.0 + growth_rate * dist / growth_length)
    return float(np.clip(target, h, max_size_factor * h))


def _diamond_refinement(cx, cy, chord, height, h, size_params):
    def refinement_func(vertices, area):
        centroid = np.mean([(v.x, v.y) for v in vertices], axis=0)
        return bool(area > triangle_area_from_h(
            _local_size(centroid, cx, cy, chord, height, h, size_params)))
    return refinement_func


def build_mesh(Lx=6.0, Ly=4.0, h=0.05, chord=1.0, height=0.24, cx=None, cy=None,
               export_vtk=False, size_params=DEFAULT_SIZE_PARAMS):

    cx = Lx / 2 if cx is None else cx
    cy = Ly / 2 if cy is None else cy
    hc, ht = chord / 2, height / 2

    diamond = [(cx-hc, cy), (cx, cy+ht), (cx+hc, cy), (cx, cy-ht)]
    if diamond[0][0] < 0 or diamond[2][0] > Lx or diamond[3][1] < 0 or diamond[1][1] > Ly:
        raise ValueError("Le diamant doit rester entièrement dans le domaine.")

    size_func     = lambda pt: _local_size(pt, cx, cy, chord, height, h, size_params)
    outer_pts, outer_ms   = build_outer_boundary(Lx, Ly, size_func, h)
    diamond_pts, diamond_ms = sample_loop(diamond, h, [WALL, WALL, WALL, WALL])
    refinement    = _diamond_refinement(cx, cy, chord, height, h, size_params)

    mesh = triangulate_with_hole(outer_pts, outer_ms, diamond_pts, diamond_ms, (cx, cy), refinement)
    mesh.set_metadata(case="diamond", h=h, domain={"Lx": Lx, "Ly": Ly},
                      obstacle_length=chord, chord=chord, height=height,
                      center={"cx": cx, "cy": cy})
    mesh.print_statistics()

    path = mesh_dir / f"diamond_h{h}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path


if __name__ == "__main__":
    Lx = 4.0; Ly = 4.0
    chord = 1.0; cx = 1.5; cy = Ly / 2
    alpha  = np.radians(5)
    height = chord * np.tan(alpha)

    for h in [0.2]:
        mesh, path = build_mesh(Lx=Lx, Ly=Ly, h=h, chord=chord, height=height,
                                cx=cx, cy=cy, export_vtk=False)
        mesh.plot_mesh(filename=mesh_dir / f"diamond_h{h}.png", dpi=500)
