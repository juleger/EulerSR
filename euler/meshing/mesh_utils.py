"""Utilitaires communs aux modules de maillage (airfoils, diamond, bump)."""
from dataclasses import dataclass
import numpy as np
import meshpy.triangle as triangle
from scipy.spatial import cKDTree
from pathlib import Path
import sys

# Ajout du répertoire euler/ pour permettre l'import de jax_fvm
_euler_dir = str(Path(__file__).resolve().parents[1])
if _euler_dir not in sys.path:
    sys.path.insert(0, _euler_dir)

from jax_fvm.src.mesh import Mesh

WALL, INLET, OUTLET = 2, 3, 4


@dataclass(frozen=True)
class MeshSizeParams:
    growth_rate:        float = 0.10
    left_growth_rate:   float = 0.25
    obstacle_factor:    float = 0.0
    max_size_factor:    float = 6.0
    left_margin_factor: float = 5.0


def sample_segment(p0, p1, target_size_func, min_step) -> np.ndarray:
    """Points adaptatifs le long d'un segment selon target_size_func"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length == 0.0:
        return np.empty((0, 2), dtype=float)

    direction = delta / length
    samples = [p0]
    s = 0.0
    while s < length:
        midpoint = p0 + direction * min(s + 0.5 * min_step, length)
        step = max(min_step, float(target_size_func(midpoint)))
        next_s = min(length, s + step)
        if next_s >= length or (length - next_s) < 0.5 * min_step:
            break
        samples.append(p0 + direction * next_s)
        s = next_s
    return np.asarray(samples, dtype=float)


def triangle_area_from_h(h: float) -> float:
    return np.sqrt(3.0) * h**2 / 4.0


def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def _menger_curvature(pts: np.ndarray) -> np.ndarray:
    """Courbure de Menger discrète pour raffinement adaptatif au bord d'attaque."""
    n = len(pts)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        cross = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
        ab, bc, ca = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)
        denom = ab * bc * ca
        kappa[i] = 2.0 * cross / denom if denom > 1e-20 else 0.0
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def _sample_half_contour(half_pts: np.ndarray, h_body: float, h_min: float) -> np.ndarray:
    """Échantillonnage adaptatif d'une demi-surface de profil selon la courbure."""
    diffs = np.diff(half_pts, axis=0)
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    total = arclen[-1]

    # Courbure évaluée sur un ré-échantillonnage à pas d'arc uniforme
    n_uni = max(int(total / (0.5 * h_min)) + 1, 8)
    s_uni = np.linspace(0.0, total, n_uni)
    xy_uni = np.column_stack([np.interp(s_uni, arclen, half_pts[:, 0]),
                              np.interp(s_uni, arclen, half_pts[:, 1])])
    kappa = _menger_curvature(xy_uni)

    pts, s = [], 0.0
    while s < total:
        pts.append([float(np.interp(s, arclen, half_pts[:, 0])),
                    float(np.interp(s, arclen, half_pts[:, 1]))])
        k = max(float(np.interp(s, s_uni, kappa)), 1e-10)
        s += float(np.clip(h_body / k * 2, h_min, h_body))
    return np.array(pts)


def sample_airfoil_boundary(dense: np.ndarray, n_half: int, chord: float, x0: float, y0: float, h: float):
    """Frontière discrète d'un profil (extrados + BF + intrados) selon la courbure et la taille h"""
    h_body = h * 2.0 / 3.0
    h_min = h / 5.0
    upper = dense[:n_half]
    lower = np.concatenate([dense[:1], dense[n_half:][::-1], dense[n_half - 1:n_half]])
    upper_pts = _sample_half_contour(upper, h_body, h_min)
    lower_pts = _sample_half_contour(lower, h_body, h_min)
    te_pt = np.array([[x0 + chord, y0]])
    # Unique point de jonction TE si le dernier point de chaque demi-surface est proche du TE
    if len(upper_pts) > 1 and np.linalg.norm(upper_pts[-1] - te_pt[0]) < 0.5 * h_body:
        upper_pts = upper_pts[:-1]
    if len(lower_pts) > 1 and np.linalg.norm(lower_pts[-1] - te_pt[0]) < 0.5 * h_body:
        lower_pts = lower_pts[:-1]
    pts = np.concatenate([upper_pts, te_pt, lower_pts[::-1][:-1]])
    return pts, [WALL] * len(pts)


def local_size_kdtree(point, tree, cx: float, chord: float, max_thickness: float, 
                    h: float, size_params: MeshSizeParams, kappa_array=None) -> float:
    x_le = cx - chord / 2.0
    px = float(point[0])
    p_arr = np.array(point, dtype=np.float64)
    dist, idx = tree.query(p_arr)
    dist = float(dist)

    if dist <= size_params.obstacle_factor * max_thickness:
        return float(h)

    growth_length = max(max_thickness, h)

    if kappa_array is not None:
        k = max(float(kappa_array[idx]), 1e-10)
        h_surf = float(np.clip(h / k, h / 5.0, h * 2.0 / 3.0))
        h_base = h_surf + min(1.0, dist / h) * (h - h_surf)
    else:
        h_base = h

    left_gap = x_le - px
    left_transition = max(size_params.left_margin_factor * max_thickness, 4.0 * h)
    left_blend = smoothstep(left_gap / left_transition)
    growth_rate = size_params.growth_rate + left_blend * (size_params.left_growth_rate - size_params.growth_rate)
    max_size_factor = size_params.max_size_factor * (1.0 + 3.0 * left_blend)

    target = h_base * (1.0 + growth_rate * dist / growth_length)
    return float(np.clip(target, h_base, max_size_factor * h))


def make_airfoil_refinement(tree, cx: float, cy: float, chord: float, max_thickness: float, h: float,
       size_params: MeshSizeParams, kappa_array=None):
    """Retourne la fonction de raffinement pour meshpy (profils naca0012/rae2822)"""
    def refinement_func(vertices, area):
        centroid = np.mean([(v.x, v.y) for v in vertices], axis=0)
        target_h = local_size_kdtree(centroid, tree, cx, chord, max_thickness, h, size_params, kappa_array)
        return bool(area > triangle_area_from_h(target_h))
    return refinement_func

def build_outer_boundary(Lx: float, Ly: float, size_func, h: float):
    """Points et marqueurs sur le contour rectangulaire du domaine"""
    outer = [(0.0, 0.0), (Lx, 0.0), (Lx, Ly), (0.0, Ly)]
    pts, ms, last = [], [], None
    for p0, p1, marker in [
        (outer[0], outer[1], OUTLET),
        (outer[1], outer[2], OUTLET),
        (outer[2], outer[3], OUTLET),
        (outer[3], outer[0], INLET),
    ]:
        seg = sample_segment(p0, p1, size_func, min_step=h)
        if last is not None and len(seg) > 0 and np.allclose(seg[0], last):
            seg = seg[1:]
        pts.append(seg)
        ms.extend([marker] * len(seg))
        if len(seg) > 0:
            last = seg[-1]
    return np.concatenate(pts), ms


def populate_mesh_from_triangle(mesh, raw_mesh):
    mesh.mesh_generator_from_points(
        raw_mesh.points, raw_mesh.elements,
        np.roll(np.asarray(raw_mesh.neighbors), 1, axis=-1),
        raw_mesh.faces, raw_mesh.face_markers,
    )
    return mesh


def triangulate_with_hole(outer_pts, outer_ms, obstacle_pts, obstacle_ms, hole_pt, refinement_func) -> Mesh:
    """Assemble la triangulation Delaunay avec obstacle creux"""
    all_pts = np.concatenate([outer_pts, obstacle_pts])
    off = len(outer_pts)
    mesh = Mesh()
    facets = (mesh.round_trip_connect(0, off - 1) + mesh.round_trip_connect(off, off + len(obstacle_pts) - 1))

    info = triangle.MeshInfo()
    info.set_points([tuple(pt) for pt in all_pts])
    info.set_facets(facets, facet_markers=list(outer_ms) + list(obstacle_ms))
    info.set_holes([hole_pt])

    raw_mesh = triangle.build(
        info, refinement_func=refinement_func,
        min_angle=30, generate_faces=True, generate_neighbor_lists=True,
    )
    return populate_mesh_from_triangle(Mesh(), raw_mesh)


DEFAULT_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "meshes"


def build_airfoil_mesh(case, dense, t_max, hole_pt, *, Lx, Ly, h, chord, cx, cy,
                       symmetric, size_params, extra_metadata,
                       out_dir=None, export_vtk=False):
    x_le = cx - chord / 2.0
    x_te = cx + chord / 2.0
    tree = cKDTree(dense)
    kappa = _menger_curvature(dense)
    size_func = lambda pt: local_size_kdtree(pt, tree, cx, chord, t_max, h, size_params, kappa)
    refinement = make_airfoil_refinement(tree, cx, cy, chord, t_max, h, size_params, kappa)

    if symmetric:
        airfoil_upper = sample_airfoil_upper(dense, 2000, chord, x_le, cy, h)
        mesh = triangulate_symmetric_airfoil(Lx, Ly, cy, x_le, x_te, airfoil_upper,
                                             size_func, refinement, h)
    else:
        outer_pts, outer_ms = build_outer_boundary(Lx, Ly, size_func, h)
        airfoil_pts, airfoil_ms = sample_airfoil_boundary(dense, 2000, chord, x_le, cy, h)
        mesh = triangulate_with_hole(outer_pts, outer_ms, airfoil_pts, airfoil_ms, hole_pt, refinement)

    mesh.set_metadata(case=case, h=h, domain={"Lx": Lx, "Ly": Ly},
                      obstacle_length=chord, chord=chord, **extra_metadata)
    mesh.print_statistics()

    out_dir = DEFAULT_MESH_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case}_h{h}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path


# Maillage symétrique : maillage supérieur puis miroir sur l'axe y=cy

def sample_airfoil_upper(dense: np.ndarray, n_half: int, chord: float, x0: float, y0: float, h: float):
    """Surface supérieure LE→BF seule (extrémités sur l'axe y0), pour maillage symétrique."""
    h_body = h * 2.0 / 3.0
    h_min = h / 5.0
    upper_pts = _sample_half_contour(dense[:n_half], h_body, h_min)
    te_pt = np.array([[x0 + chord, y0]])
    if len(upper_pts) > 1 and np.linalg.norm(upper_pts[-1] - te_pt[0]) < 0.5 * h_body:
        upper_pts = upper_pts[:-1]
    return np.concatenate([upper_pts, te_pt])


def _mirror_upper_half(pts_u: np.ndarray, tris_u: np.ndarray, cy: float, tol: float = 1e-9):
    """Reflète le demi-maillage supérieur sous l'axe y=cy"""
    on_axis = np.abs(pts_u[:, 1] - cy) < tol
    n_u = len(pts_u)
    mir = np.arange(n_u)
    lower = []
    nxt = n_u
    for i in range(n_u):
        if not on_axis[i]:
            mir[i] = nxt
            nxt += 1
            lower.append((pts_u[i, 0], 2.0 * cy - pts_u[i, 1]))
    points = np.vstack([pts_u, np.array(lower, dtype=float).reshape(-1, 2)])
    tris_l = mir[tris_u[:, ::-1]]
    tris = np.vstack([tris_u, tris_l]).astype(np.int32)
    return points, tris


def _faces_neighbors_from_tris(tris: np.ndarray):
    """Reconstruit faces (arêtes uniques), voisins (convention arête i=(v_i,v_i+1)) et masque de bord."""
    from collections import defaultdict
    edge_map = defaultdict(list)
    for t, tri in enumerate(tris):
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge_map[(a, b) if a < b else (b, a)].append((t, i))
    n_tris = len(tris)
    neighbors = -np.ones((n_tris, 3), dtype=np.int32)
    faces = np.empty((len(edge_map), 2), dtype=np.int32)
    boundary = np.zeros(len(edge_map), dtype=bool)
    for fidx, (key, owners) in enumerate(edge_map.items()):
        faces[fidx] = key
        if len(owners) == 2:
            (t0, i0), (t1, i1) = owners
            neighbors[t0, i0] = t1
            neighbors[t1, i1] = t0
        else:
            boundary[fidx] = True
    return faces, neighbors, boundary


def _classify_face_markers(points, faces, boundary, Lx, Ly, tol=1e-6):
    """Marqueurs de bord par géométrie : gauche=INLET, autres bords rect.=OUTLET, sinon WALL."""
    markers = np.zeros(len(faces), dtype=np.int32)
    for f in np.where(boundary)[0]:
        (xa, ya), (xb, yb) = points[faces[f, 0]], points[faces[f, 1]]
        if xa < tol and xb < tol:
            markers[f] = INLET
        elif ((xa > Lx - tol and xb > Lx - tol) or (ya < tol and yb < tol)
              or (ya > Ly - tol and yb > Ly - tol)):
            markers[f] = OUTLET
        else:
            markers[f] = WALL
    return markers


def triangulate_symmetric_airfoil(Lx, Ly, cy, x_le, x_te, airfoil_upper,size_func, refinement_func, h) -> Mesh:
    """Maille la moitié supérieure puis reflete"""
    seg_l = sample_segment((0.0, cy), (x_le, cy), size_func, min_step=h)
    seg_r = sample_segment((x_te, cy), (Lx, cy), size_func, min_step=h)
    right = sample_segment((Lx, cy), (Lx, Ly), size_func, min_step=h)
    top = sample_segment((Lx, Ly), (0.0, Ly), size_func, min_step=h)
    left = sample_segment((0.0, Ly), (0.0, cy), size_func, min_step=h)
    loop = np.concatenate([seg_l, airfoil_upper, seg_r[1:], right, top, left])

    mesh = Mesh()
    facets = mesh.round_trip_connect(0, len(loop) - 1)
    info = triangle.MeshInfo()
    info.set_points([tuple(pt) for pt in loop])
    info.set_facets(facets)
    raw = triangle.build(
        info, refinement_func=refinement_func,
        min_angle=30, generate_faces=True, generate_neighbor_lists=True,
    )

    points, tris = _mirror_upper_half(np.asarray(raw.points), np.asarray(raw.elements), cy)
    faces, neighbors, boundary = _faces_neighbors_from_tris(tris)
    markers = _classify_face_markers(points, faces, boundary, Lx, Ly)

    out = Mesh()
    out.mesh_generator_from_points(points, tris, neighbors, faces, markers)
    return out
