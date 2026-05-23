from pathlib import Path
from dataclasses import dataclass
import sys
import numpy as np
import meshpy.triangle as triangle

repo_root = Path(__file__).resolve().parents[1]

from jax_fvm.src.mesh import Mesh

WALL, INLET, OUTLET = 2, 3, 4
mesh_dir = repo_root / "meshes" / "diamond"
mesh_dir.mkdir(parents=True, exist_ok=True)

# Dataclass pour les paramètres "adaptatifs" du maillage diamant
@dataclass(frozen=True)
class DiamondMeshSize:
    growth_rate: float = 0.065
    left_growth_rate: float = 0.10
    obstacle_factor: float = 1.0
    max_size_factor: float = 2.0
    left_margin_factor: float = 1.0


DEFAULT_SIZE_PARAMS = DiamondMeshSize()


def sample_loop(verts, h, markers):
    # Génère des points le long d'une boucle fermée de vertices, avec une taille de maille h
    # En l'occurrence, afin de mailler le diamant
    pts, ms = [], []
    for i, m in enumerate(markers):
        p0, p1 = np.array(verts[i]), np.array(verts[(i + 1) % len(verts)])
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / h)))
        seg = np.linspace(p0, p1, n, endpoint=False)
        pts.append(seg)
        ms.extend([m] * len(seg))
    return np.concatenate(pts), ms


def sample_segment(p0, p1, target_size_func, min_step):
    # Permet de génerer des points le long d'un segment, avec une taille de maille variable
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length == 0.0:
        return np.empty((0, 2), dtype=float)

    direction = delta / length
    samples = [p0]
    s = 0.0

    # On échantillonne le long du segment en adaptant la taille des pas selon target_size_func
    while s < length:
        midpoint = p0 + direction * min(s + 0.5 * min_step, length)
        step = max(min_step, float(target_size_func(midpoint)))
        next_s = min(length, s + step)
        if next_s >= length or (length - next_s) < 0.5 * min_step:
            break
        samples.append(p0 + direction * next_s)
        s = next_s

    return np.asarray(samples, dtype=float)


def triangle_area_from_h(h):
     
    return np.sqrt(3.0) * h**2 / 4.0


def distance_diamond(point, cx, cy, chord, height):
    # Renvoie la distance d'un point au losange
    px = (point[0] - cx) / (chord / 2.0)
    py = (point[1] - cy) / (height / 2.0)
    if abs(px) + abs(py) <= 1.0:
        return 0.0
    s = abs(px) + abs(py)
    nearest = np.array([cx + np.sign(px) * (abs(px) / s) * (chord / 2.0),
                        cy + np.sign(py) * (abs(py) / s) * (height / 2.0)])
    return float(np.linalg.norm(np.array(point) - nearest))


def local_size(point, cx, cy, chord, height, h, size_params=DEFAULT_SIZE_PARAMS):
    # Simule la croissance de la taille des mailles à partir du diamants

    # Abscisse du bord gauche du diamant
    x_le = cx - chord / 2.0
    px = float(point[0])

    dist = distance_diamond(point, cx, cy, chord, height)
    if dist <= size_params.obstacle_factor * height:
        return float(h)

    growth_length = max(height, h)

    # Croissance moins rapide à droite du diamant pour garder une bonne résolution
    # Le maillage devient plus grossier plus rapidement quand on se rapproche de l'inlet
    if px < x_le - size_params.left_margin_factor * height:
        growth_rate = size_params.left_growth_rate
        max_size_factor = size_params.max_size_factor
    else:
        growth_rate = size_params.growth_rate
        max_size_factor = size_params.max_size_factor * 4

    target = h * (1.0 + growth_rate * dist / growth_length)
    return float(np.clip(target, h, max_size_factor * h))


def diamond_refinement(cx, cy, chord, height, h, size_params=DEFAULT_SIZE_PARAMS):
    # Fonction de raffinement pour le maillage du diamant, basé sur local_size
    def refinement_func(vertices, area):
        centroid = np.mean([(vertex.x, vertex.y) for vertex in vertices], axis=0)
        target_h = local_size(
            centroid, cx, cy, chord, height, h, size_params=size_params,
        )
        return bool(area > triangle_area_from_h(target_h))
    return refinement_func


def populate_mesh_from_triangle(mesh, raw_mesh):
    # Convertit le maillage brut de triangle vers la classe Mesh
    mesh.mesh_generator_from_points(raw_mesh.points, raw_mesh.elements, 
        np.roll(np.asarray(raw_mesh.neighbors), 1, axis=-1), raw_mesh.faces, raw_mesh.face_markers,
    )
    return mesh


def build_mesh(Lx=6.0, Ly=4.0, h=0.05, chord=1.0, height=0.24, cx=None, cy=None,
               export_vtk=False, size_params=DEFAULT_SIZE_PARAMS):

    cx = Lx / 2 if cx is None else cx
    cy = Ly / 2 if cy is None else cy
    hc, ht = chord / 2, height / 2

    # Domaine rectangulaire extérieur
    outer = [(0.0, 0.0), (Lx, 0.0), (Lx, Ly), (0.0, Ly)]

    # Diamant intérieur centré en (cx, cy) de longueur chord et hauteur height
    diamond = [(cx-hc, cy), (cx, cy+ht), (cx+hc, cy), (cx, cy-ht)]

    if diamond[0][0] < 0 or diamond[2][0] > Lx or diamond[3][1] < 0 or diamond[1][1] > Ly:
        raise ValueError("Le diamant doit rester entièrement dans le domaine [0, Lx] x [0, Ly].")

    # Renvoie la taille cible pour un point donné du domain
    size_func = lambda point: local_size(point, cx, cy, chord, height, h, size_params=size_params)

    outer_pts = []
    outer_ms = []
    last_point = None

    # Sur les 4 côtés, l'inlet est à gauche, tout le reste est outlet
    for p0, p1, marker in [
        (outer[0], outer[1], OUTLET),
        (outer[1], outer[2], OUTLET),
        (outer[2], outer[3], OUTLET),
        (outer[3], outer[0], INLET),
    ]:
        target_h = h
        segment = sample_segment(p0, p1, size_func, min_step=target_h)
        if last_point is not None and len(segment) > 0 and np.allclose(segment[0], last_point):
            segment = segment[1:]
        outer_pts.append(segment)
        outer_ms.extend([marker] * len(segment))
        if len(segment) > 0:
            last_point = segment[-1]
    outer_pts = np.concatenate(outer_pts)

    # Création des points de la frontière du diamant
    diamond_pts, diamond_ms = sample_loop(diamond, h, [WALL, WALL, WALL, WALL])

    all_pts = np.concatenate([outer_pts, diamond_pts])
    off = len(outer_pts)
    mesh = Mesh()

    # Création des facettes pour la triangulation (indices + marqueurs)
    facets = (mesh.round_trip_connect(0, off - 1)
               + mesh.round_trip_connect(off, off + len(diamond_pts) - 1))

    info = triangle.MeshInfo()
    info.set_points([tuple(p) for p in all_pts])
    info.set_facets(facets, facet_markers=outer_ms + diamond_ms)
    info.set_holes([(cx, cy)])

    # Génération du maillage avec triangle, avec fonction de raffinement progressif
    raw_mesh = triangle.build(
        info,
        refinement_func=diamond_refinement(cx, cy, chord, height, h, size_params=size_params),
        min_angle=30, generate_faces=True, generate_neighbor_lists=True,
    )

    # Conversion du maillage brut de triangle vers la classe Mesh
    mesh = populate_mesh_from_triangle(Mesh(), raw_mesh)
    mesh.set_metadata(case="diamond", h=h, domain={"Lx": Lx, "Ly": Ly},
        obstacle_length=chord, chord=chord, height=height, center={"cx": cx, "cy": cy})
    mesh.print_statistics()

    path = mesh_dir / f"diamond_h{h}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path

if __name__ == "__main__":
    Lx = 4; Ly = 4.0; h = 0.1
    chord = 1.0; cx = 1.5; cy = Ly / 2

    # Alpha est le demi angle du losange, height est la hauteur totale du losange
    alpha = np.radians(10)
    height = chord * np.tan(alpha)

    for h in [0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025, 0.0175, 0.0125]:
        mesh, path = build_mesh(Lx=Lx, Ly=Ly, h=h, chord=chord, height=height, cx=cx, cy=cy, export_vtk=False)
        mesh.plot_mesh(filename=mesh_dir / f"diamond_h{h}.png")