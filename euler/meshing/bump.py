from pathlib import Path
from dataclasses import dataclass
import sys
import numpy as np
import meshpy.triangle as triangle

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # euler/ (jax_fvm)

from jax_fvm.src.mesh import Mesh

BUMP, INLET, OUTLET, WALL = 2, 3, 4, 6
DEFAULT_OUT_DIR = repo_root / "data" / "meshes"


@dataclass(frozen=True)
class BumpMeshSize:
    growth_rate: float = 0.02
    left_growth_rate: float = 0.3
    bump_factor: float = 0.5
    max_size_factor: float = 4
    left_margin: float = 0.1


DEFAULT_SIZE_PARAMS = BumpMeshSize()


def triangle_area_from_h(h):
    return np.sqrt(3.0) * h**2 / 4.0


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def sample_segment(p0, p1, target_size_func, min_step):
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


def sample_circular_segment(center_x, circle_center_y, circle_radius, chord, min_step):
    half_width = chord / 2.0
    x_left = center_x - half_width
    x_right = center_x + half_width
    arc_length = chord
    if arc_length == 0.0:
        return np.empty((0, 2), dtype=float)

    n_points = max(3, int(np.ceil(arc_length / min_step )))
    xs = np.linspace(x_left, x_right, n_points, endpoint=False)
    inside = np.clip(circle_radius**2 - (xs - center_x) ** 2, 0.0, None)
    ys = circle_center_y + np.sqrt(inside)
    return np.stack([xs, ys], axis=-1)


def distance_to_bump(point, center_x, circle_center_y, circle_radius, chord):
    point = np.asarray(point, dtype=float)
    dx = float(point[0] - center_x)
    dy = float(point[1] - circle_center_y)
    half_chord = chord / 2.0

    if circle_radius <= 0.0:
        return float(np.hypot(dx, dy))

    theta_left = np.arctan2(np.sqrt(max(circle_radius**2 - half_chord**2, 0.0)), -half_chord)
    theta_right = np.arctan2(np.sqrt(max(circle_radius**2 - half_chord**2, 0.0)), half_chord)
    theta = np.arctan2(dy, dx)
    theta_clamped = np.clip(theta, theta_right, theta_left)

    nearest = np.array([
        center_x + circle_radius * np.cos(theta_clamped),
        circle_center_y + circle_radius * np.sin(theta_clamped),
    ])
    return float(np.linalg.norm(point - nearest))


def local_size(point, center_x, circle_center_y, circle_radius, chord, thickness, h,
               size_params=DEFAULT_SIZE_PARAMS):
    dist = distance_to_bump(point, center_x, circle_center_y, circle_radius, chord)
    fine_radius = size_params.bump_factor * thickness
    # bump_factor est calibré sur la hauteur physique du bump (ici 'thickness')
    if dist <= fine_radius:
        return float(h)

    growth_length = max(chord / 2.0, h)
    px = float(point[0])
    x_le = center_x - chord / 2.0
    if px < x_le:
        left_gap = x_le - px
        left_transition = max(0.25 * chord, 3.0 * thickness)
        left_blend = smoothstep(left_gap / left_transition)
        growth_rate = size_params.growth_rate + left_blend * (
            size_params.left_growth_rate - size_params.growth_rate
        )
        max_size_factor = size_params.max_size_factor * (1.0 + 3.0 * left_blend)
    else:
        growth_rate = size_params.growth_rate
        max_size_factor = size_params.max_size_factor * 4.0

    target = h * (1.0 + growth_rate * dist / growth_length)
    target = float(np.clip(target, h, max_size_factor * h))
    transition_length = max(1.5 * thickness, 4.0 * h)
    blend = smoothstep((dist - fine_radius) / transition_length)

    return float(h + blend * (target - h))


def bump_refinement(center_x, circle_center_y, circle_radius, chord, thickness, h,
                    size_params=DEFAULT_SIZE_PARAMS):
    def refinement_func(vertices, area):
        centroid = np.mean([(vertex.x, vertex.y) for vertex in vertices], axis=0)
        target_h = local_size(centroid, center_x, circle_center_y, circle_radius,
            chord, thickness, h, size_params=size_params)
        return bool(area > triangle_area_from_h(target_h))

    return refinement_func


def build_mesh(Lx=3.0, Ly=1.0, h=0.05, thickness=0.08, chord=0.25, center=1.0, export_vtk=False,
               size_params=DEFAULT_SIZE_PARAMS, out_dir=None):
    # Création d'un maillage pour un bump circulaire coupé au mur inférieur.

    if thickness <= 0.0:
        raise ValueError("La thickness du bump doit être strictement positive.")
    if chord <= 0.0:
        raise ValueError("La largeur (chord) du bump doit être strictement positive.")
    if center - chord / 2.0 < 0.0 or center + chord / 2.0 > Lx:
        raise ValueError("Le bump doit rester entièrement dans le domaine [0, Lx].")

    maxV = triangle_area_from_h(h)
    circle_radius = ((chord / 2.0)**2 + thickness**2) / (2.0 * thickness)
    circle_center_y = thickness - circle_radius
    size_func = lambda point: local_size(point, center, circle_center_y, circle_radius,
        chord, thickness, h, size_params=size_params)

    bottom_left = sample_segment((0.0, 0.0), (center - chord / 2.0, 0.0), size_func, min_step=h)
    bump = sample_circular_segment(center, circle_center_y, circle_radius, chord, min_step=h)
    bottom_right = sample_segment((center + chord / 2.0, 0.0), (Lx, 0.0), size_func, min_step=h)
    right = sample_segment((Lx, 0.0), (Lx, Ly), size_func, min_step=h)
    top = sample_segment((Lx, Ly), (0.0, Ly), size_func, min_step=h)
    left = sample_segment((0.0, Ly), (0.0, 0.0), size_func, min_step=h)

    bounds = np.concatenate([bottom_left, bump, bottom_right, right, top, left])[:-1]
    markers = ([WALL] * len(bottom_left) + [BUMP] * len(bump) + [WALL] * len(bottom_right)
               + [OUTLET] * len(right) + [WALL] * len(top) + [INLET] * len(left))

    mesh = Mesh()
    bounds, markers = mesh.clean_boundaries(bounds, markers)
    info = triangle.MeshInfo()
    info.set_points([tuple(p) for p in bounds])
    info.set_facets(mesh.round_trip_connect(0, len(bounds) - 1), facet_markers=markers)
    raw_mesh = triangle.build(
        info,
        refinement_func=bump_refinement(
            center,
            circle_center_y,
            circle_radius,
            chord,
            thickness,
            h,
            size_params=size_params,
        ),
        min_angle=30,
        generate_faces=True,
        generate_neighbor_lists=True,
    )

    mesh = Mesh()
    mesh.mesh_generator_from_points(
        raw_mesh.points,
        raw_mesh.elements,
        np.roll(np.asarray(raw_mesh.neighbors), 1, axis=-1),
        raw_mesh.faces,
        raw_mesh.face_markers,
    )
    mesh.set_metadata(
        case="bump",
        h=h,
        domain={"Lx": Lx, "Ly": Ly},
        center=center,
        thickness=thickness,
        chord=chord,
        obstacle_length=chord,
        bump_marker=BUMP,
        force_marker=BUMP,
        wall_markers=[BUMP, WALL],
        inlet_marker=INLET,
        outlet_marker=OUTLET,
        boundary_layout={"top": "wall", "left": "inlet", "right": "outlet"},
    )
    mesh.print_statistics()

    out_dir = DEFAULT_OUT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bump_h{h}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path

if __name__ == "__main__":

    Lx = 3.0; Ly = 1.0
    thickness = 0.04; chord = 1.0; center = 1.5

    for h in [0.02]:
        mesh, path = build_mesh(Lx=Lx, Ly=Ly, h=h, thickness=thickness, chord=chord, center=center, export_vtk=False)
        mesh.plot_mesh(filename=str(path.with_suffix(".png")), dpi=500)