from pathlib import Path
from dataclasses import dataclass
import numpy as np
import meshpy.triangle as triangle

repo_root = Path(__file__).resolve().parents[1]

from jax_fvm.src.mesh import Mesh

BUMP, INLET, OUTLET, WALL = 2, 3, 4, 6
mesh_dir = repo_root / "meshes" / "bump"
mesh_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BumpMeshSize:
    growth_rate: float = 0.12
    left_growth_rate: float = 0.3
    bump_factor: float = 0.5
    max_size_factor: float = 2.5
    left_margin_factor: float = 1.5


DEFAULT_SIZE_PARAMS = BumpMeshSize()


def triangle_area_from_h(h):
    return np.sqrt(3.0) * h**2 / 4.0


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


def sample_circular_segment(center_x, circle_center_y, circle_radius, half_width, min_step):
    x_left = center_x - half_width
    x_right = center_x + half_width
    arc_length = 2.0 * half_width
    if arc_length == 0.0:
        return np.empty((0, 2), dtype=float)

    n_points = max(3, int(np.ceil(arc_length / min_step )))
    xs = np.linspace(x_left, x_right, n_points, endpoint=False)
    inside = np.clip(circle_radius**2 - (xs - center_x) ** 2, 0.0, None)
    ys = circle_center_y + np.sqrt(inside)
    return np.stack([xs, ys], axis=-1)


def distance_to_bump(point, center_x, circle_center_y, circle_radius, bump_half_width):
    point = np.asarray(point, dtype=float)
    dx = float(point[0] - center_x)
    dy = float(point[1] - circle_center_y)

    radial_dist = abs((dx**2 + dy**2) ** 0.5 - circle_radius)
    extra_x = max(0.0, abs(dx) - bump_half_width)
    return radial_dist + extra_x


def local_size(point, center_x, circle_center_y, circle_radius, bump_half_width, bump_height, h,
               size_params=DEFAULT_SIZE_PARAMS):
    dist = distance_to_bump(point, center_x, circle_center_y, circle_radius, bump_half_width)
    # bump_factor est calibré sur la hauteur physique du bump
    if dist <= size_params.bump_factor * bump_height:
        return float(h)

    growth_length = max(bump_half_width, h)
    px = float(point[0])
    x_le = center_x - bump_half_width

    if px < x_le - size_params.left_margin_factor * bump_height:
        growth_rate = size_params.left_growth_rate
        max_size_factor = size_params.max_size_factor
    else:
        growth_rate = size_params.growth_rate
        max_size_factor = size_params.max_size_factor * 2.0

    target = h * (1.0 + growth_rate * dist / growth_length)
    return float(np.clip(target, h, max_size_factor * h))


def bump_refinement(center_x, circle_center_y, circle_radius, bump_half_width, bump_height, h,
                    size_params=DEFAULT_SIZE_PARAMS):
    def refinement_func(vertices, area):
        centroid = np.mean([(vertex.x, vertex.y) for vertex in vertices], axis=0)
        target_h = local_size(centroid, center_x, circle_center_y, circle_radius,
            bump_half_width, bump_height, h, size_params=size_params)
        return bool(area > triangle_area_from_h(target_h))

    return refinement_func


def build_mesh(Lx=3.0, Ly=1.0, h=0.05, bump_height=0.08, bump_half_width=0.25, center=1.0, export_vtk=False,
               size_params=DEFAULT_SIZE_PARAMS):
    # Création d'un maillage pour un bump circulaire coupé au mur inférieur.

    if bump_height <= 0.0:
        raise ValueError("La hauteur du bump doit être strictement positive.")
    if bump_half_width <= 0.0:
        raise ValueError("La demi-largeur du bump doit être strictement positive.")
    if center - bump_half_width < 0.0 or center + bump_half_width > Lx:
        raise ValueError("Le bump doit rester entièrement dans le domaine [0, Lx].")

    maxV = triangle_area_from_h(h)
    circle_radius = (bump_half_width**2 + bump_height**2) / (2.0 * bump_height)
    circle_center_y = bump_height - circle_radius
    size_func = lambda point: local_size(point, center, circle_center_y, circle_radius,
        bump_half_width, bump_height, h, size_params=size_params)

    bottom_left = sample_segment((0.0, 0.0), (center - bump_half_width, 0.0), size_func, min_step=h)
    bump = sample_circular_segment(center, circle_center_y, circle_radius, bump_half_width, min_step=h)
    bottom_right = sample_segment((center + bump_half_width, 0.0), (Lx, 0.0), size_func, min_step=h)
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
            bump_half_width,
            bump_height,
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
        bump_height=bump_height,
        bump_half_width=bump_half_width,
        obstacle_length=2.0 * bump_half_width,
        bump_marker=BUMP,
        force_marker=BUMP,
        wall_markers=[BUMP, WALL],
        inlet_marker=INLET,
        outlet_marker=OUTLET,
        boundary_layout={"top": "wall", "left": "inlet", "right": "outlet"},
    )
    mesh.print_statistics()

    path = mesh_dir / f"bump_h{h}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path

if __name__ == "__main__":

    Lx = 3.0; Ly = 1.0
    bump_height = 0.08; bump_half_width = 0.5; center = 1.5

    for h in [0.2, 0.1, 0.07, 0.05, 0.035, 0.025, 0.0175, 0.0125, 0.00875, 0.00625]:
        mesh, path = build_mesh(Lx=Lx, Ly=Ly, h=h, bump_height=bump_height, bump_half_width=bump_half_width, center=center, export_vtk=False)
        mesh.plot_mesh(filename=mesh_dir / f"bump_h{h}.png", dpi=500)