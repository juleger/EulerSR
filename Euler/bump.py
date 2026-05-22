from pathlib import Path
import sys
import numpy as np
import meshpy.triangle as triangle

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from jax_fvm.src.mesh.mesh import Mesh

WALL, INLET, OUTLET = 2, 3, 4
mesh_dir = repo_root / "meshes" / "bump"
mesh_dir.mkdir(parents=True, exist_ok=True)


def build_mesh(Lx=3.0, Ly=1.0, h=0.05, amplitude=0.1, center=1.0, sigma=0.2, export_vtk=False):
    # Création d'un maillage pour un écoulement sur une bosse (bump) dans un canal rectangulaire
    
    maxV = 3**0.5 * h**2 / 4 # formule empirique pour la taille carac d'un triangle
    y_bump = lambda x: amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    x_bot = np.linspace(0, Lx, int(Lx / h * 1.5) + 1)
    y0 = min(y_bump(x_bot))
    Ns, Nt = int(Ly / h) + 1, int(Lx / h) + 1

    # Génération des points de la frontière
    bottom = np.column_stack((x_bot, y_bump(x_bot)))
    right = np.column_stack((np.full(Ns, Lx), np.linspace(y0, Ly + y0, Ns)))
    top = np.column_stack((np.linspace(Lx, 0, Nt), np.full(Nt, Ly + y0)))
    left = np.column_stack((np.zeros(Ns), np.linspace(Ly + y0, y0, Ns)))

    bounds = np.concatenate([bottom, right, top, left])[:-1]
    # Ajout des marqueurs pour les conditions aux limites
    markers = ([WALL] * (len(bottom) - 1) + [OUTLET] * len(right)
               + [WALL] * len(top) + [INLET] * len(left))

    mesh = Mesh()
    bounds, markers = mesh.clean_boundaries(bounds, markers)
    info = triangle.MeshInfo()
    info.set_points([tuple(p) for p in bounds])
    info.set_facets(mesh.round_trip_connect(0, len(bounds) - 1), facet_markers=markers)
    mesh.mesh_generator(info, maxV=maxV, min_angle=30)
    mesh.set_metadata(case="bump", h=h, domain={"Lx": Lx, "Ly": Ly}, 
        amplitude=amplitude, center=center, sigma=sigma)
    mesh.print_statistics()

    path = mesh_dir / f"Lx{Lx}_Ly{Ly}_h{h}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path

if __name__ == "__main__":

    Lx = 3.0; Ly = 1.0; h = 0.05
    amplitude = 0.1; center = 1.0; sigma = 0.2
    build_mesh(Lx=Lx, Ly=Ly, h=h, amplitude=amplitude, center=center, sigma=sigma, export_vtk=True)