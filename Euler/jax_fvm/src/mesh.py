import sys
import meshio
import meshpy.triangle as triangle
import jax.numpy as jnp
import numpy as np
import jax_fvm.src.plot as plot
import jax
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

sys.modules.setdefault("jax_fvm.src.mesh.mesh", sys.modules[__name__])

size = 14
params = {
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size,
	'axes.titlesize' : size +2,
    'xtick.labelsize' : size+1,
    'ytick.labelsize' : size+1
}
plt.rcParams.update(params)


@jax.tree_util.register_pytree_node_class
class Mesh:
    """ Class Mesh to handle the mesh generation and storage of mesh data """
    """
     Attributes:
        points: array of shape (N_points, 2) containing the coordinates of the points
        tris: array of shape (N_tris, 3) containing the indices of the points forming each triangle
        neighbors: array of shape (N_tris, 3) containing the indices of the neighboring triangles for each triangle
        faces: array of shape (N_faces, 2) containing the indices of the points forming each face
        face_markers: array of shape (N_faces,) containing the markers for each face
        field_data: array of shape (N_points,) containing the field data at each point
        barycenter: array of shape (N_tris, 2) containing the barycenter of each triangle
        area: array of shape (N_tris,) containing the area of each triangle
        surface: array of shape (N_faces,) containing the length of each face
        normals: array of shape (N_tris, 3, 2) containing the normals of each face of each triangle
        midedge: array of shape (N_faces, 2) containing the midpoints of each face
        
    access via mesh.points, mesh.tris, etc.

        BCs handled via face_markers and neighbors:
              - BC_marker = 1 : periodic BCs
              - BC_marker = 2 : wall BCs (no-slip)
              - BC_marker = 3 : inlet BCs (prescribed values)
              - BC_marker = 4 : outlet BCs (free flow)
              - BC_marker = 5 : subsonic inflow 

    """
    def __init__(self, **kwargs):
        self.metadata = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def tree_flatten(self):
        # Pass array attributes as dynamic leaves so JIT does not treat full mesh
        # geometry as a single static Python object.
        children = []
        child_keys = []
        aux_data = {}

        for key, value in self.__dict__.items():
            if isinstance(value, (jax.Array, np.ndarray)):
                children.append(jnp.asarray(value))
                child_keys.append(key)
            else:
                aux_data[key] = value

        return tuple(children), (tuple(child_keys), aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        child_keys, aux = aux_data
        mesh = cls()
        mesh.__dict__.update(aux)
        for key, value in zip(child_keys, children):
            setattr(mesh, key, value)
        return mesh

    def mesh_generator(self, info = None, maxV = 5e-3, 
                       x_min = 0., x_max = 1., y_min = 0., y_max = 1.,
					    min_angle = 30, marker_boundary = 1):
        # info is used to create the geometry before passing to create the mesh
        # In the case there is no info, it is only a periodic square
        if info == None:
            Lx = x_max - x_min
            Ly = y_max - y_min
            N_maille_x = 2 * int(np.floor(np.sqrt(1/maxV)) * Lx / Ly)
            N_maille_y = 2 * int(np.floor(np.sqrt(1/maxV)) * Ly / Ly)

            boundaries = np.array([[x, y_min] for x in np.linspace(x_min, x_max, N_maille_x)][:-1])
            markers  = [marker_boundary ] * (N_maille_x - 1)
            boundaries = np.concatenate([boundaries, np.array([[x_max, y] for y in np.linspace(y_min, y_max, N_maille_y)][:-1])])
            markers.extend([marker_boundary] * (N_maille_y - 1))
            boundaries = np.concatenate([boundaries, np.array([[x, y_max] for x in np.linspace(x_max, x_min, N_maille_x)][:-1])])
            markers.extend([marker_boundary] * (N_maille_x - 1))
            boundaries = np.concatenate([boundaries, np.array([[x_min, y] for y in np.linspace(y_max, y_min, N_maille_y)][:-1])])
            markers.extend([marker_boundary] * (N_maille_y - 1))

            info = triangle.MeshInfo()
            info.set_points(boundaries)
            info.set_facets(self.round_trip_connect(0, len(boundaries)-1), facet_markers=markers)

        mesh = triangle.build(info, max_volume=maxV, min_angle=min_angle, generate_faces=True)

        self.points = jnp.array(mesh.points)
        self.tris = jnp.array(mesh.elements)
        self.neighbors = jnp.roll(jnp.array(mesh.neighbors), 1, axis=-1)  # For neighbors stuff, each side with v0-V1 is next to neighbor 0
        self.faces = jnp.array(mesh.faces)
        self.face_markers = jnp.array(mesh.face_markers)

        self.field_data = jnp.zeros_like(self.points[:,0])

        self.getCenterTriangles()
        self.getArea()
        self.getSurface()
        self.getNormals()
        self.get_face_connectivity()
        self.face_connectivity_opposite = self.face_connectivity.copy() 
        self.set_periodic_BC() # For periodic BCs, to ensure that neighbors are correctly set, (-1 by default)

        # BC by default
        self.inlet_supersonic = jnp.array([1.0, 1.0, 0.0, 1.0])  # rho, u, v, P
        self.inlet_subsonic = jnp.array([1.0, 1., 0.0, 1.0])  # rho, u, v, P

    def mesh_generator_from_points(self, mesh_points, mesh_tris, mesh_tris_neighbors, mesh_facets, mesh_face_markers):
        self.points = jnp.array(mesh_points)
        self.tris = jnp.array(mesh_tris)
        self.neighbors = jnp.array(mesh_tris_neighbors)  # For neighbors stuff, each side with v0-V1 is next to neighbor 0
        self.faces = jnp.array(mesh_facets)
        self.face_markers = jnp.array(mesh_face_markers)

        self.field_data = jnp.zeros_like(self.points[:,0])

        self.getCenterTriangles()
        self.getArea()
        self.getSurface()
        self.getNormals()
        self.get_face_connectivity()
        self.face_connectivity_opposite = self.face_connectivity.copy() 
        self.set_periodic_BC() 

        # BC by default
        self.inlet_supersonic = jnp.array([1.0, 1.0, 0.0, 1.0])  # rho, u, v, P
        self.inlet_subsonic = jnp.array([1.0, 1., 0.0, 1.0])  # rho, u, v, P

###################################################################
###################     Utilities    ##############################
###################################################################


    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)
        return self

    def round_trip_connect(self, start, end):
        result = []
        for i in range(start, end):
            result.append((i, i+1))
        result.append((end, start))
        return result

    def clean_boundaries(self, boundaries, markers):
        # Nettoie les points dupliqués dans les frontières pour éviter les segfaults de triangle
        unique_indices = []
        seen_points = {}
        
        for i, pt in enumerate(boundaries):
            key = tuple(np.round(pt, decimals=10))
            if key not in seen_points:
                seen_points[key] = i
                unique_indices.append(i)
        
        boundaries_clean = boundaries[unique_indices]
        markers_clean = [markers[i] for i in unique_indices]
        markers_clean.append(markers_clean[-1])
        
        return boundaries_clean, markers_clean[:len(boundaries_clean)]

    def print_statistics(self):
        x_min, x_max = self.points[:, 0].min(), self.points[:, 0].max()
        y_min, y_max = self.points[:, 1].min(), self.points[:, 1].max()
        Lx, Ly = x_max - x_min, y_max - y_min
        
        n_nodes = len(self.points)
        n_edges = len(self.faces)
        n_triangles = len(self.tris)
        
        areas = self.area
        avg_area = float(areas.mean())
        h_avg = (4 * avg_area / np.sqrt(3)) ** 0.5
        
        print("\n" + "-" * 70)
        print("STATISTIQUES DU MAILLAGE")
        print("-" * 70)
        
        # Info du domaine et maillage global
        print(f"Domaine: [{x_min:.4f}, {x_max:.4f}] x [{y_min:.4f}, {y_max:.4f}] | Taille: {Lx:.4f} x {Ly:.4f}")
        print(f"Mailles: {n_nodes} noeuds | {n_edges} aretes | {n_triangles} triangles | h_moyen: {h_avg:.4f}")
        
        # Stats par marqueur (afficher juste les edges pour éviter ambiguïté des noeuds partagés)
        print(f"\nMarqueurs:")
        face_markers_arr = np.array(self.face_markers)
        unique_markers = np.unique(face_markers_arr)
        
        for marker in sorted(unique_markers):
            mask = face_markers_arr == marker
            n_edges_marker = np.sum(mask)
            
            print(f"  Marqueur {marker}: {n_edges_marker:4d} aretes")
        
        print("\n" + "-" * 70 + "\n")

    def getCenterTriangles(self):
        self.barycenter = jnp.mean(self.points[self.tris], axis=-2)

    def getNormals(self):
        diff = jnp.roll(self.points[self.tris], -1, axis = -2) - self.points[self.tris]
        norm = jnp.linalg.norm(diff, axis=-1)
        normals = jnp.stack([diff[...,1], -diff[...,0]], -1)
        self.normals = normals / norm[..., None] # For neighbors stuff

    def getArea(self):
        points = self.points[self.tris]
        area = 0.5 * ((points[...,1,1] - points[...,0,1]) * (points[...,2,0] - points[...,0,0]) - 
                            (points[...,1,0] - points[...,0,0]) * (points[...,2,1] - points[...,0,1]))
        self.area = jnp.abs(area)
        del area, points

    def getSurface(self):
        self.surface = jnp.linalg.norm(self.points[self.faces[:,1]] - self.points[self.faces[:,0]], axis=1)

    def get_face_connectivity(self):
        """ For each triangle, find the indices of the faces that form its edges.
            The order of the faces corresponds to the order of the edges:
            Edge 0: between vertex 0 and 1
            Edge 1: between vertex 1 and 2
            Edge 2: between vertex 2 and 0"""
        faces = np.asarray(self.faces, dtype=np.int32)
        tris = np.asarray(self.tris, dtype=np.int32)

        # Build a direct edge -> face index map once; this is preprocessing only.
        face_lookup = {tuple(sorted(face.tolist())): idx for idx, face in enumerate(faces)}

        face_connectivity = np.empty((tris.shape[0], 3), dtype=np.int32)
        for tri_idx, tri in enumerate(tris):
            face_connectivity[tri_idx, 0] = face_lookup[tuple(sorted((int(tri[0]), int(tri[1]))))]
            face_connectivity[tri_idx, 1] = face_lookup[tuple(sorted((int(tri[1]), int(tri[2]))))]
            face_connectivity[tri_idx, 2] = face_lookup[tuple(sorted((int(tri[2]), int(tri[0]))))]

        self.face_connectivity = jnp.asarray(face_connectivity, dtype=jnp.int32)

###################################################################
################     Boundary conditions    #######################
###################################################################
    
    def set_periodic_BC(self, tol=3e-7):
        if self.points.dtype == jnp.float64:
            tol = 1e-10
        # Domain bounds
        x_min, x_max = self.points[:, 0].min(), self.points[:, 0].max()
        y_min, y_max = self.points[:, 1].min(), self.points[:, 1].max()
        dx, dy = x_max - x_min, y_max - y_min

        # Boundary faces
        faces_mask = self.face_markers == 1
        boundary_face_ids = jnp.where(faces_mask)[0]
        
        # Skip if no periodic BCs
        if len(boundary_face_ids) == 0:
            self.face_connectivity_opposite = jnp.arange(self.faces.shape[0])
            return

        diff = self.points[self.faces[boundary_face_ids]][:,1,:] - self.points[self.faces[boundary_face_ids]][:,0,:]

        # Classify boundaries
        is_left = (jnp.abs(diff[:, 0]) < tol) & (jnp.abs(self.points[self.faces[boundary_face_ids]][:, 0, 0] - x_min) < tol)
        is_right = (jnp.abs(diff[:, 0]) < tol) & (jnp.abs(self.points[self.faces[boundary_face_ids]][:, 0, 0] - x_max) < tol)
        is_bottom = (jnp.abs(diff[:, 1]) < tol) & (jnp.abs(self.points[self.faces[boundary_face_ids]][:, 0, 1] - y_min) < tol)
        is_top = (jnp.abs(diff[:, 1]) < tol) & (jnp.abs(self.points[self.faces[boundary_face_ids]][:, 0, 1] - y_max) < tol)

        # Compute shifts
        shifts = (is_left[:, None] * jnp.array([dx, 0.]) +  
                    is_right[:, None] * jnp.array([-dx, 0.]) +
                    is_bottom[:, None] * jnp.array([0., dy]) +
                    is_top[:, None] * jnp.array([0., -dy]))

        opposite_points = self.points[self.faces[boundary_face_ids]] + shifts[:, None, :]

        # Distance matrices
        dist = jnp.linalg.norm(
            self.points[self.faces[boundary_face_ids]][:, None, :, :] - opposite_points[None, :, :, :],
            axis=-1
        )  # (N_b, N_b, 2)

        dist_shifted = jnp.linalg.norm(
            self.points[self.faces[boundary_face_ids]][:, None, :, :] - jnp.roll(opposite_points[None, :, :, :], 1, axis=-2),
            axis=-1
        )

        # One point must match
        id_f_opposite = jnp.where((jnp.sum(dist < tol, axis = -1) == 2) | (jnp.sum(dist_shifted < tol, axis = -1) == 2))[1]

        # update neighbors
        for i, boundary_face_id in enumerate(boundary_face_ids):
            id_tri = jnp.where(boundary_face_id == self.face_connectivity)[0]
            id_opposite_tri = jnp.where(boundary_face_ids[id_f_opposite[i]] == self.face_connectivity)[0]
            self.neighbors = self.neighbors.at[id_tri,jnp.where(self.face_connectivity[id_tri] == boundary_face_id)[1]].set(id_opposite_tri)
        
        # knowing a face on edge, it get the corresponding face on the opposite edge
        self.face_connectivity_opposite = jnp.arange(self.faces.shape[0])  # (N_faces,) initialized to identity
        self.face_connectivity_opposite = self.face_connectivity_opposite.at[boundary_face_ids].set(boundary_face_ids[id_f_opposite])

###################################################################
###################     mesh I/O (numpy)  ########################
###################################################################

    def save_mesh(self, filename="mesh.npy"):
        # Sauvegarde le maillage format numpy avec pickle
        if not filename.endswith('.npy'):
            filename = filename.replace('.npz', '.npy').replace('.vtk', '.npy')
            if not filename.endswith('.npy'):
                filename += '.npy'
        
        jnp.save(filename, self)
        print(f"Maillage sauvegardé dans {filename}")

    def load_mesh(self, filename="mesh.npy"):
        # Charge le maillage sauvegardé avec pickle
        if not filename.endswith('.npy'):
            filename = filename.replace('.npz', '.npy').replace('.vtk', '.npy')
            if not filename.endswith('.npy'):
                filename += '.npy'
        
        mesh_loaded = jnp.load(filename, allow_pickle=True).item()
        
        # Copie les attributs du maillage chargé
        self.__dict__.update(mesh_loaded.__dict__)
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        legacy_keys = ("h", "Lx", "Ly", "obstacle_length", "chord", "height", "cx", "cy")
        for key in legacy_keys:
            if key not in self.metadata and hasattr(self, key):
                self.metadata[key] = getattr(self, key)
        if "h" not in self.metadata and hasattr(self, "h"):
            self.metadata["h"] = self.h
        if "obstacle_length" not in self.metadata and hasattr(self, "chord"):
            self.metadata["obstacle_length"] = self.chord
        
        print(f"Maillage chargé depuis {filename}")
        print(f"  Points: {self.points.shape}")
        print(f"  Triangles: {self.tris.shape}")
        print(f"  Faces: {self.faces.shape}")

    def export_vtk(self, filename="mesh.vtk"):
        # Exporte le maillage au format VTK pour visualisation avec Paraview
        import time
        start_time = time.time()
        
        # Convertir en tableaux numpy
        points = np.array(self.points, dtype=np.float64)
        triangles = np.array(self.tris, dtype=np.int32)
        faces = np.array(self.faces, dtype=np.int32)
        face_markers = np.array(self.face_markers, dtype=np.int32)
        
        # Écrire le fichier VTK en ASCII legacy format
        with open(filename, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 2.0\n")
            f.write("Generated by GenerativePDE\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n\n")
            
            # Points
            f.write(f"POINTS {len(points)} double\n")
            for pt in points:
                f.write(f"{pt[0]:.16e} {pt[1]:.16e} 0.0\n")
            f.write("\n")
            
            # Cells (triangles + lines)
            n_cells = len(triangles) + len(faces)
            cell_list_size = len(triangles) * 4 + len(faces) * 3 
            f.write(f"CELLS {n_cells} {cell_list_size}\n")
            
            # Triangles (type 5)
            for tri in triangles:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
            
            # Lines/Edges (type 3)
            for edge in faces:
                f.write(f"2 {edge[0]} {edge[1]}\n")
            f.write("\n")
            
            # Cell types
            f.write(f"CELL_TYPES {n_cells}\n")
            for _ in range(len(triangles)):
                f.write("5\n")  # VTK_TRIANGLE
            for _ in range(len(faces)):
                f.write("3\n")  # VTK_LINE
            f.write("\n")
            
            # Cell data - boundary markers
            f.write(f"CELL_DATA {n_cells}\n")
            f.write("SCALARS boundary_marker int\n")
            f.write("LOOKUP_TABLE default\n")
            for _ in range(len(triangles)):
                f.write("0\n")
            for marker in face_markers:
                f.write(f"{int(marker)}\n")
        
        elapsed = time.time() - start_time
        print(f"Maillage exporté vers {filename} ({elapsed:.4f} secondes)")

###################################################################
###################     plotting     ##############################
###################################################################

    def plot_mesh(self, *args, **kwargs):
        plot.plot_mesh(self, *args, **kwargs)

    def plot_solution(self, field_data, *args, **kwargs):
        plot.plot_solution(self, field_data, *args, **kwargs)
    
    def plot_contour_solution(self, field_data, *args, **kwargs):
        plot.plot_contour_solution(self, field_data, *args, **kwargs)

    def animate_field(self, field_sequence, path = "animation.gif", interval=100):
        plot.animate_solution(self, field_sequence, path = path, interval=interval)

    def plot_slice(self, X, y = 0.5, n = 1000, labels = r'$\rho$'):
        out, slice = plot.getSliceinMesh(X, self, y = y, n = n)
        fig, ax = plt.subplots()
        ax.plot(slice[...,0], out)
        ax.grid()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(labels)


if __name__ == "__main__":
    mesh = Mesh()
    mesh.mesh_generator(maxV=1e-2, marker_boundary=1, x_min=-1., x_max=1., y_min=-1, y_max=1.)
    mesh.plot_mesh()



