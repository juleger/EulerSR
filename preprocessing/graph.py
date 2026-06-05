import numpy as np
from pathlib import Path

"""
Sauvegarde et chargement du graphe de voisinage du maillage, utilisé pour le GNN. Chaque noeud correspond à un centroid du maillage, et les arêtes connectent les noeuds voisins. Les features d'arête incluent les différences de coordonnées, les normales, etc.
"""

def build(mesh, save_path):
    N = np.asarray(mesh.barycenter).shape[0]
    barycenter = np.asarray(mesh.barycenter)
    area = np.asarray(mesh.area)
    neighbors = np.asarray(mesh.neighbors)
    normals = np.asarray(mesh.normals)
    surface = np.asarray(mesh.surface)
    face_conn = np.asarray(mesh.face_connectivity)

    i_idx = np.repeat(np.arange(N), 3)
    k_idx = np.tile(np.arange(3), N)
    j_idx = neighbors.ravel()
    mask = j_idx >= 0
    i_src = i_idx[mask]
    j_dst = j_idx[mask]
    k_face = k_idx[mask]

    dx = barycenter[j_dst, 0] - barycenter[i_src, 0]
    dy = barycenter[j_dst, 1] - barycenter[i_src, 1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    nx = normals[i_src, k_face, 0]
    ny = normals[i_src, k_face, 1]
    fl = surface[face_conn[i_src, k_face]]

    edge_index = np.stack([i_src, j_dst], axis=0).astype(np.int32)
    edge_attr = np.stack([dx, dy, dist, nx, ny, fl], axis=1).astype(np.float32)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path,
             centroids=barycenter.astype(np.float32),
             log_area=np.log(area).astype(np.float32),
             edge_index=edge_index,
             edge_attr=edge_attr)
    print(f"Graphe exporté: {save_path}  ({N} noeuds, {edge_index.shape[1]} aretes)")


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}
