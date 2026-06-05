import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

"""
Module de construction du voisinage kNN entre les centroides du maillage HR avec les centroides du maillage LR, utilisé pour l'interpolation IDW. Les indices des k plus proches voisins et leurs distances sont sauvegardés pour une utilisation rapide lors de l'entraînement des modèles de super-resolution.
"""

def build(fine_centroids, coarse_centroids, k, save_path):
    tree = cKDTree(coarse_centroids)
    dists, indices = tree.query(fine_centroids, k=k, workers=-1)
    rel_pos = coarse_centroids[indices] - fine_centroids[:, None, :]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path,
             indices=indices.astype(np.int32),
             rel_pos=rel_pos.astype(np.float32),
             dists=dists.astype(np.float32))
    print(f"kNN exporté: {save_path}  (N_fine={len(fine_centroids)}, k={k})")


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}
