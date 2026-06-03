import numpy as np
from scipy.spatial import Delaunay, cKDTree
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from euler.jax_fvm.src.mesh import Mesh
import matplotlib.pyplot as plt


class InterpolationMapping:
    """ 
    Représente le mapping d'interpolation entre un maillage source (LR) et un maillage cible (HR).
    """

    def __init__(self, hr_mesh, lr_mesh, lr_field, hr_field):
        self.hr_mesh = hr_mesh
        self.lr_mesh = lr_mesh
        self.tri = Delaunay(lr_mesh.points)
        self.tree = cKDTree(lr_mesh.points)
        self.weights = None

    def interpolate(self, lr_field):

        hr_points = self.hr_mesh.points
        lr_points = self.lr_mesh.points
        _, idx = self.tree.query(hr_points)
        self.weights = lr_field[idx]

    def plot_mapping(self, lr_field, hr_field):
        # Plots montrant l'interpolation d'une solution avec reference LR, interpolation LR sur maillage HR, et solution HR
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.tricontourf(self.lr_mesh.points[:, 0], self.lr_mesh.points[:, 1], self.tri.simplices, lr_field, levels=50)
        plt.title("Champ LR")
        plt.colorbar() 
        plt.subplot(1, 3, 2)
        plt.tricontourf(self.hr_mesh.points[:, 0], self.hr_mesh.points[:, 1], self.tri.simplices, self.weights, levels=50)
        plt.title("Interpolation LR sur maillage HR")
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.tricontourf(self.hr_mesh.points[:, 0], self.hr_mesh.points[:, 1], self.tri.simplices, hr_field, levels=50)
        plt.title("Champ HR")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    lr_mesh = Mesh.load("data/diamond/mesh_0.5.npy")
    hr_mesh = Mesh.load("data/diamond/mesh_0.025.npy")
    lr_field = np.load("data/diamond/raw/



    