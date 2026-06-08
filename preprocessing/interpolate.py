import numpy as np


def lr_on_hr(d: dict, knn: dict) -> np.ndarray:
    # Interpole les primitives LR sur les positions HR à partir des K plus proches voisins par IDW
    lr = d['lr_primitives'].astype(np.float32)
    w  = 1.0 / (np.asarray(knn['dists']) + 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    return (w[:, :, None] * lr[np.asarray(knn['indices'])]).sum(axis=1)


def load_lr_on_hr(path: str, knn: dict) -> np.ndarray:
    return lr_on_hr(np.load(path), knn).ravel()
