import sys
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / 'euler'))
try:
    import euler.jax_fvm.src.mesh  # noqa: F401
except ModuleNotFoundError:
    pass

_PROC_RE = re.compile(r'^aoa([+-]?\d+\.\d+)_m(\d+\.\d+)$')
VAR_CMAPS = ['viridis', 'RdBu_r', 'RdBu_r', 'plasma']


def load_lr_on_hr(path: str, knn: dict) -> np.ndarray:
    # Charge les primitives LR et interpole sur les centroides HR par IDW 
    d = np.load(path)
    lr_prim = d['lr_primitives'].astype(np.float32)
    w = 1.0 / (np.asarray(knn['dists']) + 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    return (w[:, :, None] * lr_prim[np.asarray(knn['indices'])]).sum(axis=1).ravel()


def idw(lr_field: np.ndarray, knn_indices: np.ndarray,
        knn_dists: np.ndarray, power: float = 2) -> np.ndarray:
    """ Interpolation par inverse distance weighting : les centroides HR sont estimés par moyenne pondérée des centroides LR (inversement prop à la distance) """
    w = 1.0 / (knn_dists.astype(np.float64) ** power + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    return (w[:, :, None] * lr_field[knn_indices]).sum(axis=1).astype(np.float32)

