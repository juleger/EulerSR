"""Construction du voisinage kNN LR->HR, self-kNN HR->HR et opérateur gradient LSQ."""
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

_K_SELF = 4  # voisins HR->HR pour la reconstruction du gradient


def build_grad_op(rel_pos: np.ndarray, dists: np.ndarray) -> np.ndarray:
    # Opérateur de grad LSQ pour reconstruction des gradients
    w = 1.0 / (dists**2 + 1e-12)  # (N, k)
    R = rel_pos  # (N, k, 2)
    Rw = R * w[:, :, None]  # (N, k, 2)
    A = np.einsum('nki,nkj->nij', Rw, R)  # (N, 2, 2) = R^T W R
    RtW = Rw.transpose(0, 2, 1)  # (N, 2, k) = R^T W
    G = np.linalg.solve(A, RtW)  # (N, 2, k)
    return G.astype(np.float32)


def build(fine_centroids: np.ndarray, coarse_centroids: np.ndarray,
          k: int, save_path) -> None:
    """Construit le kNN LR->HR (coarse->fine) et le self-kNN HR->HR avec opérateur gradient."""
    # kNN LR->HR
    tree_lr = cKDTree(coarse_centroids)
    dists, indices = tree_lr.query(fine_centroids, k=k, workers=-1)
    rel_pos = coarse_centroids[indices] - fine_centroids[:, None, :]

    # Self-kNN HR->HR pour l'opérateur gradient LSQ
    tree_hr = cKDTree(fine_centroids)
    sd, si = tree_hr.query(fine_centroids, k=_K_SELF + 1, workers=-1)
    hr_idx = si[:, 1:].astype(np.int32)  # exclure le point lui-même (dist=0)
    hr_dist = sd[:, 1:].astype(np.float32)
    hr_rel = (fine_centroids[hr_idx] - fine_centroids[:, None, :]).astype(np.float32)
    grad_op = build_grad_op(hr_rel, hr_dist)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, indices=indices.astype(np.int32), rel_pos=rel_pos.astype(np.float32), dists=dists.astype(np.float32),
            hr_idx=hr_idx, hr_rel=hr_rel, hr_dist=hr_dist, grad_op=grad_op)
    print(f"kNN exporté: {save_path}  "
          f"(N_fine={len(fine_centroids)}, k={k}, k_self={_K_SELF})")


def build_hierarchy(lvl_pos: list, lr_pos: np.ndarray, k_pool: int, k_up: int, k_self: int, k_cond: int) -> dict:
    # Tables kNN entre niveaux + PE Fourier + poids IDW, rel_pos normalisées par l'échelle locale.
    import sys
    from pathlib import Path as _Path
    import jax.numpy as jnp
    _root = str(_Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from utils.attention import fourier_pos_enc

    pts = np.concatenate([lvl_pos[0], lr_pos], axis=0)
    ctr = (pts.max(0) + pts.min(0)) / 2
    scl = float((pts.max(0) - pts.min(0)).max() / 2)

    z = {}
    tree_lr = cKDTree(lr_pos)
    for l, p in enumerate(lvl_pos):
        pn = ((p - ctr) / scl).astype(np.float32)
        z[f'pe{l}'] = np.asarray(fourier_pos_enc(jnp.array(pn)))
        dist, idx = tree_lr.query(p, k=k_cond, workers=-1)
        w = 1.0 / (dist ** 2 + 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        z[f'cond{l}_idx'] = idx.astype(np.int32)
        z[f'cond{l}_w'] = w.astype(np.float32)

    for l in range(len(lvl_pos) - 1):
        fine, coarse = lvl_pos[l], lvl_pos[l + 1]
        dist, idx = cKDTree(fine).query(coarse, k=k_pool, workers=-1)
        rel = (fine[idx] - coarse[:, None, :]) / dist.mean(1)[:, None, None]
        z[f'down{l}_idx'] = idx.astype(np.int32)
        z[f'down{l}_rel'] = rel.astype(np.float32)
        dist, idx = cKDTree(coarse).query(fine, k=k_up, workers=-1)
        rel = (coarse[idx] - fine[:, None, :]) / dist.mean(1)[:, None, None]
        z[f'up{l}_idx'] = idx.astype(np.int32)
        z[f'up{l}_rel'] = rel.astype(np.float32)

    d0, i0 = cKDTree(lvl_pos[0]).query(lvl_pos[0], k=k_self + 1, workers=-1)
    idx = i0[:, 1:]
    rel = (lvl_pos[0][idx] - lvl_pos[0][:, None, :]) / d0[:, 1:].mean(1)[:, None, None]
    z['self_idx'] = idx.astype(np.int32)
    z['self_rel'] = rel.astype(np.float32)
    return z
