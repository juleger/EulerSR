"""Métriques d'erreur pour la super-résolution CFD et les grandeurs aérodynamiques (CL, CD, Mach paroi)."""
from __future__ import annotations
import numpy as np
from utils.refs import to_mach

_PRIM_NAMES = ['rho', 'u', 'v', 'p']


def idw_weights(dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Poids IDW normalisés (p=2) à partir des distances kNN (N, k)"""
    d = np.asarray(dist, dtype=np.float64)
    w = 1.0 / (d ** 2 + eps)
    w /= w.sum(axis=1, keepdims=True)
    return w


def l2_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.sum((pred - ref) ** 2) / (np.sum(ref ** 2) + 1e-8)))


def linf_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(pred - ref)) / (np.mean(np.abs(ref)) + 1e-8))


def l2w_rel(pred: np.ndarray, ref: np.ndarray, w: np.ndarray) -> float:
    # L2 pondérée relative (ex : w = |grad p| par cellule)
    w = w / (w.sum() + 1e-12)
    return float(np.sqrt(np.sum(w * (pred - ref) ** 2) / (np.sum(w * ref ** 2) + 1e-8)))


def w2(pred: np.ndarray, ref: np.ndarray, seed: int = 0) -> float:
    # W2 Wasserstein entre deux champs scalaires (sort-based, O(N log N))
    rng = np.random.default_rng(seed)
    p = pred.ravel().astype(np.float64)
    r = ref.ravel().astype(np.float64)
    if len(p) != len(r):
        n = min(len(p), len(r))
        p = rng.choice(p, n, replace=False)
        r = rng.choice(r, n, replace=False)
    return float(np.sqrt(np.mean((np.sort(p) - np.sort(r)) ** 2)))


def compute_field_errors(prim_pred: np.ndarray, prim_ref: np.ndarray, cell_adj_edges: np.ndarray,
            bary: np.ndarray, dxy: np.ndarray | None = None) -> dict[str, float]:
    # L2, Linf, L2w (pondérée |grad p|) et W2 sur Mach et les 4 primitives.
    mach_p = to_mach(prim_pred)
    mach_r = to_mach(prim_ref)

    dp = np.abs(prim_ref[cell_adj_edges[:, 0], 3] -
                prim_ref[cell_adj_edges[:, 1], 3])
    if dxy is None:
        dxy = np.linalg.norm(bary[cell_adj_edges[:, 0]] -
                             bary[cell_adj_edges[:, 1]], axis=1)
    ge = (dp / (dxy + 1e-12)).astype(np.float32)
    n_cells = prim_ref.shape[0]
    w = (np.bincount(cell_adj_edges[:, 0], weights=ge, minlength=n_cells) +
         np.bincount(cell_adj_edges[:, 1], weights=ge, minlength=n_cells)).astype(np.float32)

    res: dict[str, float] = {
        'l2_mach': l2_rel(mach_p, mach_r),
        'linf_mach': linf_rel(mach_p, mach_r),
        'l2w_mach': l2w_rel(mach_p, mach_r, w),
        'w2_mach': w2(mach_p, mach_r),
    }
    for i, nm in enumerate(_PRIM_NAMES):
        res[f'l2_{nm}'] = l2_rel(prim_pred[:, i], prim_ref[:, i])
        res[f'linf_{nm}'] = linf_rel(prim_pred[:, i], prim_ref[:, i])
    return res


def aero_metrics(prim_pred: np.ndarray, prim_ref: np.ndarray, wc, mach_in: float) -> dict:
    # Erreurs aérodynamiques prédiction vs GT (CL, CD, Mach paroi)
    # Import local pour éviter la dépendance circulaire avec utils.aero
    from utils.aero import aero_coeffs
    ac_p = aero_coeffs(prim_pred, wc, mach_in)
    ac_r = aero_coeffs(prim_ref, wc, mach_in)
    wall_mach_l2 = float(np.linalg.norm(ac_p['wall_mach'] - ac_r['wall_mach']) /
                         (np.linalg.norm(ac_r['wall_mach']) + 1e-8))
    wall_cp_l2 = float(np.linalg.norm(ac_p['wall_cp'] - ac_r['wall_cp']) /
                       (np.linalg.norm(ac_r['wall_cp']) + 1e-8))
    return {
        'CL_pred': float(ac_p['CL']),
        'CL_ref': float(ac_r['CL']),
        'CD_pred': float(ac_p['CD']),
        'CD_ref': float(ac_r['CD']),
        'CL_err_abs': abs(ac_p['CL'] - ac_r['CL']),
        'CD_err_abs': abs(ac_p['CD'] - ac_r['CD']),
        'wall_mach_l2': wall_mach_l2,
        'wall_mach_pred': ac_p['wall_mach'],
        'wall_mach_ref': ac_r['wall_mach'],
        'wall_cp_l2': wall_cp_l2,
        'wall_cp_pred': ac_p['wall_cp'],
        'wall_cp_ref': ac_r['wall_cp'],
    }


def euler_scales(mu: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    # Échelles de référence par équation pour adimensionner les résidus
    scale = np.array([
        mu[0] * mu[1],
        mu[0] * mu[1]**2 + mu[3],
        mu[0] * mu[2]**2 + mu[3],
        (mu[3] / (gamma - 1) + 0.5 * mu[0] * (mu[1]**2 + mu[2]**2) + mu[3]) * mu[1],
    ], dtype=np.float64)
    return np.maximum(np.abs(scale), 1e-4)


_P_INF = 1.0
_RHO_INF = 1.0


def enthalpy_rms(prim: np.ndarray, mach_in: float,
                 mask: np.ndarray | None = None, gamma: float = 1.4) -> float:
    """RMS relatif de H_tot par rapport à H_inf (= 0 pour solution Euler parfaite)"""
    H = gamma / (gamma - 1) * prim[:, 3] / np.maximum(prim[:, 0], 1e-12) + 0.5 * (prim[:, 1]**2 + prim[:, 2]**2)
    v_inf = mach_in * np.sqrt(gamma * _P_INF / _RHO_INF)
    H_inf = gamma / (gamma - 1) * _P_INF / _RHO_INF + 0.5 * v_inf**2
    dH = ((H - H_inf) / H_inf) ** 2
    if mask is not None:
        dH = dH[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(dH.mean()))


_INTERIOR_MASK_CACHE: dict[int, np.ndarray] = {}


def interior_cell_mask(mesh) -> np.ndarray:
    """Masque booléen (N_cells,) : True pour les cellules purement intérieures"""
    key = id(mesh)
    cached = _INTERIOR_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    fm = np.asarray(mesh.face_markers)
    fc = np.asarray(mesh.face_connectivity).astype(np.int64)  # (N_cells, n_faces) -> face ids
    touches_bnd = (fm[fc] != 0).any(axis=1)
    mask = ~touches_bnd
    _INTERIOR_MASK_CACHE[key] = mask
    return mask


def fvm_euler_rms(prim_phys: np.ndarray, mesh, mach_in: float, aoa_deg: float,
                  mu: np.ndarray, gamma: float = 1.4,
                  reconstruction: str = 'muscl', flux: str = 'hllc',
                  interior_only: bool = True) -> float:
    """Résidu Euler FVM exact (même opérateur que le solveur), normalisé par euler_scales(mu)."""
    import sys
    from pathlib import Path
    _root = str(Path(__file__).resolve().parents[1])
    for d in (_root + '/euler', _root):
        if d not in sys.path:
            sys.path.insert(0, d)
    import jax.numpy as jnp
    from jax_fvm.src.helper import getConserved
    from jax_fvm.src.euler_solver import residual as _fvm_res

    W = getConserved(jnp.array(prim_phys, dtype=jnp.float32), gamma=gamma, M=1.0)
    c_inf = float(np.sqrt(gamma))
    aoa_rad = float(np.deg2rad(aoa_deg))
    prim_inf = jnp.array([1.0, mach_in * c_inf * np.cos(aoa_rad),
                           mach_in * c_inf * np.sin(aoa_rad), 1.0])
    inlet = getConserved(prim_inf[None], gamma=gamma, M=1.0)[0]

    res = np.asarray(_fvm_res(W, mesh, gamma=gamma, M=1.0,
                              reconstruction=reconstruction, flux=flux,
                              value=inlet, entropy=False))
    resn = res / euler_scales(mu, gamma)[None, :]
    if interior_only:
        resn = resn[interior_cell_mask(mesh)]
    return float(np.sqrt((resn ** 2).mean()))


def entropy_violation(prim: np.ndarray, gamma: float = 1.4) -> float:
    # RMS de s < s_inf
    s = prim[:, 3] / np.maximum(prim[:, 0], 1e-12) ** gamma
    s_inf = _P_INF / _RHO_INF ** gamma
    viol = np.minimum(s - s_inf, 0.0) / s_inf
    return float(np.sqrt((viol ** 2).mean()))
