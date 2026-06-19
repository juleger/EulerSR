"""Métriques d'erreur pour la super-résolution CFD et les grandeurs aérodynamiques (CL, CD, Mach paroi)."""
from __future__ import annotations
import numpy as np
from utils.refs import to_mach

_EULER_EQ_NAMES = ['euler_mass', 'euler_xmom', 'euler_ymom', 'euler_energy']
_PRIM_NAMES = ['rho', 'u', 'v', 'p']


def l2_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.sum((pred - ref) ** 2) / (np.sum(ref ** 2) + 1e-8)))


def linf_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(pred - ref)) / (np.mean(np.abs(ref)) + 1e-8))


def l2w_rel(pred: np.ndarray, ref: np.ndarray, w: np.ndarray) -> float:
    # L2 pondérée relative (ex : w = |grad p| par cellule)
    w = w / (w.sum() + 1e-12)
    return float(np.sqrt(np.sum(w * (pred - ref) ** 2) / (np.sum(w * ref ** 2) + 1e-8)))


def sliced_wasserstein(pred: np.ndarray, ref: np.ndarray, n_proj: int = 50, seed: int = 0) -> float:
    # W2 sliced entre deux champs scalaires (sort-based, O(N log N))
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
        'l2_mach':   l2_rel(mach_p, mach_r),
        'linf_mach': linf_rel(mach_p, mach_r),
        'l2w_mach':  l2w_rel(mach_p, mach_r, w),
        'w2_mach':   sliced_wasserstein(mach_p, mach_r),
    }
    for i, nm in enumerate(_PRIM_NAMES):
        res[f'l2_{nm}']   = l2_rel(prim_pred[:, i], prim_ref[:, i])
        res[f'linf_{nm}'] = linf_rel(prim_pred[:, i], prim_ref[:, i])
    return res


def euler_residuals(prim: np.ndarray, hr_idx: np.ndarray, grad_op: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    # Résidus Euler steady 2D sur prim (N, 4). Retourne (N, 4), doit valoir 0 pour solution exacte.
    rho, u, v, p = prim[:, 0], prim[:, 1], prim[:, 2], prim[:, 3]

    def nabla(f: np.ndarray) -> np.ndarray:
        delta = f[hr_idx] - f[:, None]
        return np.einsum('ndk,nk->nd', grad_op, delta)

    drho, du, dv, dp = nabla(rho), nabla(u), nabla(v), nabla(p)

    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    dE = (dp / (gamma - 1)
          + 0.5 * ((u**2 + v**2)[:, None] * drho
                   + rho[:, None] * (2*u[:, None]*du + 2*v[:, None]*dv)))
    H = E + p
    dH = dE + dp

    R1 = u*drho[:,0] + rho*du[:,0] + v*drho[:,1] + rho*dv[:,1]
    R2 = (u**2*drho[:,0] + 2*rho*u*du[:,0] + dp[:,0]
        + u*v*drho[:,1] + rho*v*du[:,1] + rho*u*dv[:,1])
    R3 = (u*v*drho[:,0] + rho*v*du[:,0] + rho*u*dv[:,0]
        + v**2*drho[:,1] + 2*rho*v*dv[:,1] + dp[:,1])
    R4 = u*dH[:,0] + H*du[:,0] + v*dH[:,1] + H*dv[:,1]

    return np.stack([R1, R2, R3, R4], axis=-1)


def euler_rms(prim: np.ndarray, hr_idx: np.ndarray, grad_op: np.ndarray, gamma: float = 1.4) -> dict[str, float]:
    res = euler_residuals(prim, hr_idx, grad_op, gamma)
    return {name: float(np.sqrt((res[:, i]**2).mean()))
            for i, name in enumerate(_EULER_EQ_NAMES)}


def aero_metrics(prim_pred: np.ndarray, prim_ref: np.ndarray, wc, mach_in: float) -> dict:
    # Erreurs aérodynamiques prédiction vs GT (CL, CD, Mach paroi)
    # Import local pour éviter la dépendance circulaire avec utils.aero
    from utils.aero import aero_coeffs
    ac_p = aero_coeffs(prim_pred, wc, mach_in)
    ac_r = aero_coeffs(prim_ref,  wc, mach_in)
    wall_mach_l2 = float(np.linalg.norm(ac_p['wall_mach'] - ac_r['wall_mach']) /
                         (np.linalg.norm(ac_r['wall_mach']) + 1e-8))
    return {
        'CL_pred':      float(ac_p['CL']),
        'CL_ref':       float(ac_r['CL']),
        'CD_pred':      float(ac_p['CD']),
        'CD_ref':       float(ac_r['CD']),
        'CL_err_abs':   abs(ac_p['CL'] - ac_r['CL']),
        'CD_err_abs':   abs(ac_p['CD'] - ac_r['CD']),
        'wall_mach_l2': wall_mach_l2,
        'wall_mach_pred': ac_p['wall_mach'],
        'wall_mach_ref':  ac_r['wall_mach'],
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


def euler_rms_norm(prim: np.ndarray, hr_idx: np.ndarray, grad_op: np.ndarray, mu: np.ndarray, gamma: float = 1.4) -> float:
    # RMS adimensionné moyen des 4 résidus Euler
    res = euler_residuals(prim, hr_idx, grad_op, gamma)
    return float(np.sqrt(((res / euler_scales(mu, gamma)) ** 2).mean()))


_P_INF = 1.0
_RHO_INF = 1.0


def enthalpy_rms(prim: np.ndarray, mach_in: float, gamma: float = 1.4) -> float:
    # RMS relatif de l'enthalpie totale par rapport à H_inf (constante pour Euler stationnaire)
    H = gamma / (gamma - 1) * prim[:, 3] / np.maximum(prim[:, 0], 1e-12) + 0.5 * (prim[:, 1]**2 + prim[:, 2]**2)
    v_inf = mach_in * np.sqrt(gamma * _P_INF / _RHO_INF)
    H_inf = gamma / (gamma - 1) * _P_INF / _RHO_INF + 0.5 * v_inf**2
    return float(np.sqrt((((H - H_inf) / H_inf) ** 2).mean()))


def entropy_violation(prim: np.ndarray, gamma: float = 1.4) -> float:
    # RMS de s < s_inf
    s = prim[:, 3] / np.maximum(prim[:, 0], 1e-12) ** gamma
    s_inf = _P_INF / _RHO_INF ** gamma
    viol = np.minimum(s - s_inf, 0.0) / s_inf
    return float(np.sqrt((viol ** 2).mean()))
