"""Coefficients aérodynamiques depuis un champ CFD cell-centré (face_markers == 2 : paroi)."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from utils.refs import to_mach, parse_case

GAMMA = 1.4
_P_INF = 1.0
_RHO_INF = 1.0


@dataclass
class WallCache:
    """Données paroi précalculées une fois par maillage HR."""
    wall_cell_ids: np.ndarray        # cellule FVM adjacente à chaque face paroi
    nx: np.ndarray                   # normale sortante (de la cellule fluide)
    ny: np.ndarray
    ds: np.ndarray                   # longueur de la face
    chord: float
    unique_wall_cell_ids: np.ndarray
    bary: np.ndarray                 # (N_cells, 2) barycentres
    cell_adj_edges: np.ndarray       # paires de cellules adjacentes intérieures


def build_wall_cache(mesh) -> WallCache:
    """Construit WallCache depuis un objet Mesh (euler.jax_fvm.src.mesh.Mesh)."""
    face_markers = np.asarray(mesh.face_markers)
    face_connectivity = np.asarray(mesh.face_connectivity)
    normals = np.asarray(mesh.normals)
    surface = np.asarray(mesh.surface)
    bary = np.asarray(mesh.barycenter, dtype=np.float32)
    neighbors = np.asarray(mesh.neighbors)
    N_cells = bary.shape[0]

    wall_ids = np.where(face_markers == 2)[0]

    face_owner: dict[int, tuple[int, int]] = {}
    for cell_id, row in enumerate(face_connectivity):
        for lf, fid in enumerate(row):
            key = int(fid)
            if key not in face_owner:
                face_owner[key] = (cell_id, lf)

    cell_ids = np.array([face_owner[int(fid)][0] for fid in wall_ids], np.int32)
    local_faces = np.array([face_owner[int(fid)][1] for fid in wall_ids], np.int32)

    nx = normals[cell_ids, local_faces, 0].astype(np.float32)
    ny = normals[cell_ids, local_faces, 1].astype(np.float32)
    ds = surface[wall_ids].astype(np.float32)
    unique_wall_cell_ids = np.unique(cell_ids)

    adj: set[tuple[int, int]] = set()
    for ci in range(N_cells):
        for ni_raw in neighbors[ci]:
            ni = int(ni_raw)
            if 0 <= ni < N_cells:
                adj.add((min(ci, ni), max(ci, ni)))
    cell_adj_edges = np.array(sorted(adj), dtype=np.int32)

    chord = float(getattr(mesh, 'metadata', {}).get('chord', 1.0))
    return WallCache(
        wall_cell_ids=cell_ids,
        nx=nx, ny=ny, ds=ds,
        chord=chord,
        unique_wall_cell_ids=unique_wall_cell_ids,
        bary=bary,
        cell_adj_edges=cell_adj_edges,
    )


def q_inf(mach_in: float) -> float:
    # Pression dynamique de référence : 0.5 * rho_inf * U_inf^2.
    c_inf = float(np.sqrt(GAMMA * _P_INF / _RHO_INF))
    return 0.5 * _RHO_INF * (mach_in * c_inf) ** 2


def aero_coeffs(prim: np.ndarray, wc: WallCache, mach_in: float) -> dict:
    # CL, CD, grad_p_max, Mach paroi depuis prim
    qi = q_inf(mach_in)

    p_wall = prim[wc.wall_cell_ids, 3]
    CL = float(np.sum(p_wall * wc.ny * wc.ds)) / (qi * wc.chord)
    CD = float(np.sum(p_wall * wc.nx * wc.ds)) / (qi * wc.chord)

    dp = np.abs(prim[wc.cell_adj_edges[:, 0], 3] -
                prim[wc.cell_adj_edges[:, 1], 3])
    dxy = np.linalg.norm(wc.bary[wc.cell_adj_edges[:, 0]] -
                         wc.bary[wc.cell_adj_edges[:, 1]], axis=1)
    grad_p_max = float((dp / (dxy + 1e-12)).max())

    mach = to_mach(prim)
    wall_mach = mach[wc.unique_wall_cell_ids]

    return dict(
        CL=CL, CD=CD,
        grad_p_max=grad_p_max,
        mach_max=float(mach.max()),
        mach_min=float(mach.min()),
        mach_mean=float(mach.mean()),
        wall_mach=wall_mach.astype(np.float32),
    )


_MACH_MID, _MACH_SCALE, _AOA_SCALE = 2.35, 1.65, 5.0


def compute_aero_scalars(d: dict, path: str, wc_lr: WallCache | None = None) -> np.ndarray:
    # Vecteur 7-dim [mach_n, aoa_n, mach_max_LR, mach_min_LR, grad_p_max_LR, CL_LR, mach_wall_min_LR].
    mach_in, aoa_in = parse_case(path)
    lr = d['lr_primitives'].astype(np.float32)
    mach_lr = to_mach(lr)

    if 'lr_primitives_grad' in d:
        gp = np.linalg.norm(d['lr_primitives_grad'][:, 3, :].astype(np.float32), axis=-1)
        grad_p_max = float(gp.max())
    else:
        grad_p_max = float(np.abs(np.diff(lr[:, 3])).max() if len(lr) > 1 else 0.0)

    Cl, mach_wall_min = 0.0, 0.0
    if wc_lr is not None:
        try:
            coeffs = aero_coeffs(lr, wc_lr, mach_in)
            Cl = float(coeffs['CL'])
            mach_wall_min = float(coeffs['wall_mach'].min())
        except Exception:
            pass

    return np.array([
        float((mach_in - _MACH_MID) / _MACH_SCALE),
        float(aoa_in / _AOA_SCALE),
        float(mach_lr.max()),
        float(mach_lr.min()),
        grad_p_max,
        Cl,
        mach_wall_min,
    ], dtype=np.float32)
