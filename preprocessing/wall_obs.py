"""Extraction des observations de bord (paroi) pour FAMWall.

La geometrie (position, normale, parametrisation le long du contour) ne depend
que du maillage, calculee UNE FOIS par geometrie via build_wall_layout(), puis
reutilisee pour tous les snapshots Mach/AoA (memes points physiques observes
partout, cf. extract_wall_values()).
"""
from __future__ import annotations

import numpy as np

from utils.aero import build_wall_cache


def _wall_face_geometry(mesh) -> dict:
    """Geometrie des faces de paroi (face_markers==2), meme ordre que
    build_wall_cache().wall_cell_ids (toutes deux derivees de
    wall_ids = where(face_markers==2)).
    """
    face_markers = np.asarray(mesh.face_markers)
    faces = np.asarray(mesh.faces)
    pts = np.asarray(mesh.points, dtype=np.float64)

    wall_ids = np.where(face_markers == 2)[0]
    ep = faces[wall_ids]
    a, b = pts[ep[:, 0]], pts[ep[:, 1]]
    mid = (a + b) / 2.0

    c = mesh.metadata['center']
    center = np.array([c['cx'], c['cy']])
    theta = np.arctan2(mid[:, 1] - center[1], mid[:, 0] - center[0])
    s = ((theta + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)).astype(np.float32)

    seg_vec = b - a
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_dir = seg_vec / np.maximum(seg_len, 1e-12)[:, None]

    return dict(wall_ids=wall_ids, mid=mid, s=s, seg_dir=seg_dir)


def _turning_angle(s: np.ndarray, seg_dir: np.ndarray, smooth_frac: float = 0.02) -> np.ndarray:
    """Courbure discrete (angle de virage entre segments consecutifs) le long
    du contour, ordre angulaire (s croissant) = ordre de parcours pour un
    contour en etoile. Retourne un array aligne sur l'ordre D'ORIGINE de s
    (pas l'ordre trie).
    """
    order = np.argsort(s)
    d = seg_dir[order]
    d_next = np.roll(d, -1, axis=0)
    cosang = np.clip((d * d_next).sum(axis=1), -1.0, 1.0)
    turn_sorted = np.arccos(cosang)  # in [0, pi], 0 = tout droit

    n = len(turn_sorted)
    w = max(3, int(round(smooth_frac * n)))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    padded = np.concatenate([turn_sorted[-(w // 2):], turn_sorted, turn_sorted[:w // 2]])
    turn_smooth_sorted = np.convolve(padded, kernel, mode='valid')

    turn = np.empty_like(turn_smooth_sorted)
    turn[order] = turn_smooth_sorted
    return turn.astype(np.float32)


def _quantile_subsample(density: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    """Choisit n indices (dans le referentiel de `density`/`s`) en inversant la
    CDF cumulee de `density` le long du contour trie par `s` -- plus `density`
    est grande a un endroit, plus les points y sont concentres. `density`
    uniforme => echantillonnage uniforme en position angulaire.
    """
    order = np.argsort(s)
    dens = np.maximum(density[order], 1e-6)
    cdf = np.concatenate([[0.0], np.cumsum(dens)])
    cdf /= cdf[-1]
    targets = (np.arange(n) + 0.5) / n
    pos = np.searchsorted(cdf, targets, side='right') - 1
    pos = np.clip(pos, 0, len(order) - 1)
    return order[pos]


def build_wall_layout(mesh, n_wall: int, strategy: str = 'curvature', eps: float = 0.15) -> dict:
    """Calcule UNE FOIS par geometrie/maillage HR les N_wall points de bord
    observes : indices de cellule (pour extraire les valeurs par snapshot,
    cf. extract_wall_values), position, normale, parametrisation s.

    strategy:
      'curvature' (defaut) : densifie pres des zones de forte courbure
          (bord d'attaque/de fuite, sommets du diamant) -- densite ~ eps + |virage|.
      'uniform'  : espacement constant en position angulaire s, baseline de
          comparaison.
    """
    wc = build_wall_cache(mesh)
    geo = _wall_face_geometry(mesh)
    n_faces = len(geo['wall_ids'])

    if n_wall >= n_faces:
        sel = np.arange(n_faces)  # pas assez de faces : on les garde toutes, rien a completer
    else:
        if strategy == 'uniform':
            density = np.ones(n_faces)
        elif strategy == 'curvature':
            kappa = _turning_angle(geo['s'], geo['seg_dir'])
            density = eps + kappa / (kappa.mean() + 1e-9)
            density = np.minimum(density, 10.0 * eps + 10.0)  # garde-fou : evite qu'un seul pic residuel n'avale tous les quantiles
        else:
            raise ValueError(f"wall_strategy inconnue: {strategy!r} (attendu 'curvature' ou 'uniform')")

        sel = np.unique(_quantile_subsample(density, geo['s'], n_wall))
        if len(sel) < n_wall:
            # la quantile-inversion peut ponctuellement repeter un indice (zone de tres forte
            # densite concentree sur peu de faces) -- complete avec les faces de plus forte
            # densite encore non retenues, pour garantir exactement n_wall points (N_wall doit
            # rester statique par branche pour le jit shape-statique de SRModel.fit).
            remaining = np.setdiff1d(np.arange(n_faces), sel)
            fill = remaining[np.argsort(-density[remaining])[:n_wall - len(sel)]]
            sel = np.sort(np.concatenate([sel, fill]))
    cell_ids = wc.wall_cell_ids[sel]
    normal = np.stack([wc.nx[sel], wc.ny[sel]], axis=1)

    return dict(
        cell_ids=cell_ids.astype(np.int32),
        pos=geo['mid'][sel].astype(np.float32),
        normal=normal.astype(np.float32),
        s=geo['s'][sel].astype(np.float32),
        n_wall=len(sel),
        strategy=strategy,
    )


def extract_wall_values(wall_layout: dict, hr_primitives: np.ndarray) -> np.ndarray:
    """Valeurs physiques vraies (rho,u,v,p) aux points de bord retenus, pour un
    snapshot HR donne -- extraction pure, aucun calcul, reutilise l'indexation
    figee par build_wall_layout()."""
    return hr_primitives[wall_layout['cell_ids']].astype(np.float32)


def extract_wall_observations(mesh, hr_primitives: np.ndarray, n_wall: int,
                              strategy: str = 'curvature') -> dict:
    """Convenience : layout + valeurs en un seul appel (usage one-off / sanity
    check). Pour le preprocessing en batch (nombreux snapshots d'une meme
    geometrie), preferer build_wall_layout() une fois + extract_wall_values()
    par fichier -- evite de refaire le sous-echantillonnage a chaque snapshot.
    """
    layout = build_wall_layout(mesh, n_wall, strategy)
    value = extract_wall_values(layout, hr_primitives)
    return dict(pos=layout['pos'], normal=layout['normal'], value=value, s=layout['s'])
